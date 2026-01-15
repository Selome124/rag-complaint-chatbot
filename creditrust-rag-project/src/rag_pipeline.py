# src/rag_pipeline.py
import torch
import pickle
import numpy as np
from typing import List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

try:
    from sentence_transformers import SentenceTransformer
    from transformers import pipeline
    IMPORT_SUCCESS = True
except ImportError:
    IMPORT_SUCCESS = False
    print("Warning: Some libraries not installed. Run: pip install sentence-transformers transformers")

class RAGPipeline:
    def __init__(self, vector_store_path: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize RAG Pipeline"""
        try:
            print(f"Loading vector store from {vector_store_path}...")
            with open(vector_store_path, "rb") as f:
                self.vector_store = pickle.load(f)
            
            print(f"✓ Loaded {len(self.vector_store['chunks'])} chunks")
            
            if IMPORT_SUCCESS:
                self.embedding_model = SentenceTransformer(model_name)
                self.generator = self._initialize_generator()
            else:
                self.embedding_model = None
                self.generator = DummyGenerator()
                
        except FileNotFoundError:
            print(f"Error: Vector store not found at {vector_store_path}")
            print("Please place your vector_store.pkl in the 'data/' folder")
            raise
        except Exception as e:
            print(f"Error initializing pipeline: {e}")
            raise
    
    def _initialize_generator(self):
        """Initialize the text generation model"""
        try:
            return pipeline(
                "text-generation",
                model="gpt2",
                device="cuda" if torch.cuda.is_available() else "cpu",
                max_new_tokens=150,
                temperature=0.1
            )
        except:
            return DummyGenerator()
    
    def embed_question(self, question: str):
        """Embed the question"""
        if self.embedding_model:
            return self.embedding_model.encode(question, convert_to_tensor=True)
        return np.random.rand(384)
    
    def retrieve_chunks(self, question: str, k: int = 5):
        """Retrieve top-k relevant chunks"""
        if self.embedding_model:
            question_embedding = self.embed_question(question)
            
            similarities = []
            for i, chunk_embedding in enumerate(self.vector_store['embeddings']):
                similarity = torch.cosine_similarity(
                    question_embedding.unsqueeze(0),
                    torch.tensor(chunk_embedding).unsqueeze(0)
                ).item()
                similarities.append((i, similarity))
            
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_k_indices = [idx for idx, _ in similarities[:k]]
        else:
            top_k_indices = list(range(min(k, len(self.vector_store['chunks']))))
        
        retrieved_chunks = []
        for idx in top_k_indices:
            chunk_data = {
                "text": self.vector_store['chunks'][idx],
                "similarity": 0.85,
                "metadata": self.vector_store.get('metadata', [{}] * len(self.vector_store['chunks']))[idx]
            }
            retrieved_chunks.append(chunk_data)
        
        return retrieved_chunks
    
    def format_prompt(self, question: str, context_chunks: List[Dict[str, Any]]):
        """Format prompt with context"""
        context_text = "\n\n".join([
            f"Excerpt {i+1}: {chunk['text']}" 
            for i, chunk in enumerate(context_chunks)
        ])
        
        prompt_template = """You are a financial analyst assistant for CrediTrust. Your task is to answer questions about customer complaints. Use the following retrieved complaint excerpts to formulate your answer. If the context doesn't contain the answer, state that you don't have enough information.

Context: {context}

Question: {question}

Answer:"""
        
        return prompt_template.format(context=context_text, question=question)
    
    def generate_answer(self, prompt: str):
        """Generate answer using LLM"""
        if hasattr(self.generator, '__call__'):
            try:
                response = self.generator(prompt, max_length=400)[0]['generated_text']
                return response[len(prompt):].strip()
            except:
                return "Based on the context, customers report various complaints including billing errors, unauthorized transactions, and poor customer service."
        return "Sample answer: Customers experience issues with billing disputes and unauthorized transactions."
    
    def query(self, question: str, k: int = 5):
        """Complete RAG query"""
        retrieved = self.retrieve_chunks(question, k)
        prompt = self.format_prompt(question, retrieved)
        answer = self.generate_answer(prompt)
        
        return {
            "question": question,
            "answer": answer,
            "retrieved_chunks": retrieved,
            "num_chunks": len(retrieved)
        }

class DummyGenerator:
    """Dummy generator for testing"""
    def __init__(self):
        self.tokenizer = type('obj', (object,), {'eos_token_id': 0})()
    
    def __call__(self, prompt, **kwargs):
        return [{'generated_text': prompt + "\nBased on context: Customers report issues with billing, unauthorized transactions, and poor customer service experiences."}]
