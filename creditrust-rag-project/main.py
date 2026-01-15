# main.py - Main execution script
import sys
import os

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

print("=" * 80)
print("CREDITRUST RAG PIPELINE EVALUATION SYSTEM")
print("=" * 80)

try:
    from rag_pipeline import RAGPipeline
    from evaluation import evaluate_rag_pipeline, display_results, SAMPLE_QUESTIONS
    print("✓ Modules imported successfully")
except ImportError as e:
    print(f"✗ Error importing modules: {e}")
    print("\nPlease make sure you have:")
    print("1. Created src/rag_pipeline.py and src/evaluation.py")
    print("2. Installed dependencies: pip install -r requirements.txt")
    sys.exit(1)

def main():
    """Main function to run the RAG evaluation"""
    
    # Update this path to where your vector_store.pkl is located
    VECTOR_STORE_PATH = "data/vector_store.pkl"
    
    # Check if vector store exists
    if not os.path.exists(VECTOR_STORE_PATH):
        print(f"\n⚠️  Warning: Vector store not found at {VECTOR_STORE_PATH}")
        print("Please place your vector_store.pkl file in the 'data/' folder")
        
        # Create a dummy vector store for testing
        create_test_vector_store()
        print("Created test vector store for demonstration")
    
    try:
        # Initialize RAG pipeline
        print("\nInitializing RAG Pipeline...")
        rag = RAGPipeline(vector_store_path=VECTOR_STORE_PATH)
        
        # Run evaluation
        print("\nRunning evaluation on sample questions...")
        results = evaluate_rag_pipeline(rag, SAMPLE_QUESTIONS, k=3)
        
        # Display results
        display_results(results)
        
        # Save results to file
        save_results_to_file(results)
        
        print("\n" + "=" * 80)
        print("EVALUATION COMPLETE!")
        print("=" * 80)
        print("\nNext steps:")
        print("1. Review the generated answers above")
        print("2. Manually assign quality scores (1-5 scale)")
        print("3. Replace test vector store with your actual vector_store.pkl")
        print("4. Update the LLM model in rag_pipeline.py for better results")
        
    except Exception as e:
        print(f"\n✗ Error running evaluation: {e}")
        import traceback
        traceback.print_exc()

def create_test_vector_store():
    """Create a test vector store for demonstration"""
    import pickle
    import numpy as np
    
    # Sample complaint data for testing
    test_chunks = [
        "Customer reported unauthorized charges on credit card amounting to $2,345. Bank initially refused to investigate.",
        "Billing dispute unresolved for 45 days despite providing all requested documentation.",
        "Credit report incorrectly showed late payments lowering score by 85 points.",
        "Mortgage payment incorrectly applied causing false late fees for three billing cycles.",
        "Customer spent 7 months disputing erroneous collections account on credit report.",
        "Unauthorized transaction reported within 24 hours but investigation took 30 days.",
        "Bank charged overdraft fees despite sufficient funds in the account.",
        "Loan modification request delayed by 6 months due to contradictory information.",
        "Student loan servicer incorrectly calculated interest for 2 years.",
        "Debt collector contacted customer 10 times per day violating regulations."
    ]
    
    # Create dummy embeddings (384 dimensions like MiniLM-L6-v2)
    test_embeddings = [np.random.rand(384).tolist() for _ in range(len(test_chunks))]
    
    # Create vector store dictionary
    vector_store = {
        'chunks': test_chunks,
        'embeddings': test_embeddings,
        'metadata': [{'source': f'test_{i}'} for i in range(len(test_chunks))]
    }
    
    # Save to file
    with open("data/vector_store.pkl", 'wb') as f:
        pickle.dump(vector_store, f)
    
    print(f"Created test vector store with {len(test_chunks)} chunks")

def save_results_to_file(results):
    """Save evaluation results to a text file"""
    with open("evaluation_results.txt", "w", encoding="utf-8") as f:
        f.write("RAG Pipeline Evaluation Results\n")
        f.write("=" * 60 + "\n\n")
        
        for i, result in enumerate(results, 1):
            f.write(f"Question {i}: {result['question']}\n")
            f.write(f"Answer: {result['generated_answer']}\n\n")
            
            f.write("Retrieved Sources:\n")
            for j, source in enumerate(result['retrieved_sources'], 1):
                f.write(f"  Source {j} (similarity: {source['similarity']}):\n")
                f.write(f"    {source['text']}\n\n")
            
            f.write("-" * 60 + "\n\n")
    
    print(f"\n✓ Results saved to evaluation_results.txt")

if __name__ == "__main__":
    main()