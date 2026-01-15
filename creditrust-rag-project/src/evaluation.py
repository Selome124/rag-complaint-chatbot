# src/evaluation.py
def evaluate_rag_pipeline(rag_pipeline, questions, k=5):
    """Evaluate RAG pipeline on questions"""
    print(f"\nEvaluating {len(questions)} questions...")
    print("=" * 60)
    
    results = []
    for i, question in enumerate(questions, 1):
        print(f"\nQ{i}: {question}")
        
        result = rag_pipeline.query(question, k)
        
        eval_result = {
            "question": question,
            "generated_answer": result["answer"],
            "retrieved_sources": [
                {
                    "text": chunk["text"][:80] + "..." if len(chunk["text"]) > 80 else chunk["text"],
                    "similarity": round(chunk.get("similarity", 0.8), 3)
                }
                for chunk in result["retrieved_chunks"][:2]
            ],
            "quality_score": None,
            "comments": None
        }
        results.append(eval_result)
    
    return results

def display_results(results):
    """Display evaluation results"""
    print("\n" + "="*80)
    print("EVALUATION TABLE")
    print("="*80)
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Question: {result['question']}")
        print(f"   Answer: {result['generated_answer'][:150]}...")
        print(f"   Sources:")
        for j, source in enumerate(result["retrieved_sources"], 1):
            print(f"     {j}. (sim: {source['similarity']}) {source['text']}")
        print(f"   Score: {result['quality_score'] or 'Not scored'}")
        print(f"   Comments: {result['comments'] or 'None'}")
        print("-" * 80)

# Sample questions for evaluation
SAMPLE_QUESTIONS = [
    "What are common credit card complaints?",
    "How long to resolve billing disputes?",
    "What to do for unauthorized transactions?",
    "What are mortgage servicing issues?",
    "How do credit errors affect customers?",
    "What customer service issues are reported?",
    "What problems occur with debt collection?",
    "What are student loan servicing issues?",
    "How to report payment processing problems?",
    "What are outcomes for fraud complaints?"
]
