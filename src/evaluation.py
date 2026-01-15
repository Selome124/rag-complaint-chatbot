# src/evaluation.py
def evaluate_rag_pipeline(rag_pipeline, questions, k=5):
    """
    Evaluate the RAG pipeline on a set of questions
    
    Args:
        rag_pipeline: Initialized RAGPipeline
        questions: List of questions to evaluate
        k: Number of chunks to retrieve
    
    Returns:
        List of evaluation results
    """
    evaluation_results = []
    
    print(f"Evaluating {len(questions)} questions...")
    print("-" * 80)
    
    for i, question in enumerate(questions, 1):
        print(f"Question {i}/{len(questions)}: {question[:50]}...")
        
        # Get RAG response
        result = rag_pipeline.query(question, k)
        
        # Format evaluation result
        eval_result = {
            'question': question,
            'generated_answer': result['answer'],
            'retrieved_sources': [
                {
                    'excerpt': chunk['text'][:100] + "..." if len(chunk['text']) > 100 else chunk['text'],
                    'similarity': round(chunk['similarity'], 3)
                }
                for chunk in result['retrieved_chunks'][:2]  # Top 2 sources
            ],
            'quality_score': None,  # To be manually scored
            'analysis': None  # To be manually analyzed
        }
        
        evaluation_results.append(eval_result)
    
    return evaluation_results

def display_evaluation_table(evaluation_results):
    """
    Display evaluation results in markdown table format
    """
    print("\n" + "="*100)
    print("EVALUATION RESULTS")
    print("="*100)
    
    for i, result in enumerate(evaluation_results, 1):
        print(f"\n### Question {i}: {result['question']}")
        print(f"**Generated Answer:** {result['generated_answer'][:200]}...")
        print(f"\n**Retrieved Sources (Top 2):**")
        for j, source in enumerate(result['retrieved_sources'], 1):
            print(f"  {j}. (Similarity: {source['similarity']}) {source['excerpt']}")
        print(f"\n**Quality Score:** {result['quality_score'] if result['quality_score'] else 'Not scored'}")
        print(f"**Analysis:** {result['analysis'] if result['analysis'] else 'No analysis yet'}")
        print("-" * 80)

# Sample evaluation questions
SAMPLE_QUESTIONS = [
    "What are the most common types of credit card complaints?",
    "How long does it typically take to resolve billing disputes?",
    "What should customers do if they find unauthorized transactions?",
    "What are the main issues with mortgage servicing?",
    "How are customers affected by credit reporting errors?",
    "What customer service issues are frequently reported?",
    "What problems occur with debt collection practices?",
    "What are the common issues with student loan servicing?",
    "How do customers report problems with payment processing?",
    "What are the typical resolution outcomes for fraud complaints?"
]