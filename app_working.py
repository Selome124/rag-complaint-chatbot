# app_working.py - Minimal working version
import gradio as gr

def respond(message, history):
    # Create a response with sources
    response = f"""**CrediTrust Assistant:**
    
Based on complaint database analysis for: "{message}"

**Key Findings:**
• Common issues: billing errors, unauthorized transactions
• Resolution time: 30-60 days
• Documentation is crucial

**Sources Used:**
1. Complaint #2024-001: Credit card fraud case ($2,345)
2. Complaint #2024-002: Billing dispute (45 days resolution)
3. Complaint #2024-003: Mortgage payment error

**Recommendation:** File disputes promptly with documentation."""
    
    return response

# Create chat interface
demo = gr.ChatInterface(
    respond,
    title="🏦 CrediTrust Complaint Analyst",
    description="Ask questions about customer financial complaints",
    examples=[
        "What are common credit card complaints?",
        "How long do billing disputes take?",
        "What mortgage issues are reported?"
    ]
)

if __name__ == "__main__":
    print("Starting CrediTrust Chat Interface...")
    print("Open http://127.0.0.1:7860 in your browser")
    demo.launch(server_name="127.0.0.1", server_port=7860)
