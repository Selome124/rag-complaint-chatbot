# app_minimal.py - Simple working version
import gradio as gr
import sys
import os

print("Starting CrediTrust Chat Interface...")

# Simple response function
def respond(message, history):
    # Simulate RAG response
    response = f"""Based on complaint database search for "{message}":
    
Common issues reported:
• Unauthorized transactions
• Billing errors
• Poor customer service

**Sample Sources:**
1. Customer reported $2,345 credit card fraud
2. Billing dispute resolved in 45 days
3. Mortgage payment error caused $450 in fees

Resolution typically takes 30-60 days."""
    
    return response

# Create interface
demo = gr.Interface(
    fn=lambda x: respond(x, []),
    inputs=gr.Textbox(label="Ask about customer complaints", lines=2),
    outputs=gr.Textbox(label="AI Response", lines=10),
    title="🏦 CrediTrust Complaint Analyst",
    description="Ask questions about financial complaints",
    examples=[
        ["What are common credit card complaints?"],
        ["How long do billing disputes take?"],
        ["What mortgage issues are reported?"]
    ]
)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
