import gradio as gr
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from test_odata_servicelayer import perform_odata_query


def query_odata(question: str) -> tuple[str, str, str]:
    """Generate OData query from natural language question."""
    if not question.strip():
        return "", "", "Please enter a question."
    
    try:
        result = perform_odata_query(question)
        # Parse JSON response
        parsed = json.loads(result)
        
        entity = parsed.get("entity", "")
        query_parts = parsed.get("query", {})
        full_url = parsed.get("full_url", "")
        
        # Format query parts for display
        query_display = json.dumps(query_parts, indent=2)
        
        return entity, query_display, full_url
    except json.JSONDecodeError:
        # If response isn't valid JSON, return raw result
        return "", "", result
    except Exception as e:
        return "", "", f"Error: {str(e)}"


def run_app():
    with gr.Blocks(title="SAP OData Query Generator") as interface:
        gr.Markdown("# 🔍 SAP Service Layer OData Query Generator")
        gr.Markdown("Enter a natural language question to generate an OData query for SAP Business One Service Layer.")
        
        with gr.Row():
            question_input = gr.Textbox(
                label="Your Question",
                placeholder="e.g., Find Business Partner with name 'John Doe'",
                lines=2,
                scale=4
            )
            query_btn = gr.Button("Generate Query", variant="primary", scale=1)
        
        with gr.Row():
            with gr.Column():
                entity_output = gr.Textbox(label="Entity", lines=1)
            with gr.Column():
                url_output = gr.Textbox(label="Full URL", lines=1)
        
        query_output = gr.Code(
            label="Query Parameters",
            language="json",
            lines=10
        )
        
        gr.Markdown("### Example Questions")
        gr.Examples(
            examples=[
                ["Find Business Partner with name 'John Doe'"],
                ["Get all open sales orders from customer C20000"],
                ["List top 10 items with price greater than 100"],
                ["Find all invoices from July 2025"],
            ],
            inputs=question_input
        )
        
        query_btn.click(
            query_odata,
            inputs=question_input,
            outputs=[entity_output, query_output, url_output]
        )
        
        question_input.submit(
            query_odata,
            inputs=question_input,
            outputs=[entity_output, query_output, url_output]
        )
    
    interface.launch(server_name="0.0.0.0", server_port=7861)


if __name__ == "__main__":
    run_app()
