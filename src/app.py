import os
import gradio as gr
import warnings
from pathlib import Path
import shutil

# Suppress LangChain deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from rag_tool import RAGTool

# Initialize the RAG Tool
rag_tool = RAGTool()

# Function to handle document uploads
def upload_file(file):
    try:
        # Create documents directory if it doesn't exist
        os.makedirs("./documents", exist_ok=True)
        
        # Get the file path and name
        file_path = Path(file.name)
        destination = Path("./documents") / file_path.name
        
        # Copy the file to documents directory
        shutil.copy(file_path, destination)
        
        # Configure RAG tool
        rag_tool.configure(
            documents_path=str(destination),
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            persist_directory="./vector_store"
        )
        
        return f"File uploaded and processed: {file_path.name}"
    except Exception as e:
        return f"Error processing file: {str(e)}"

# Function to query the documents
def query_document(question):
    try:
        if not hasattr(rag_tool, 'vector_store') or rag_tool.vector_store is None:
            return "Please upload a document first."
        
        response = rag_tool(question)
        return response
    except Exception as e:
        return f"Error querying document: {str(e)}"

# Create a simple Gradio interface
with gr.Blocks(title="RAG Tool") as demo:
    gr.Markdown("# Document Question Answering System")
    gr.Markdown("Upload a document (PDF, TXT) and ask questions about it")
    
    with gr.Row():
        with gr.Column():
            file_input = gr.File(label="Upload Document")
            upload_button = gr.Button("Process Document")
            upload_result = gr.Textbox(label="Upload Status")
        
        with gr.Column():
            query_input = gr.Textbox(label="Ask a Question", placeholder="What would you like to know?")
            query_button = gr.Button("Get Answer")
            response_output = gr.Textbox(label="Answer")
    
    # Set up the button click events
    upload_button.click(
        upload_file,
        inputs=file_input,
        outputs=upload_result
    )
    
    query_button.click(
        query_document,
        inputs=query_input,
        outputs=response_output
    )

# Launch the app
if __name__ == "__main__":
    demo.launch()