# Advanced RAG Tool for smolagents

This repository contains an improved Retrieval-Augmented Generation (RAG) tool built for the `smolagents` library from Hugging Face. This tool allows you to:

- Create vector stores from various document types (PDF, TXT, HTML, etc.)
- Choose different embedding models for better semantic understanding
- Configure chunk sizes and overlaps for optimal text splitting
- Select between different vector stores (FAISS or Chroma)
- Share your tool on the Hugging Face Hub

## Installation

```bash
pip install smolagents langchain-community langchain-text-splitters faiss-cpu chromadb sentence-transformers pypdf2 gradio
```

## Basic Usage

```python
from rag_tool import RAGTool

# Initialize the RAG tool
rag_tool = RAGTool()

# Configure with custom settings
rag_tool.configure(
    documents_path="./my_document.pdf",  
    embedding_model="BAAI/bge-small-en-v1.5",
    vector_store_type="faiss",
    chunk_size=1000,
    chunk_overlap=200,
    persist_directory="./vector_store",
    device="cpu"  # Use "cuda" for GPU acceleration
)

# Query the documents
result = rag_tool("What is attention in transformer architecture?", top_k=3)
print(result)
```

## Using with an Agent

```python
import warnings
# Suppress LangChain deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from smolagents import CodeAgent, InferenceClientModel
from rag_tool import RAGTool

# Initialize and configure the RAG tool
rag_tool = RAGTool()
rag_tool.configure(documents_path="./my_document.pdf")

# Create an agent model
model = InferenceClientModel(
    model_id="mistralai/Mistral-7B-Instruct-v0.2",
    token="your_huggingface_token"
)

# Create the agent with our RAG tool
agent = CodeAgent(tools=[rag_tool], model=model, add_base_tools=True)

# Run the agent
result = agent.run("Explain the key components of the transformer architecture")
print(result)
```

## Gradio Interface

For an interactive experience, run the Gradio app:

```bash
python gradio_app.py
```

This provides a web interface where you can:
- Upload documents
- Configure embedding models and chunk settings
- Query your documents with semantic search

## Customization Options

### Embedding Models

You can choose from various embedding models:
- `sentence-transformers/all-MiniLM-L6-v2` (fast, smaller model)
- `BAAI/bge-small-en-v1.5` (good balance of performance and speed)
- `BAAI/bge-base-en-v1.5` (better performance, slower)
- `thenlper/gte-small` (good for general text embeddings)
- `thenlper/gte-base` (larger GTE model)

### Vector Store Types

- `faiss`: Fast, in-memory vector database (better for smaller collections)
- `chroma`: Persistent vector database with metadata filtering capabilities

### Document Types

The tool supports multiple document types:
- PDF documents
- Text files (.txt)
- Markdown files (.md)
- HTML files (.html)
- Entire directories of mixed document types

## Sharing Your Tool

You can share your tool on the Hugging Face Hub:

```python
rag_tool.push_to_hub("your-username/rag-retrieval-tool", token="your_huggingface_token")
```

## Limitations

- The tool currently doesn't support image content from PDFs
- Very large documents may require additional memory
- Some embedding models may be slow on CPU-only environments

## Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.

## License

MIT