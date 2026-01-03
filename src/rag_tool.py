import os
from typing import Dict, List, Optional, Union, Any
from smolagents import Tool
from langchain_community.vectorstores import FAISS, Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings, HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
import json

class RAGTool(Tool):
    name = "rag_retriever"
    description = """
    Advanced RAG (Retrieval-Augmented Generation) tool that searches in vector stores based on given prompts.
    This tool allows you to query documents stored in vector databases using semantic similarity.
    It supports various configurations including different embedding models, vector stores, and document types.
    """
    inputs = {
        "query": {
            "type": "string",
            "description": "The search query to retrieve relevant information from the document store",
        },
        "top_k": {
            "type": "integer",
            "description": "Number of most relevant documents to retrieve (default: 3)",
            "nullable": True
        }
    }
    output_type = "string"
    
    def __init__(self):
        """
        Initialize the RAG Tool with default settings.
        All configuration is done via class attributes or through the configure method.
        """
        super().__init__()
        self.documents_path = "./documents"
        self.embedding_model = "BAAI/bge-small-en-v1.5"
        self.vector_store_type = "faiss"
        self.chunk_size = 1000
        self.chunk_overlap = 200
        self.persist_directory = "./vector_store"
        self.device = "cpu"
        
        # Don't automatically create storage initially, wait for explicit setup
        self.vector_store = None
        
    def configure(self, 
                 documents_path: str = "./documents",
                 embedding_model: str = "BAAI/bge-small-en-v1.5",
                 vector_store_type: str = "faiss",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 persist_directory: str = "./vector_store",
                 device: str = "cpu"):
        """
        Configure the RAG Tool with custom parameters.
        
        Args:
            documents_path: Path to documents or folder containing documents
            embedding_model: HuggingFace model ID for embeddings
            vector_store_type: Type of vector store ('faiss' or 'chroma')
            chunk_size: Size of text chunks for splitting documents
            chunk_overlap: Overlap between text chunks
            persist_directory: Directory to persist vector store
            device: Device to run embedding model on ('cpu' or 'cuda')
        """
        self.documents_path = documents_path
        self.embedding_model = embedding_model
        self.vector_store_type = vector_store_type
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.persist_directory = persist_directory
        self.device = device
        
        # Create the vector store if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        self._setup_vector_store()
        
        return self
        
    def _setup_vector_store(self):
        """Set up the vector store with documents if it doesn't exist"""
        # Always try to create directories if they don't exist
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Check if documents path exists
        if not os.path.exists(self.documents_path):
            print(f"Warning: Documents path {self.documents_path} does not exist.")
            return
        
        # Force creation of vector store from documents
        documents = self._load_documents()
        if not documents:
            print("No documents loaded. Vector store not created.")
            return
        
        # Create the vector store
        self._create_vector_store(documents)
    
    def _get_embeddings(self):
        """Get embedding model based on configuration"""
        try:
            if "bge" in self.embedding_model.lower():
                encode_kwargs = {"normalize_embeddings": True}
                return HuggingFaceBgeEmbeddings(
                    model_name=self.embedding_model,
                    encode_kwargs=encode_kwargs,
                    model_kwargs={"device": self.device}
                )
            else:
                return HuggingFaceEmbeddings(
                    model_name=self.embedding_model,
                    model_kwargs={"device": self.device}
                )
        except Exception as e:
            print(f"Error loading embedding model: {e}")
            # Fallback to a reliable model
            print("Falling back to sentence-transformers/all-MiniLM-L6-v2")
            return HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": self.device}
            )
            
    def _load_documents(self):
        """Load documents from the documents path"""
        documents = []
        
        # Check if documents_path is a file or directory
        if os.path.isfile(self.documents_path):
            # Load single file
            if self.documents_path.lower().endswith('.pdf'):
                try:
                    loader = PyPDFLoader(self.documents_path)
                    documents = loader.load()
                except Exception as e:
                    print(f"Error loading PDF: {e}")
                    # Fallback to using PdfReader
                    try:
                        text = self._extract_text_from_pdf(self.documents_path)
                        splitter = CharacterTextSplitter(
                            separator="\n", 
                            chunk_size=self.chunk_size, 
                            chunk_overlap=self.chunk_overlap
                        )
                        documents = splitter.create_documents([text])
                    except Exception as e2:
                        print(f"Error with fallback PDF extraction: {e2}")
            elif self.documents_path.lower().endswith(('.txt', '.md', '.html')):
                loader = TextLoader(self.documents_path)
                documents = loader.load()
        elif os.path.isdir(self.documents_path):
            # Load all supported files in directory
            try:
                loader = DirectoryLoader(
                    self.documents_path,
                    glob="**/*.*",
                    loader_cls=TextLoader,
                    loader_kwargs={"autodetect_encoding": True}
                )
                documents = loader.load()
            except Exception as e:
                print(f"Error loading directory: {e}")
        
        # Split documents into chunks if they exist
        if documents:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            return splitter.split_documents(documents)
        return []
    
    def _extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF using PyPDF2"""
        text = ""
        pdf_reader = PdfReader(pdf_path)
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
            
    def _create_vector_store(self, documents):
        """Create a new vector store from documents"""
        embeddings = self._get_embeddings()
        
        if self.vector_store_type.lower() == "faiss":
            vector_store = FAISS.from_documents(documents, embeddings)
            vector_store.save_local(self.persist_directory)
            print(f"Created FAISS vector store at {self.persist_directory}")
        else:  # Default to Chroma
            vector_store = Chroma.from_documents(
                documents, 
                embeddings,
                persist_directory=self.persist_directory
            )
            vector_store.persist()
            print(f"Created Chroma vector store at {self.persist_directory}")
        
        self.vector_store = vector_store
    
    def _load_vector_store(self):
        """Load an existing vector store"""
        embeddings = self._get_embeddings()
        
        try:
            if self.vector_store_type.lower() == "faiss":
                self.vector_store = FAISS.load_local(self.persist_directory, embeddings)
                print(f"Loaded FAISS vector store from {self.persist_directory}")
            else:  # Default to Chroma
                self.vector_store = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=embeddings
                )
                print(f"Loaded Chroma vector store from {self.persist_directory}")
        except Exception as e:
            print(f"Error loading vector store: {e}")
            print("Creating a new vector store...")
            documents = self._load_documents()
            if documents:
                self._create_vector_store(documents)
            else:
                print("No documents available. Cannot create vector store.")
                self.vector_store = None

    def forward(self, query: str, top_k: int = None) -> str:
        """
        Retrieve relevant documents based on the query.
        
        Args:
            query: The search query
            top_k: Number of results to return (default: 3)
            
        Returns:
            String with formatted search results
        """
        # Set default value if None
        if top_k is None:
            top_k = 3
        if not hasattr(self, 'vector_store') or self.vector_store is None:
            return "Vector store is not initialized. Please check your configuration."
        
        try:
            # Perform similarity search
            results = self.vector_store.similarity_search(query, k=top_k)
            
            # Format results
            formatted_results = []
            for i, doc in enumerate(results):
                content = doc.page_content
                metadata = doc.metadata
                
                # Format metadata nicely
                meta_str = ""
                if metadata:
                    meta_str = "\nSource: "
                    if "source" in metadata:
                        meta_str += metadata["source"]
                    if "page" in metadata:
                        meta_str += f", Page: {metadata['page']}"
                
                formatted_results.append(f"Document {i+1}:\n{content}{meta_str}\n")
            
            if formatted_results:
                return "Retrieved relevant information:\n\n" + "\n".join(formatted_results)
            else:
                return "No relevant information found for the query."
        except Exception as e:
            return f"Error retrieving information: {str(e)}"

# Example usage:
# rag_tool = RAGTool(
#     documents_path="./my_docs",
#     embedding_model="sentence-transformers/all-MiniLM-L6-v2",
#     vector_store_type="faiss",
#     chunk_size=1000,
#     chunk_overlap=200
# )