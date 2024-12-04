import os
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain.embeddings import HuggingFaceEmbeddings  # Import HuggingFaceEmbeddings

COURSES_PATH = r"C:\Users\Raghu\Downloads\private KEY\ETLpipeline\llm-vectors-unstructured\llm-vectors-unstructured\data\asciidoc\courses"

# Load lesson documents
loader = DirectoryLoader(COURSES_PATH, glob='**/*lesson.adoc', loader_cls=TextLoader)
docs = loader.load()

# Create a text splitter
text_splitter = CharacterTextSplitter(separator="\n\n", chunk_size=1500, chunk_overlap=200)

# Split documents into chunks
chunks = text_splitter.split_documents(docs)

# Initialize Hugging Face embeddings
huggingface_emb = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # Choose a suitable model from Hugging Face

# Embed the documents
embedded_chunks = huggingface_emb.embed_documents([chunk.page_content for chunk in chunks])

# Create a Neo4j vector store using Hugging Face embeddings
neo4j_db = Neo4jVector.from_documents(
    chunks,
    huggingface_emb,  # Use the Hugging Face embeddings instance
    url=os.getenv('NEO4J_URI'),
    username=os.getenv('NEO4J_USERNAME'),
    password=os.getenv('NEO4J_PASSWORD'),
    database="neo4j",  
    index_name="chunkVector",
    node_label="Chunk", 
    text_node_property="text", 
    embedding_node_property="embedding"
)

# Example of embedding a query
query_embedding = huggingface_emb.embed_query("What is the second letter of the Greek alphabet")
