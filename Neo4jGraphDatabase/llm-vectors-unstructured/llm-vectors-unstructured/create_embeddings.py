import os
from dotenv import load_dotenv
load_dotenv()

from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.graphs import Neo4jGraph
# Initialize Hugging Face embeddings
huggingface_emb = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # You can choose a different model as needed

# Create embeddings for your input text
input_text = "I am doing totally great"
embedding = huggingface_emb.embed_query(input_text)  # Use embed_query for single input

print(len(embedding))

graph = Neo4jGraph(
    url=os.getenv('NEO4J_URI'),
    username=os.getenv('NEO4J_USERNAME'),
    password=os.getenv('NEO4J_PASSWORD')
)

result = graph.query("""
CALL db.index.vector.queryNodes('chunkVector', 6, $embedding)
YIELD node, score
RETURN node.text, score
""", {"embedding": embedding})

for row in result:
    print(row['node.text'], row['score'])