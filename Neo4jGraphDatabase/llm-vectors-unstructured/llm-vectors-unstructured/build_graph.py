import os
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings 
from textblob import TextBlob
from neo4j import GraphDatabase

COURSES_PATH = r"C:\Users\Raghu\Downloads\private KEY\ETLpipeline\llm-vectors-unstructured\llm-vectors-unstructured\data\asciidoc\courses"

huggingface_emb = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

loader = DirectoryLoader(COURSES_PATH, glob="**/lesson.adoc", loader_cls=TextLoader)
docs = loader.load()

text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=1500,
    chunk_overlap=200,
)

chunks = text_splitter.split_documents(docs)

def get_embedding(huggingface_emb, text):
    embedding = huggingface_emb.embed_query(text)  
    return embedding

def get_course_data(huggingface_emb, chunk):
    data = {}
    path = chunk.metadata['source'].split(os.path.sep)
    data['course'] = path[-6]
    data['module'] = path[-4]
    data['lesson'] = path[-2]
    data['url'] = f"https://graphacademy.neo4j.com/courses/{data['course']}/{data['module']}/{data['lesson']}"
    data['text'] = chunk.page_content
    data['embedding'] = get_embedding(huggingface_emb, data['text'])
    data['topics'] = TextBlob(data['text']).noun_phrases
    return data

def create_chunk(tx, data):
    tx.run("""
        MERGE (c:Course {name: $course})
        MERGE (c)-[:HAS_MODULE]->(m:Module{name: $module})
        MERGE (m)-[:HAS_LESSON]->(l:Lesson{name: $lesson, url: $url})
        MERGE (l)-[:CONTAINS]->(p:Paragraph{text: $text})
        WITH p
        CALL db.create.setNodeVectorProperty(p, "embedding", $embedding)
           
        FOREACH (topic in $topics |
            MERGE (t:Topic {name: topic})
            MERGE (p)-[:MENTIONS]->(t)
        )
        """, 
        data
    )

# Connect to Neo4j
driver = GraphDatabase.driver(
    os.getenv('NEO4J_URI'),
    auth=(
        os.getenv('NEO4J_USERNAME'),
        os.getenv('NEO4J_PASSWORD')
    )
)
driver.verify_connectivity()

# Process each chunk
for chunk in chunks:
    with driver.session(database="neo4j") as session:
        session.execute_write(
            create_chunk,
            get_course_data(huggingface_emb, chunk)
        )

driver.close()
