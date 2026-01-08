

import psycopg2
from psycopg2.extras import execute_values
from sentence_transformers import SentenceTransformer
from groq import Groq
import json
from datetime import datetime

# =============================================================================
# CONFIG
# =============================================================================
DB_CONFIG = {
    "host": "localhost",
    "database": "postgres",
    "user": "postgres",
    "password": "postgres"
}
GROQ_API_KEY = ""  # <-- Replace this!
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # 384 dimensions
LLM_MODEL = "llama-3.1-8b-instant"
VECTOR_SIZE = 384

# =============================================================================
# WHAT IS EPISODIC MEMORY?
# =============================================================================


# =============================================================================
# SETUP: Create tables with pgvector
# =============================================================================

def setup_db():
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cur.execute("DROP TABLE IF EXISTS memories;")
    cur.execute(f"""
        CREATE TABLE memories (
            id SERIAL PRIMARY KEY,
            user_id VARCHAR(100),
            content TEXT,
            embedding vector({VECTOR_SIZE})
        );
        CREATE INDEX ON memories USING hnsw (embedding vector_cosine_ops) WITH (m=16, ef_construction=200);
    """)
    conn.commit()
    conn.close()
    print("✅ Database ready with HNSW index")


# =============================================================================
# SINGLETON: Load models once
# =============================================================================

class Clients:
    # Class variables (shared by all) - start as None
    encoder = None  # Will hold SentenceTransformer model
    llm = None      # Will hold Groq client

    @classmethod
    def init(cls):
        """Load models once. Called automatically if needed."""

        # Load embedding model (only if not already loaded)
        if cls.encoder is None:
            print("🔧 Loading embedding model...")
            cls.encoder = SentenceTransformer("all-MiniLM-L6-v2")

        # Connect to Groq LLM (only if not already connected)
        if cls.llm is None:
            print("🔧 Connecting to Groq...")
            cls.llm = Groq(api_key=GROQ_API_KEY)

    @classmethod
    def embed(cls, text):
        """Convert text to vector (384 numbers)."""

        # # Auto-init if not initialized
        # if cls.encoder is None:
        #     cls.init()

        # Encode text to vector, convert to Python list
        vector = cls.encoder.encode(str(text))
        return vector.tolist()

    @classmethod
    def ask(cls, prompt):
        """Send prompt to LLM, get response."""

        # # Auto-init if not initialized
        # if cls.llm is None:
        #     cls.init()

        response = cls.llm.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000
        )


        return response.choices[0].message.content


# =============================================================================
# EPISODIC MEMORY EXTRACTION
# =============================================================================

EPISODIC_EXTRACTION_PROMPT = """You are a memory extraction system. Extract specific, factual episodic memories about the user from this conversation.

EPISODIC MEMORIES are:
- Specific facts about the user (name, job, preferences)
- Events or experiences they mention
- Relationships (pets, family, friends)
- Habits or routines
- Skills or expertise

Rules:
1. Write each memory as a standalone fact (should make sense without context)
2. Write from third person perspective ("User prefers...", "User has...")
3. Only extract FACTS, not opinions or general statements
4. Be specific - include names, numbers, dates when mentioned
5. Return as JSON array of strings

CONVERSATION:
{conversation}

Return ONLY a JSON array of extracted memories, like:
["User's name is Alice", "User works as a software engineer", "User has a dog named Max"]

If no episodic memories can be extracted, return: []
"""

def extract_memories(conversation):
    """Use LLM to extract facts from conversation."""
    prompt = EPISODIC_EXTRACTION_PROMPT.format(conversation=conversation)
    # response = Clients.ask(f'Extract user facts from this conversation as JSON array: {conversation}')
    response = Clients.ask(prompt)
    

    try:
        result = json.loads(response[response.find("["):response.rfind("]")+1])
        
        return [str(m) for m in result] if isinstance(result, list) else []
    except Exception as e:
        print(f"  Parse error: {e}")
        return []

def to_pgvector(text):
    """
    Convert text to pgvector format.

    pgvector needs vectors as strings like: '[0.1, 0.2, 0.3]'
    Then we cast it with ::vector in SQL.

    Steps:
    1. text → embedding (list of 384 floats)
    2. embedding → string format "[0.1, 0.2, ...]"
    """

    # Step 1: Convert text to embedding (384 numbers)
    # Example: "User is Alice" → [0.12, -0.34, 0.56, ...]
    embedding = Clients.embed(text)

    # Step 2: Convert each number to string
    # Example: [0.12, -0.34] → ["0.12", "-0.34"]
    string_numbers = [str(num) for num in embedding]

    # Step 3: Join with commas
    # Example: ["0.12", "-0.34"] → "0.12,-0.34"
    joined = ",".join(string_numbers)

    # Step 4: Wrap in brackets
    # Example: "0.12,-0.34" → "[0.12,-0.34]"
    pgvector_format = "[" + joined + "]"

    return pgvector_format

def store_memory(user_id, content):
    """Store memory with its vector embedding."""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("INSERT INTO memories (user_id, content, embedding) VALUES (%s, %s, %s::vector)",
                (user_id, content, to_pgvector(content)))
    conn.commit()
    conn.close()

def search_memories(user_id, query, limit=5):
    """
    Search similar memories using HNSW vector search.
    """

    # Convert query text to vector
    query_vec = to_pgvector(query)

    # Connect to database
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    # Search for similar vectors
    # <=> is cosine distance (0 = identical, 2 = opposite)
    # 1 - distance = similarity score (1 = identical, 0 = different)
    cur.execute("""
        SELECT content, 1 - (embedding <=> %s::vector) as similarity
        FROM memories
        WHERE user_id = %s
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """, (query_vec, user_id, query_vec, limit))

    # Fetch and format results
    rows = cur.fetchall()
    conn.close()

    results = []
    for row in rows:
        results.append({
            "content": row[0],
            "score": row[1]
        })

    return results

def chat(user_id, message):
    """Chat using retrieved memories for context."""
    memories = search_memories(user_id, message, limit=3)
    context = "\n".join([f"- {m['content']}" for m in memories]) or "No memories yet"
    return Clients.ask(f"User memories:\n{context}\n\nUser: {message}\n\nRespond personally:")


# =============================================================================
# DEMO
# =============================================================================
def run_demo():
    print(" EPISODIC MEMORY DEMO\n")

    # Setup
    setup_db()
    Clients.init()
    user_id = "alice"

    # 1. Process conversation → Extract & store memories
    print("\n📝 EXTRACTING MEMORIES FROM CONVERSATION:")
    conversation = """User: Hi! I'm Alice, a software engineer. I have a dog named Max.
Assistant: Nice! What do you work on?
User: Backend systems in Python. I drink lots of coffee - about 4 espressos daily!"""

    memories = extract_memories(conversation)
    for m in memories:
        store_memory(user_id, m)
        print(f"  {m}")

    # 2. Search memories
    print(" SEARCHING MEMORIES:")
    for query in ["What's her job?", "Does she have pets?", "What does she drink?"]:
        results = search_memories(user_id, query, limit=2)
        print(f"  Q: {query}")
        for r in results:
            print(f"     [{r['score']:.2f}] {r['content']}")

    # 3. Chat with memory
    print("CHAT WITH MEMORY:")
    response = chat(user_id, "What do you know about me?")
    print(f"  🤖 {response}")

    print("DEMO COMPLETE!")

if __name__ == "__main__":
    run_demo()
