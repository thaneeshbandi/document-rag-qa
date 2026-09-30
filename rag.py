"""
The RAG pipeline.

Indexing (when documents are uploaded):
    document text -> chunks -> embedding vectors -> FAISS index (saved to disk)

Answering (when a question is asked):
    question -> embedding vector -> FAISS finds top-k similar chunks
             -> chunks become the context -> Groq LLM writes the answer
"""

import os
import pickle

import faiss
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

from loaders import is_supported_file, load_document

load_dotenv()  # reads GROQ_API_KEY from the .env file

# ---------------- Settings ----------------
DATA_DIR = "data"  # uploaded documents are saved here
FAISS_DIR = "faiss_store"  # the FAISS index and metadata are saved here
INDEX_PATH = os.path.join(FAISS_DIR, "faiss.index")
METADATA_PATH = os.path.join(FAISS_DIR, "metadata.pkl")

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "openai/gpt-oss-20b"

CHUNK_SIZE = 1000  # characters per chunk
CHUNK_OVERLAP = 200  # characters shared between neighbouring chunks
TOP_K = 3  # number of chunks sent to the LLM
MIN_SIMILARITY = 0.2  # chunks less similar than this are treated as not relevant

NOT_FOUND_MESSAGE = "I couldn't find this information in the uploaded documents."

SYSTEM_PROMPT = f"""You are a document question-answering assistant.
Answer the question using only the provided context.
Do not use outside knowledge and do not make up information.
If the answer is not present in the context, reply exactly: "{NOT_FOUND_MESSAGE}\""""

# ---------------- State ----------------
# The embedding model is loaded once at startup and reused for every request.
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# The FAISS index and its metadata are kept in memory while the app runs.
# metadata[i] holds the text and source file of the vector at position i in the index.
index = None
metadata = []


# ---------------- Indexing ----------------
def split_into_chunks(text):
    """Split text into overlapping fixed-size chunks of characters."""
    chunks = []
    step = CHUNK_SIZE - CHUNK_OVERLAP  # each chunk starts 800 characters after the previous one
    for start in range(0, len(text), step):
        chunk = text[start : start + CHUNK_SIZE].strip()
        if chunk:
            chunks.append(chunk)
        if start + CHUNK_SIZE >= len(text):  # this chunk reached the end of the text
            break
    return chunks


def embed_texts(texts):
    """Convert a list of texts into embedding vectors (one 384-number vector per text)."""
    # normalize_embeddings=True gives every vector a length of 1,
    # so the inner product FAISS computes is the cosine similarity.
    vectors = embedding_model.encode(texts, normalize_embeddings=True)
    return vectors.astype("float32")  # FAISS expects float32


def rebuild_index():
    """Read every document in data/, embed all chunks, and save a fresh FAISS index."""
    global index, metadata

    all_chunks = []
    for filename in sorted(os.listdir(DATA_DIR)):
        if not is_supported_file(filename):
            continue  # skip files like .DS_Store
        document = load_document(os.path.join(DATA_DIR, filename))
        for chunk_text in split_into_chunks(document["text"]):
            all_chunks.append({"text": chunk_text, "source": document["source"]})

    if not all_chunks:
        return 0

    vectors = embed_texts([chunk["text"] for chunk in all_chunks])

    # IndexFlatIP = exact search using inner product (cosine similarity for normalized vectors)
    new_index = faiss.IndexFlatIP(vectors.shape[1])
    new_index.add(vectors)

    os.makedirs(FAISS_DIR, exist_ok=True)
    faiss.write_index(new_index, INDEX_PATH)
    with open(METADATA_PATH, "wb") as file:
        pickle.dump(all_chunks, file)

    index = new_index
    metadata = all_chunks
    return len(all_chunks)


def load_index():
    """Load the saved FAISS index from disk (called once when the app starts)."""
    global index, metadata
    if os.path.exists(INDEX_PATH) and os.path.exists(METADATA_PATH):
        index = faiss.read_index(INDEX_PATH)
        with open(METADATA_PATH, "rb") as file:
            metadata = pickle.load(file)
        print(f"Loaded FAISS index with {index.ntotal} chunks.")
    else:
        print("No FAISS index found yet. Upload a document to create one.")


# ---------------- Answering ----------------
def retrieve_chunks(question, top_k=TOP_K):
    """Return the top_k chunks most similar to the question."""
    question_vector = embed_texts([question])
    scores, positions = index.search(question_vector, top_k)

    results = []
    for score, position in zip(scores[0], positions[0]):
        if position == -1:  # FAISS returns -1 when the index has fewer than top_k chunks
            continue
        chunk = metadata[position]
        results.append({"text": chunk["text"], "source": chunk["source"], "score": round(float(score), 3)})
    return results


def generate_answer(question, chunks):
    """Send the retrieved chunks and the question to the Groq LLM."""
    context = "\n\n".join(f"[Source: {chunk['source']}]\n{chunk['text']}" for chunk in chunks)
    user_prompt = f"Context:\n{context}\n\nQuestion:\n{question}"

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = client.chat.completions.create(
        model=LLM_MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,  # 0 = most predictable answers, no creative guessing
    )
    return response.choices[0].message.content.strip()


def answer_question(question):
    """The full RAG flow: retrieve relevant chunks, then generate a grounded answer."""
    chunks = retrieve_chunks(question)
    relevant_chunks = [chunk for chunk in chunks if chunk["score"] >= MIN_SIMILARITY]

    # If nothing in the documents is similar enough, don't ask the LLM at all.
    if not relevant_chunks:
        return {"answer": NOT_FOUND_MESSAGE, "sources": [], "chunks": chunks}

    answer = generate_answer(question, relevant_chunks)

    # The chunks were similar but did not contain the answer, so no source was used.
    if NOT_FOUND_MESSAGE in answer:
        return {"answer": answer, "sources": [], "chunks": relevant_chunks}

    # Unique source filenames, in order of relevance
    sources = []
    for chunk in relevant_chunks:
        if chunk["source"] not in sources:
            sources.append(chunk["source"])

    return {"answer": answer, "sources": sources, "chunks": relevant_chunks}
