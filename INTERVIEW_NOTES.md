# Interview Notes: Document Question Answering RAG System

These notes explain **this exact project** in simple language.
Everything here describes what the code actually does. File and function names are included so you can point to them.

---

## 1. What problem does the project solve?

People have documents (PDFs, Word files, spreadsheets, JSON, text files) and want to ask questions about them in plain English instead of reading everything.
A normal LLM like ChatGPT has never seen *your* files, so it can't answer questions about them, and it may invent an answer.

This project lets a user **upload documents and ask questions**, and it answers **using only the content of those documents**, telling the user **which files the answer came from**.

---

## 2. What is RAG?

**RAG = Retrieval-Augmented Generation.**

- **Retrieval:** find the pieces of your documents that are relevant to the question.
- **Augmented:** add those pieces to the prompt.
- **Generation:** the LLM generates an answer using those pieces.

Analogy: an **open-book exam**. Instead of answering from memory, the LLM is handed the right pages of the book and told to answer from them.

---

## 3. Why RAG instead of directly asking an LLM?

| Asking an LLM directly | Using RAG |
|---|---|
| The LLM doesn't know your private documents | Relevant parts of your documents are put in the prompt |
| It may hallucinate (make up) an answer | It's told to answer only from the given context |
| You can't tell where the answer came from | We return the **source file names** and the chunks used |
| Pasting whole documents into the prompt is slow, costly, and limited by context size | Only the **3 most relevant chunks** are sent |

We also don't need to **retrain** the LLM. We just upload new documents.

---

## 4. Complete request flow

**Upload flow (`POST /upload`)**

```
User picks files → app.py saves them to data/
  → loaders.py extracts text from each file
  → rag.py splits text into chunks (1000 chars, 200 overlap)
  → each chunk → embedding vector (384 numbers) with all-MiniLM-L6-v2
  → all vectors go into a FAISS index
  → index saved to faiss_store/faiss.index, chunk text + filename saved to faiss_store/metadata.pkl
```

**Question flow (`POST /ask`)**

```
User question → embedding vector (same model)
  → FAISS finds the 3 most similar chunk vectors
  → drop chunks whose similarity < 0.2
  → if none are left: return "I couldn't find this information in the uploaded documents."
  → otherwise: join the chunks into a "context"
  → send system prompt + context + question to Groq (openai/gpt-oss-20b)
  → if the LLM says it couldn't find the answer: return it with an empty sources list
  → return { answer, sources, chunks }
```

---

## 5. What does document chunking mean?

**Chunking = cutting a long text into smaller pieces.**

A 50-page PDF becomes many small chunks of about 1000 characters each.
Each chunk is embedded and searched separately, so we can find the **specific part** of the document that answers the question.

In the code (`split_into_chunks` in `rag.py`): take characters 0–1000, then 800–1800, then 1600–2600, and so on.

**Why do we need it?**
1. An embedding of a whole document is too "blurry", because it mixes many topics. A small chunk is about one thing, so matching is more accurate.
2. We send only a few chunks to the LLM, not whole documents, which keeps the prompt small.
3. The embedding model only reads a limited amount of text at once (MiniLM reads about 256 word-pieces and ignores the rest), so very long inputs lose information.

---

## 6. Why chunk size = 1000?

It's a common, balanced starting value (also used in the reference tutorial):
- **Big enough** to hold a complete idea (roughly a paragraph or two, about 150–200 words).
- **Small enough** that the chunk stays focused on one topic and 3 chunks fit easily in the prompt.

Too small (e.g., 100 characters) loses context, because a sentence gets separated from its meaning.
Too large (e.g., 5000 characters) makes each chunk mix many topics, which makes retrieval less precise.

*Honest note:* MiniLM only reads about the first 256 word-pieces (roughly 1000 characters of English), so 1000 characters also roughly matches what the model can actually "see".

---

## 7. What does chunk overlap = 200 mean?

Each chunk **repeats the last 200 characters of the previous chunk**.

```
Chunk 1: characters    0 – 1000
Chunk 2: characters  800 – 1800   (800–1000 is shared with chunk 1)
Chunk 3: characters 1600 – 2600
```

**Why?** A fixed cut can split a sentence in the middle. With overlap, a sentence near the border appears whole in at least one chunk, so it can still be found.

---

## 8. What are embeddings?

An **embedding** is a list of numbers (a **vector**) that represents the **meaning** of a text.

- Our model turns any text into **384 numbers**.
- Texts with **similar meaning** get vectors that are **close together**, even if they use different words.
  Example: *"How do I get my money back?"* is close to *"refund policy"*.

This is why the search is **semantic** (by meaning), not keyword matching.

In the code (`embed_texts` in `rag.py`) we call `embedding_model.encode(texts, normalize_embeddings=True)`. Normalizing makes every vector length 1, so we can use cosine similarity (see question 11).

---

## 9. Why all-MiniLM-L6-v2?

- **Small and fast:** about 90 MB, runs well on a **CPU**, so there's no GPU and no paid API for embeddings.
- **Good quality** for general English semantic search. It's one of the most widely used sentence-transformers models.
- **384-dimension vectors**, which are small, so the index uses little memory and search is fast.
- **Free and runs locally**, so document text isn't sent to a third party just to create embeddings.

---

## 10. What is FAISS?

**FAISS (Facebook AI Similarity Search)** is an open-source library from Meta for **searching vectors quickly**.
You give it many vectors, and then for a new vector it tells you which stored vectors are the **most similar**.

In this project:
- We use `faiss.IndexFlatIP`. **Flat** means it compares the question with **every** stored vector (exact search, which is fine for small data). **IP** means **Inner Product**, which equals **cosine similarity** because our vectors are normalized.
- The index is saved to `faiss_store/faiss.index`.
- FAISS stores only numbers, so we keep a separate list in `faiss_store/metadata.pkl` with the **chunk text and source filename**. Position `i` in the index matches `metadata[i]`.

---

## 11. What does vector similarity search mean?

1. Every chunk is stored as a vector.
2. The question is turned into a vector **with the same model**.
3. We compute how similar the question vector is to each chunk vector.
4. The chunks with the highest similarity are the most relevant.

We use **cosine similarity**, which measures the angle between two vectors:
- close to **1** means very similar meaning
- close to **0** means unrelated

Because all vectors have length 1, cosine similarity is simply the inner product, and that's what `IndexFlatIP` computes.

---

## 12. What does top_k = 3 mean?

**"Give me the 3 most similar chunks."** (`TOP_K = 3` in `rag.py`)

Those 3 chunks (after the similarity check) become the context for the LLM.

---

## 13. What does the LLM receive?

Two messages (`generate_answer` in `rag.py`):

**System message** (the rules):
```
You are a document question-answering assistant.
Answer the question using only the provided context.
Do not use outside knowledge and do not make up information.
If the answer is not present in the context, reply exactly:
"I couldn't find this information in the uploaded documents."
```

**User message** (the data):
```
Context:
[Source: sample_faq.json]
...chunk text...

[Source: sample_ai.txt]
...chunk text...

Question:
What is the refund policy?
```

We also set `temperature=0`, so the model gives its most likely, predictable answer instead of a "creative" one.

---

## 14. How is hallucination reduced?

Hallucination is when the LLM confidently makes up information. The project reduces it in **four simple ways**:

1. **Grounding:** the LLM is given the actual document text as context.
2. **Strict instructions:** the system prompt says to use *only* the context, not to invent anything, and what exact sentence to reply if the answer isn't there.
3. **Similarity threshold:** if no retrieved chunk has similarity ≥ **0.2**, we **don't call the LLM at all** and directly return *"I couldn't find this information in the uploaded documents."*
4. **Temperature 0:** less random, more predictable output.

We also return **sources and retrieved chunks**, so a user can check the answer themselves.

*Honest limitation:* this **reduces** hallucination but can't guarantee zero. The LLM could still misread the context.

---

## 15. Why FastAPI?

- **Simple:** an endpoint is just a Python function with a decorator, like `@app.post("/ask")`.
- **Automatic validation:** `QuestionRequest(BaseModel)` makes FastAPI check that the request body has a `question` string, and it returns an error automatically if not.
- **Automatic docs:** Swagger UI at `/docs` lets anyone test the API in the browser.
- **Popular in ML/AI** projects for serving Python models, and it's easy to put in Docker.

---

## 16. What does each endpoint do?

| Endpoint | What it does | Code |
|---|---|---|
| `GET /` | Returns the HTML page (`index.html`) | `home()` |
| `GET /health` | Returns `{"status": "ok"}`, a quick "is the server alive?" check | `health()` |
| `POST /upload` | Receives files, checks the type, saves to `data/`, checks that text can be extracted, then rebuilds the FAISS index | `upload_documents()` |
| `POST /ask` | Receives `{"question": "..."}`, validates it, retrieves the top 3 chunks, calls the LLM, and returns answer + sources + chunks | `ask_question()` |

---

## 17. Why Docker?

Docker packages **the code + Python 3.11 + all libraries + the embedding model** into one **image**.

- "It works on my machine" becomes "it works on **every** machine". The EC2 server runs exactly what I tested.
- One command to build (`docker build`) and one to run (`docker run`).
- There's no need to install Python or libraries manually on the server.

Details in our `Dockerfile`:
- It installs the **CPU-only** version of PyTorch, which is much smaller than the default GPU version.
- It downloads the embedding model **during the build**, so the container starts quickly, and sets `HF_HUB_OFFLINE=1` so the app doesn't contact Hugging Face at startup.
- `.env` is **not** copied into the image (`.dockerignore`). The key is passed at runtime with `--env-file .env`.

---

## 18. Why EC2?

- **EC2** is a virtual server on AWS: a Linux computer in the cloud with a public IP.
- It's the **simplest real cloud deployment**. I control the machine, install Docker, and run the container, much like running it on my laptop.
- It has a **persistent disk (EBS)**, so the uploaded files and FAISS index are kept.
- I chose an **x86_64** Ubuntu instance with **4 GB RAM** (t3.medium), because PyTorch + the embedding model need more than 1 GB.

---

## 19. How does deployment work?

```
Local computer → git push → GitHub → git clone on EC2 → docker build → docker run → open http://<public-ip>:8000
```

1. Push code to GitHub (without `.env`).
2. Launch an Ubuntu EC2 instance, and open **port 22** (SSH) and **port 8000** (app) in the **security group** (the AWS firewall).
3. SSH into the server and install Docker and Git.
4. `git clone` the repo and create `.env` with the Groq key.
5. `docker build -t rag-app .`
6. `docker run -d -p 8000:8000 --env-file .env -v ./data:/app/data -v ./faiss_store:/app/faiss_store --restart unless-stopped rag-app`
7. Open `http://<EC2 public IP>:8000`.

**Why the `-v` volume mounts?** Files written inside a container disappear when the container is removed. Mounting `data/` and `faiss_store/` from the EC2 disk means uploaded documents and the index **survive container restarts and rebuilds**.

---

## 20. What happens when a new file is uploaded?

Step by step (`upload_documents` in `app.py` → `rebuild_index` in `rag.py`):

1. The filename is cleaned with `os.path.basename()` (so a name like `../../x` can't write outside `data/`).
2. If the extension isn't one of the 6 supported types, the file is **rejected** with an error message.
3. The file is saved into `data/`.
4. `load_document()` tries to extract text. If it fails, or no text is found (e.g. a scanned PDF), the file is **deleted** and an error is reported.
5. `rebuild_index()` then:
   - reads **every** file in `data/`
   - splits each into chunks and records `{"text": chunk, "source": filename}`
   - embeds all chunks
   - creates a **new** FAISS index, adds all vectors, and saves `faiss.index` + `metadata.pkl`
   - replaces the in-memory index
6. The response says how many files were uploaded and how many chunks the index now has.

**Why rebuild everything instead of adding?** It's the simplest approach that's always correct. `data/` is the single source of truth, and re-uploading a file with the same name doesn't create duplicate chunks. The trade-off is that it gets slower as the number of documents grows.

---

## 21. What happens when a user asks a question?

Step by step (`ask_question` in `app.py` → `answer_question` in `rag.py`):

1. FastAPI checks the JSON body has a `question` field (via `QuestionRequest`).
2. We check: the question isn't empty, an index exists (documents were uploaded), and `GROQ_API_KEY` is set. Otherwise we return a clear JSON error.
3. `retrieve_chunks()`: embed the question and call `index.search(question_vector, 3)`, which gives the 3 closest chunks with their similarity scores.
4. Keep only chunks with score ≥ 0.2. If none remain, return the "couldn't find" message **without calling the LLM**.
5. `generate_answer()`: build the context from the chunks (each labeled with its source) and call Groq.
6. Return JSON:

```json
{
  "answer": "...",
  "sources": ["sample_faq.json"],
  "chunks": [{"text": "...", "source": "sample_faq.json", "score": 0.41}]
}
```

---

# Likely Interview Questions (with simple answers)

**1. Explain your RAG project.**
It's a web app where you upload documents (PDF, DOCX, CSV, XLSX, TXT, JSON) and ask questions about them. When you upload, I extract the text, split it into 1000-character chunks, convert each chunk into an embedding with the all-MiniLM-L6-v2 model, and store the vectors in a FAISS index. When you ask a question, I embed the question, use FAISS to find the 3 most similar chunks, and send them with the question to an LLM on Groq, which answers only from that context. The API returns the answer and the source file names. It's built with FastAPI, containerized with Docker, and deployed on an AWS EC2 instance.

**2. What is RAG?**
Retrieval-Augmented Generation. Before the LLM answers, we retrieve relevant text from our own documents and add it to the prompt, so the LLM generates an answer based on that text. It's like an open-book exam.

**3. Why did you use RAG?**
The LLM doesn't know the content of the user's documents. RAG gives it the relevant parts at question time, without retraining the model, and lets me show which documents the answer came from. It also reduces made-up answers.

**4. What is chunking?**
Splitting a long document into smaller pieces. I use fixed-size chunks of 1000 characters, with 200 characters of overlap between neighbouring chunks.

**5. Why do we need chunking?**
So we can find the specific part of a document that answers the question. A small chunk has a focused meaning, so its embedding matches questions better. It also keeps the prompt small, because we send 3 chunks, not whole files. And the embedding model can only read a limited amount of text at once.

**6. What are embeddings?**
Lists of numbers that represent the meaning of text. Texts with similar meaning have vectors that are close together. My model produces 384 numbers per text.

**7. What is FAISS?**
A library from Meta for fast similarity search over vectors. I store all chunk vectors in a FAISS index, and it finds the vectors most similar to the question vector.

**8. How does similarity search work?**
The question is converted into a vector using the same embedding model. FAISS compares it with the stored chunk vectors using the inner product. Since my vectors are normalized, that equals cosine similarity. The highest scores are the most relevant chunks.

**9. What is top-k retrieval?**
Returning the k most similar results. I use k = 3, so the 3 most similar chunks are retrieved.

**10. Why did you choose 3?**
It's a balance. One chunk may miss part of the answer, and too many chunks add irrelevant text that can confuse the LLM and make the prompt longer. 3 chunks of 1000 characters is enough context for the small documents this app is built for. It's a setting (`TOP_K`) that can be tuned.

**11. What is a vector database?**
A system designed to store vectors and search them by similarity, usually with extra database features like persistence, filtering, updates/deletes, and scaling across servers. Examples are Pinecone, Chroma, Weaviate, and Milvus.

**12. Is FAISS a vector database?**
Not exactly. FAISS is a **vector search library**, not a full database. It doesn't store the original text, has no server, and has no built-in metadata filtering. That's why I store the chunk text and filenames separately in `metadata.pkl`, and save/load the index file myself.

**13. Why did you use FAISS?**
It's free, fast, runs locally in the same Python process, and needs no separate server. For a small single-server project, saving the index to a file on disk is enough.

**14. Why use sentence-transformers?**
It's a simple Python library for turning sentences into embeddings with pre-trained models. It's one line to load a model and one line to encode text, and it runs locally without an API key.

**15. Why MiniLM?**
all-MiniLM-L6-v2 is small (about 90 MB), fast on a CPU, and gives good quality for English semantic search. Its 384-dimensional vectors keep the index small. It's a very common default for RAG projects.

**16. What happens when a document is uploaded?**
The app checks the file type, saves it to `data/`, and checks that text can be extracted (unreadable or empty files are deleted and reported). Then it rebuilds the FAISS index from all documents in `data/`: extract text, chunk, embed, add to a new index, and save `faiss.index` and `metadata.pkl` to disk.

**17. What happens when a question is asked?**
The question is validated, embedded, and searched in FAISS for the top 3 chunks. Chunks below a similarity of 0.2 are dropped. If nothing remains, the app says it couldn't find the information. Otherwise, the chunks plus the question go to the LLM, and the app returns the answer, the source filenames, and the chunks.

**18. How does the LLM get the document information?**
Through the prompt. The LLM never sees the whole document or the FAISS index. I paste the text of the retrieved chunks, each labeled with its source filename, into the message under "Context:", followed by the question.

**19. How do you reduce hallucinations?**
Four things: the LLM gets the real document text as context; the system prompt tells it to use only that context and not invent anything; if no chunk is similar enough (score below 0.2), I skip the LLM and return "I couldn't find this information"; and I use temperature 0. I also return the sources and chunks so users can verify.

**20. What happens if the answer isn't in the document?**
Two layers. If the question isn't similar to anything in the documents, the similarity check catches it and returns "I couldn't find this information in the uploaded documents." without calling the LLM. If some chunks pass the check but don't contain the answer, the system prompt tells the LLM to reply with that same sentence, and then the app returns an empty sources list, because no document was actually used. For example, "Who is the CEO of Nimbus Cloud Services?" matches the company FAQ, but the CEO isn't mentioned, so the LLM says it couldn't find it.

**21. Why use FastAPI?**
It's simple (endpoints are normal Python functions), validates request data automatically using type hints and Pydantic, generates interactive API docs at `/docs`, and is widely used for serving ML models in Python.

**22. What is an API endpoint?**
A specific URL + HTTP method that the server responds to. For example, `POST /ask` receives a question and returns an answer. My app has four: `GET /`, `GET /health`, `POST /upload`, `POST /ask`.

**23. What is Uvicorn?**
The web server that runs the FastAPI application. It listens on a port (8000), receives HTTP requests, and passes them to FastAPI. `uvicorn app:app` means "run the `app` object from `app.py`".

**24. Why Docker?**
To package the app with Python, all dependencies, and the embedding model into one image, so it runs the same way on my laptop and on EC2. Deployment becomes `docker build` and `docker run`.

**25. How did you deploy it on AWS?**
I launched an Ubuntu x86 EC2 instance (t3.medium), opened ports 22 and 8000 in the security group, SSHed in, installed Docker and Git, cloned my GitHub repo, created the `.env` file with the Groq key, built the image, and ran the container with port 8000 mapped, the data folders mounted as volumes, and `--restart unless-stopped`. Then I accessed it at `http://<public-ip>:8000`.

**26. Where is the FAISS index stored?**
In the `faiss_store/` folder: `faiss.index` (the vectors) and `metadata.pkl` (chunk text + source filename for each vector). It's loaded into memory when the app starts. On EC2, that folder is mounted from the host disk into the container, so it survives container restarts.

**27. Why do you need `metadata.pkl`?**
FAISS only stores vectors and returns positions (like "vector #5"). `metadata.pkl` is a list where item #5 holds that chunk's text and source file, so I can build the context and return the source names.

**28. How do you keep the API key safe?**
It's stored in a `.env` file that's read with python-dotenv. `.env` is in `.gitignore` (not pushed to GitHub) and in `.dockerignore` (not baked into the image). Docker receives it at runtime with `--env-file`.

**29. What file types do you support, and how?**
PDF (pypdf), DOCX (python-docx), CSV (Python's csv module), XLSX (openpyxl), TXT (normal file reading), and JSON (json module). `load_document()` looks at the file extension and calls the right loader. Every loader returns the same shape: `{"text": ..., "source": filename}`. For CSV and Excel, each row becomes a line like `name: Asha Rao, department: Engineering`, so column names stay next to their values. JSON is turned into readable lines like `faq > answer: Support is available...` instead of raw brackets and quotes. When I tested it, this raised the similarity score for "What are the support hours?" from about 0.20 to 0.37, because embeddings work best on natural text.

**30. What are the limitations of your system?**
- The whole index is rebuilt on every upload, which is slow for many documents.
- Character-based chunking can cut sentences at chunk borders.
- There's no OCR for scanned PDFs, and DOCX tables aren't extracted.
- There's no authentication, and all users share one index.
- There's no way to delete a document from the UI.
- It's a single server with plain HTTP.
- The similarity threshold is a fixed value I chose by testing, and it may not suit every kind of document.

**31. What would you improve in a production version?**
Add new documents to the index incrementally instead of rebuilding; use smarter chunking that respects paragraphs and sentences; add authentication and per-user document collections; add endpoints to list and delete documents; add OCR; use HTTPS; store files in S3 and consider a managed vector database for larger data; and build a small evaluation set of questions to measure answer quality before and after changes.

**32. How did you choose the similarity threshold of 0.2?**
By testing. I asked questions whose answers are in the sample documents and questions that have nothing to do with them, and looked at the top similarity scores. Relevant questions scored roughly 0.26–0.57, and unrelated ones (like "What is the capital of Australia?") scored about 0.15 or lower. 0.2 sits in the gap. It's a simple heuristic that works for these documents, not a universal value.

**33. How did you test the project?**
End to end through the API: `/health`; uploading all six file types; one question per file whose answer is in that file (checking both the answer and the returned source); questions whose answers are not in the documents; and error cases (empty question, unsupported file type, corrupted PDF, empty file, invalid JSON, missing or invalid API key). I also checked that the FAISS index survives deleting and re-creating the Docker container when the folders are mounted as volumes.

**34. Did testing find any problems?**
Yes. The question "What are the support hours?" returned "not found" even though the answer was in the JSON file. The JSON was being indexed as raw text full of brackets and quotes, and its similarity score (about 0.20) fell just below the threshold. I changed the JSON loader to turn the data into readable lines like `faq > answer: Support is available...`, and the score rose to about 0.37. The lesson: how you convert documents to text directly affects retrieval quality.

---

## Quick numbers to remember

| Setting | Value | Where |
|---|---|---|
| Chunk size | 1000 characters | `CHUNK_SIZE` in `rag.py` |
| Chunk overlap | 200 characters | `CHUNK_OVERLAP` |
| Embedding model | all-MiniLM-L6-v2 (384 dims) | `EMBEDDING_MODEL_NAME` |
| FAISS index type | `IndexFlatIP` (exact, cosine via normalized vectors) | `rebuild_index()` |
| Top-k | 3 | `TOP_K` |
| Similarity threshold | 0.2 | `MIN_SIMILARITY` |
| LLM | `openai/gpt-oss-20b` on Groq, temperature 0 | `LLM_MODEL_NAME` |
| Port | 8000 | Uvicorn / Docker |
