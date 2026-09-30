# Document Question Answering – RAG System

A simple, end-to-end **Retrieval-Augmented Generation (RAG)** web application.
Upload your documents (PDF, DOCX, CSV, XLSX, TXT, JSON), ask questions in plain English, and get answers that are based **only** on your documents, along with the source file names.

Built with **FastAPI**, **sentence-transformers**, **FAISS**, and the **Groq** LLM API. It runs in **Docker** and can be deployed on **AWS EC2**.

---

## Overview

Large Language Models (LLMs) don't know what's inside *your* files, and when they don't know something they sometimes make things up.
This project solves that with RAG:

1. **Ingest:** uploaded documents are converted to text, split into chunks, turned into embedding vectors, and stored in a FAISS index.
2. **Retrieve:** a question is turned into a vector, and FAISS finds the 3 most similar chunks.
3. **Generate:** those chunks are given to the LLM as context, and it is instructed to answer *only* from that context.

---

## Architecture

```
                 ┌──────────────┐
                 │    User      │
                 └──────┬───────┘
                        │
                  Upload Document
                        │
                        ▼
               ┌─────────────────┐
               │ Document Loader │   loaders.py  (PDF, DOCX, CSV, XLSX, TXT, JSON)
               └────────┬────────┘
                        │
                        ▼
                  Text Chunks          rag.py  (1000 chars, 200 overlap)
                        │
                        ▼
               ┌─────────────────┐
               │   Embeddings    │   rag.py  (all-MiniLM-L6-v2, 384 numbers per chunk)
               │ MiniLM-L6-v2    │
               └────────┬────────┘
                        │
                        ▼
               ┌─────────────────┐
               │      FAISS      │   faiss_store/faiss.index + metadata.pkl
               │   Vector Index  │
               └────────┬────────┘
                        │
User Question ──────────┘
        │
        ▼
 Question Embedding
        │
        ▼
 Top-K Similar Chunks  (k = 3)
        │
        ▼
 Retrieved Context
        │
        ▼
      Groq LLM          (openai/gpt-oss-20b)
        │
        ▼
   Answer + Sources
```

---

## Features

- Upload one or more documents from a web page or the API
- Supports **6 file types**: PDF, DOCX, CSV, XLSX, TXT, JSON
- Fixed-size chunking with overlap (1000 / 200 characters)
- Semantic search with `sentence-transformers/all-MiniLM-L6-v2` embeddings and FAISS cosine similarity
- Answers grounded in the retrieved context, with **source file names** and the retrieved chunks
- Replies *"I couldn't find this information in the uploaded documents."* when the answer isn't there
- FAISS index saved to disk and reloaded when the app restarts
- Simple HTML frontend, no frontend framework
- Interactive API docs (Swagger UI) at `/docs`
- Dockerized and deployable on AWS EC2

---

## Tech Stack

| Part | Technology |
|---|---|
| Language | Python 3.11 |
| Web framework | FastAPI + Uvicorn |
| Document parsing | pypdf, python-docx, openpyxl, csv, json |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Vector search | FAISS (`faiss-cpu`) |
| LLM | Groq API (`openai/gpt-oss-20b`) |
| Config | python-dotenv (`.env` file) |
| Deployment | Docker on AWS EC2 |

---

## Project Structure

```
rag-document-qa/
├── app.py               # FastAPI app: the 4 endpoints
├── rag.py               # RAG pipeline: chunking, embeddings, FAISS, retrieval, LLM call
├── loaders.py           # Reads PDF/DOCX/CSV/XLSX/TXT/JSON into plain text
├── index.html           # The simple web page (HTML + CSS + JavaScript)
├── requirements.txt     # Python dependencies
├── Dockerfile           # How to build the Docker image
├── .env.example         # Template for your .env file
├── README.md
├── INTERVIEW_NOTES.md   # Beginner-friendly explanation + interview Q&A
├── sample_documents/    # Small demo files, one per supported format
├── data/                # Uploaded documents (created automatically)
└── faiss_store/         # faiss.index + metadata.pkl (created automatically)
```

---

## Local Setup

You need **Python 3.11** and a free **Groq API key** from <https://console.groq.com/keys>.

**1. Get the code**

```bash
git clone https://github.com/<your-username>/rag-document-qa.git
cd rag-document-qa
```

**2. Create a virtual environment.** This is a private folder of packages just for this project.

```bash
python3.11 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

**3. Install the dependencies**

```bash
pip install -r requirements.txt
```

This downloads FastAPI, FAISS, sentence-transformers (including PyTorch), and the other libraries. The first install takes a few minutes.

---

## Environment Variables

**4. Create your `.env` file** by copying the example:

```bash
cp .env.example .env
```

**5. Add your Groq API key.** Open `.env` and replace the placeholder:

```
GROQ_API_KEY=gsk_your_real_key_here
```

| Variable | Required | What it is |
|---|---|---|
| `GROQ_API_KEY` | Yes | Your key for the Groq LLM API. It is read with `python-dotenv` and never hard-coded. |

`.env` is listed in `.gitignore`, so your key is never pushed to GitHub.

---

## Running Locally

**6. Start the server**

```bash
uvicorn app:app --reload
```

What this command means:
- `uvicorn` is the web server that runs the FastAPI app.
- `app:app` means "in the file `app.py`, use the variable named `app`".
- `--reload` restarts the server automatically when you change the code (useful during development only).

The first start downloads the embedding model (about 90 MB), which takes a moment. When you see `Uvicorn running on http://127.0.0.1:8000`, the server is ready.

**7. Open the browser**

- **<http://localhost:8000>** opens the web page. `localhost` means "this computer" and `8000` is the port Uvicorn listens on.
- **<http://localhost:8000/docs>** opens the **Swagger UI**, an interactive API page that FastAPI generates automatically from the code. It lists every endpoint, shows what input each one expects, and has a **"Try it out"** button so you can call the API without writing any code.

**8. Upload a document.** Choose files, for example everything in `sample_documents/`, and click **Upload**.

**9. Ask a question**, for example *"What is the refund policy?"*, and click **Ask**.

---

## API Endpoints

| Method | Path | What it does |
|---|---|---|
| `GET` | `/` | Returns the HTML page |
| `GET` | `/health` | Returns `{"status": "ok"}`, which is useful to check that the server is running |
| `POST` | `/upload` | Uploads one or more files, saves them to `data/`, and rebuilds the FAISS index |
| `POST` | `/ask` | Takes `{"question": "..."}` and returns the answer, the source files, and the retrieved chunks |

### Error responses

Errors are returned as JSON in the form `{"detail": "..."}`:

| Situation | Status |
|---|---|
| Unsupported file type / unreadable file / file with no text | 400 |
| Empty question | 400 |
| No documents uploaded yet (no FAISS index) | 400 |
| `GROQ_API_KEY` missing | 500 |
| Groq API call failed (e.g. invalid key) | 502 |

---

## Example Usage

**Health check**

```bash
curl http://localhost:8000/health
```
```json
{"status": "ok"}
```

**Upload documents**

```bash
curl -X POST http://localhost:8000/upload \
  -F "files=@sample_documents/sample_faq.json" \
  -F "files=@sample_documents/sample_rag_notes.pdf"
```
```json
{
  "message": "Uploaded 2 file(s). The index now has 3 chunks.",
  "uploaded_files": ["sample_faq.json", "sample_rag_notes.pdf"],
  "errors": [],
  "total_chunks": 3
}
```

**Ask a question**

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the refund policy?"}'
```
```json
{
  "answer": "Annual plans can be refunded in full within 14 days of purchase. Monthly plans are not refundable.",
  "sources": ["sample_faq.json"],
  "chunks": [
    {"text": "company: Nimbus Cloud Services (fictional)\nfaq > question: What are the support hours? ...", "source": "sample_faq.json", "score": 0.482}
  ]
}
```

**A question that is not in the documents**

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Who won the 2018 football world cup?"}'
```
```json
{"answer": "I couldn't find this information in the uploaded documents.", "sources": [], "chunks": [...]}
```

If the question is unrelated to every document (best similarity below 0.2), the LLM is not called at all.
If the question is related but the answer isn't there (e.g. *"Who is the CEO of Nimbus Cloud Services?"*), the LLM itself replies with the same message and `sources` is empty.

---

## Docker Setup

Docker packages the app, Python, and all dependencies into one **image**, so it runs the same way on any machine.

**Build the image**

```bash
docker build -t rag-app .
```

`-t rag-app` gives the image the name `rag-app`. The first build takes several minutes because it installs PyTorch and downloads the embedding model.

**Run the container (quick test)**

```bash
docker run -p 8000:8000 --env-file .env rag-app
```

- `-p 8000:8000` connects port 8000 on your computer to port 8000 inside the container.
- `--env-file .env` passes your `GROQ_API_KEY` into the container. The `.env` file is *not* copied into the image (see `.dockerignore`).

**Run the container and keep your data (recommended)**

```bash
docker run -d --name rag-app -p 8000:8000 --env-file .env \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/faiss_store:/app/faiss_store" \
  rag-app
```

Anything written inside a container is lost when the container is deleted.
The two `-v` options **mount** folders from your machine into the container, so uploaded documents (`data/`) and the FAISS index (`faiss_store/`) are stored on the host disk and survive container restarts and re-creations.
`-d` runs the container in the background, and `--name rag-app` lets you manage it by name:

```bash
docker logs rag-app       # see the server output
docker stop rag-app       # stop it
docker start rag-app      # start it again (your documents and index are still there)
```

---

## AWS EC2 Deployment

Deployment flow:

```
Local computer ──git push──► GitHub ──git clone──► AWS EC2 ──docker build/run──► FastAPI RAG app
```

### 1. Push the project to GitHub

On your computer, from inside the project folder:

```bash
git init
git add .
git commit -m "Document Q&A RAG system"
git branch -M main
git remote add origin https://github.com/<your-username>/rag-document-qa.git
git push -u origin main
```

(Create the empty repository on github.com first. `.env`, `data/` and `faiss_store/` are ignored and will not be uploaded.)

### 2. Create an EC2 instance

EC2 is a virtual server that you rent from AWS.

1. Log in to the AWS Console, search for **EC2**, and click **Launch instance**.
2. **Name:** `rag-app`.
3. **AMI (operating system):** **Ubuntu Server 24.04 LTS**, architecture **64-bit (x86)**. (x86 avoids any compatibility surprises with FAISS and PyTorch.)
4. **Instance type:** **t3.medium** (2 vCPU, 4 GB RAM) is recommended.
   PyTorch and the embedding model need roughly 1–1.5 GB of RAM, and the Docker build needs more, so 1 GB instances (t2.micro / t3.micro) are too small. A t3.small (2 GB) can work but may be tight.
   *t3.medium is not free-tier. Stop the instance when you're not using it to avoid charges.*
5. **Key pair:** click **Create new key pair**, choose type RSA and format `.pem`, and download it. You need this file to log in.
6. **Network settings → Edit**:
   - Keep the rule **SSH, port 22**, source **My IP**.
   - Click **Add security group rule**: type **Custom TCP**, port **8000**, source **Anywhere (0.0.0.0/0)**, or **My IP** to keep it private.
     *This opens port 8000 in the security group (the instance's firewall), so your browser can reach the app.*
7. **Storage:** change it to **20 GiB** (Docker + PyTorch needs more than the 8 GiB default).
8. Click **Launch instance**, then copy the instance's **Public IPv4 address** from its details page.

### 3. Connect through SSH

On your computer:

```bash
chmod 400 ~/Downloads/your-key.pem
ssh -i ~/Downloads/your-key.pem ubuntu@<EC2_PUBLIC_IP>
```

- `chmod 400` makes the key readable only by you, and SSH refuses keys that are too open.
- `ssh -i key.pem ubuntu@IP` logs you into the server as the default `ubuntu` user.

### 4. Install Docker and Git on the server

```bash
sudo apt update
sudo apt install -y docker.io git
sudo systemctl enable --now docker
sudo usermod -aG docker ubuntu
exit
```

- `apt update` refreshes Ubuntu's package list, and `apt install` installs Docker and Git.
- `systemctl enable --now docker` starts Docker now and on every reboot.
- `usermod -aG docker ubuntu` lets the `ubuntu` user run Docker without `sudo`.
- `exit` logs out, and **you must reconnect with the same `ssh` command** for the group change to apply.

### 5. Clone the repository

```bash
git clone https://github.com/<your-username>/rag-document-qa.git
cd rag-document-qa
```

### 6. Create the `.env` file

```bash
cp .env.example .env
nano .env
```

Replace the placeholder with your real `GROQ_API_KEY`, then save (`Ctrl+O`, `Enter`) and exit (`Ctrl+X`).

### 7. Build the Docker image

```bash
docker build -t rag-app .
```

This takes 5–10 minutes the first time.

### 8. Run the container

```bash
mkdir -p data faiss_store
docker run -d --name rag-app --restart unless-stopped \
  -p 8000:8000 --env-file .env \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/faiss_store:/app/faiss_store" \
  rag-app
```

- `-v ...` stores uploaded files and the FAISS index on the EC2 disk, so they stay when the container restarts or is rebuilt.
- `--restart unless-stopped` makes Docker restart the app automatically if it crashes or the server reboots.

Check that it is running:

```bash
docker ps
curl http://localhost:8000/health
```

### 9. Open the app

In your browser go to:

```
http://<EC2_PUBLIC_IP>:8000
http://<EC2_PUBLIC_IP>:8000/docs
```

If the page doesn't load, check that the security group allows port 8000 (step 2.6), and that you used `http://`, not `https://`.

### Updating the app after code changes

```bash
cd rag-document-qa
git pull
docker build -t rag-app .
docker stop rag-app && docker rm rag-app
# then run the same "docker run" command from step 8 again
```

Your uploaded documents and index stay safe because they live in the mounted folders.

> **Note:** the public IP changes if you *stop* and *start* the instance. Attach an **Elastic IP** if you need a fixed address.

---

## Limitations

- **Rebuilds the whole index on every upload.** Simple and always consistent, but slow for large document collections.
- **Fixed-size character chunking** can cut a sentence or a word in half at chunk boundaries.
- **No OCR:** scanned PDFs (images of text) have no extractable text and are rejected.
- DOCX loading reads paragraphs only (text inside Word tables is not extracted).
- **No authentication:** anyone who can reach the URL can upload documents and ask questions.
- **Single shared index** for all users, with no per-user document separation and no way to delete a document from the UI.
- Uploading a file with the same name as an existing file replaces it.
- Runs over plain HTTP (no HTTPS) and uses a single process with in-memory state.
- The similarity threshold (0.2) and `top_k = 3` are fixed values chosen for small document sets.

## Future Improvements

- Add new chunks to the index incrementally instead of rebuilding everything
- Smarter chunking (split on paragraphs/sentences, e.g. LangChain's `RecursiveCharacterTextSplitter`)
- Endpoints to list and delete uploaded documents
- OCR support for scanned PDFs and table extraction for DOCX
- User authentication and per-user document collections
- HTTPS with a domain name and a reverse proxy (e.g. Nginx)
- Store documents in S3 and use a managed vector database for larger scale
- An evaluation set of questions to measure answer quality
