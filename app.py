"""
FastAPI web server for the Document Question Answering RAG system.

Endpoints:
    GET  /        -> the HTML page
    GET  /health  -> {"status": "ok"}
    POST /upload  -> save document(s) and rebuild the FAISS index
    POST /ask     -> answer a question using the uploaded documents
"""

import os
import shutil

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

import rag
from loaders import SUPPORTED_EXTENSIONS, is_supported_file, load_document

app = FastAPI(title="Document Q&A RAG System")

# Runs once when the server starts
os.makedirs(rag.DATA_DIR, exist_ok=True)
rag.load_index()
if not os.getenv("GROQ_API_KEY"):
    print("WARNING: GROQ_API_KEY is not set. /ask will not work until you add it to .env")


class QuestionRequest(BaseModel):
    question: str


@app.get("/")
def home():
    return FileResponse("index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
def upload_documents(files: list[UploadFile] = File(...)):
    uploaded_files = []
    errors = []

    for file in files:
        filename = os.path.basename(file.filename or "")  # basename() blocks paths like "../../x"

        if not filename:
            errors.append("A file was sent without a filename.")
            continue
        if not is_supported_file(filename):
            errors.append(f"{filename}: unsupported file type. Supported: {', '.join(SUPPORTED_EXTENSIONS)}")
            continue

        # Save the uploaded file into data/
        file_path = os.path.join(rag.DATA_DIR, filename)
        with open(file_path, "wb") as saved_file:
            shutil.copyfileobj(file.file, saved_file)

        # Make sure we can actually read text from it before indexing
        try:
            document = load_document(file_path)
        except Exception as error:
            os.remove(file_path)
            errors.append(f"{filename}: could not be read ({error})")
            continue
        if not document["text"].strip():
            os.remove(file_path)
            errors.append(f"{filename}: no text found in this file")
            continue

        uploaded_files.append(filename)

    if not uploaded_files:
        raise HTTPException(status_code=400, detail=" | ".join(errors))

    total_chunks = rag.rebuild_index()

    return {
        "message": f"Uploaded {len(uploaded_files)} file(s). The index now has {total_chunks} chunks.",
        "uploaded_files": uploaded_files,
        "errors": errors,
        "total_chunks": total_chunks,
    }


@app.post("/ask")
def ask_question(request: QuestionRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if rag.index is None:
        raise HTTPException(status_code=400, detail="No documents have been indexed yet. Please upload a document first.")
    if not os.getenv("GROQ_API_KEY"):
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is missing. Add it to your .env file and restart the app.")

    try:
        return rag.answer_question(question)
    except Exception as error:
        raise HTTPException(status_code=502, detail=f"The LLM request failed: {error}")
