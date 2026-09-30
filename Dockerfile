# Small official Python 3.11 image
FROM python:3.11-slim

# All app files live in /app inside the container
WORKDIR /app

# Install the CPU-only version of PyTorch first.
# (sentence-transformers needs PyTorch; the default version includes large GPU libraries we don't need.)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install the rest of the Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download the embedding model during the build, so the container starts quickly
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# The model is now inside the image, so don't contact Hugging Face when the app starts
ENV HF_HUB_OFFLINE=1

# Copy the application code
COPY . .

EXPOSE 8000

# 0.0.0.0 makes the server reachable from outside the container
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
