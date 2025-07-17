# Use Python 3.13 slim as the base image
FROM python:3.13-slim

# Set the working directory
WORKDIR /app

# Copy app folder
COPY app/ .

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download the BGE embedding model
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"

# Expose Gradio's default port
EXPOSE 7860

# Run the app
CMD ["python", "app.py"]