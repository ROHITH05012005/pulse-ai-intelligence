# Stage 1: Build the React frontend
FROM node:20-slim AS frontend-build
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# Stage 2: Setup the Python backend
FROM python:3.11-slim
WORKDIR /app

# Install system dependencies for torch/transformers
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements and install
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY backend/main.py ./

# Copy the built frontend from Stage 1
COPY --from=frontend-build /app/frontend/dist ./static

# Set environment variables
ENV PORT=8000
ENV MODEL_NAME=j-hartmann/emotion-english-distilroberta-base

# Expose the port
EXPOSE 8000

# Start the server
CMD ["python", "main.py"]
