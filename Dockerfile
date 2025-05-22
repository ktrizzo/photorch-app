# Use an official Python image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies (optional, if needed by app)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy Streamlit app files into the container
COPY . .

# Expose Streamlit's default port
EXPOSE 8501

# Run the Streamlit app
CMD ["sh", "-c","echo '\n'; echo '\n\n======================================'; echo '🌱  PhoTorch App running at http://localhost:8501'; echo '======================================\n'; exec streamlit run app.py --server.port=8501 --server.address=0.0.0.0"]