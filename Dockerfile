# Use lightweight Python base
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Install system-level dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project files into the container
COPY . /app

# Upgrade pip and install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose Streamlit's default port
EXPOSE 8501

# Set environment variable for Hugging Face (optional)
# ENV HUGGINGFACE_HUB_TOKEN=your_token_here

# Run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
