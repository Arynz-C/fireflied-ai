FROM ubuntu:latest

# Install dependencies
RUN apt update && apt install -y curl && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Set up workdir
WORKDIR /workspace
