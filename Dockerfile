FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    gcc \
    g++ \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install numpy first separately
RUN pip install --no-cache-dir numpy==1.21.2

# Uninstall existing Flask and Werkzeug
RUN pip uninstall -y flask werkzeug

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create directories
RUN mkdir -p model templates

# Copy files explicitly
COPY MTLiensWebApp.py .
COPY templates/index.html templates/

# Debug: Print directory contents
RUN echo "Main directory:" && ls -la && \
    echo "\nTemplates directory:" && ls -la templates

# Expose port
EXPOSE 5000

# Run the application
CMD ["python", "MTLiensWebApp.py"]