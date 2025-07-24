# Use Apify's Python base image
FROM apify/actor-python:3.13

# Install system dependencies for Tesseract OCR
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-spa \
    tesseract-ocr-osd \
    && rm -rf /var/lib/apt/lists/*

# Set Tesseract to use multiple CPU cores efficiently
ENV OMP_THREAD_LIMIT=4

# Copy requirements.txt
COPY requirements.txt ./

# Install Python packages
RUN echo "Python version:" \
 && python --version \
 && echo "Pip version:" \
 && pip --version \
 && echo "Installing dependencies:" \
 && pip install -r requirements.txt \
 && echo "All installed Python packages:" \
 && pip freeze

# Copy source code
COPY . ./

# Compile Python code for syntax checking
RUN python3 -m compileall -q src/

# Create non-root user
RUN useradd --create-home apify && \
    chown -R apify:apify ./
USER apify

# Run the Actor
CMD ["python3", "-m", "src"]
