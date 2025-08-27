FROM python:3.11-slim

# System dependencies for OCR (Tesseract) and pdf2image (poppler), plus OpenCV runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
  && python -m spacy download en_core_web_sm

# Copy application source
COPY . .

ENV PYTHONPATH=src
ENV PORT=8080

# Default command: start web server
CMD ["uvicorn", "ea_importer.web.app:app", "--host", "0.0.0.0", "--port", "8080", "--proxy-headers"]

