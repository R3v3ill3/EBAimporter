# EA Importer Quick Start Guide

This guide will get you up and running with the EA Importer system in minutes.

## 🚀 Installation (5 minutes)

### Option 1: Automated Installation
```bash
git clone <repository-url>
cd EBAimporter
chmod +x scripts/install.sh
./scripts/install.sh
```

### Option 2: Manual Installation
```bash
# 1. Clone repository
git clone <repository-url>
cd EBAimporter

# 2. Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install core dependencies
pip install PyPDF2 pdfplumber scikit-learn pandas sqlalchemy click rich typer

# 4. Create data directories
mkdir -p data/eas/{raw,text,clauses,fp}
mkdir -p data/{reports,instances,families}

# 5. Test installation
python3 scripts/basic_test.py
```

## 📄 Process Your First PDFs (2 minutes)

### Step 1: Add PDF Files
```bash
# Copy your EA PDFs to the raw directory
cp /path/to/your/EA_files/*.pdf data/eas/raw/

# Or download from the web
# Place URLs in a CSV file (see data/examples/sample_ea_urls.csv)
```

### Step 2: Run the Ingestion Pipeline
```bash
# Basic ingestion (without external dependencies)
python3 scripts/demo.py

# Or use the full CLI (if dependencies installed)
python3 -m ea_importer.cli ingest run data/eas/raw/
```

### Step 3: Check Results
```bash
# View generated files
ls data/eas/text/     # Cleaned text files
ls data/eas/clauses/  # Segmented clauses (JSONL format)
ls data/eas/fp/       # Document fingerprints

# View processing statistics
cat data/reports/ingest_runs/*.csv
```

## 🔍 Analyze Document Similarity (1 minute)

```bash
# Run clustering analysis
python3 -c "
import sys, json
sys.path.append('src')
from ea_importer.utils.fingerprinter import create_text_fingerprinter
from ea_importer.pipeline.clustering import create_ea_clusterer

# Load fingerprints and run clustering
# (See full example in scripts/demo.py)
"
```

## 📊 Example Outputs

### 1. Cleaned Text (`data/eas/text/EA_example.txt`)
```
1. INTRODUCTION

This Enterprise Agreement covers employees of XYZ Company.

2. DEFINITIONS

2.1 In this agreement:
(a) "Employee" means a person employed under this agreement
(b) "Employer" means XYZ Company Pty Ltd

3. CLASSIFICATION AND RATES

The minimum rates of pay are:
Level 1: $25.50 per hour
Level 2: $28.75 per hour
```

### 2. Segmented Clauses (`data/eas/clauses/EA_example.jsonl`)
```json
{"ea_id": "EA_example", "clause_id": "1", "heading": "INTRODUCTION", "text": "This Enterprise Agreement covers employees of XYZ Company.", "path": ["INTRODUCTION"], "level": 1}
{"ea_id": "EA_example", "clause_id": "2.1.a", "heading": "Employee definition", "text": "\"Employee\" means a person employed under this agreement", "path": ["DEFINITIONS", "Employee definition"], "level": 3}
```

### 3. Clustering Results (`data/reports/clusters/demo/family_candidates.csv`)
```csv
cluster_id,size,centroid_ea_id,confidence_score,ea_ids
abc123,3,EA_construction_001,0.92,"EA_construction_001,EA_construction_002,EA_construction_003"
def456,2,EA_healthcare_001,0.88,"EA_healthcare_001,EA_healthcare_002"
```

## 🎯 Common Use Cases

### 1. Process a Single PDF
```python
from ea_importer.pipeline import create_ingest_pipeline

pipeline = create_ingest_pipeline()
stats = pipeline.process_single_pdf(Path("my_agreement.pdf"))
print(f"Success: {stats.success}, Clauses: {stats.num_clauses}")
```

### 2. Find Similar Documents
```python
from ea_importer.utils.fingerprinter import create_text_fingerprinter

fingerprinter = create_text_fingerprinter()
fp1 = fingerprinter.fingerprint_document("doc1", text1)
fp2 = fingerprinter.fingerprint_document("doc2", text2)
similarity = fingerprinter.compute_jaccard_similarity(fp1.minhash_signature, fp2.minhash_signature)
print(f"Similarity: {similarity:.2%}")
```

### 3. Extract Specific Clause Types
```python
import json
from pathlib import Path

# Load clauses from JSONL file
clauses = []
with open("data/eas/clauses/EA_example.jsonl") as f:
    for line in f:
        clauses.append(json.loads(line))

# Find rate-related clauses
rate_clauses = [c for c in clauses if c.get('clause_type') == 'rate']
definition_clauses = [c for c in clauses if 'definition' in c.get('heading', '').lower()]
```

## 🔧 Configuration Tips

### Environment Setup (`.env`)
```bash
# For development (SQLite)
DATABASE_URL=sqlite:///ea_importer.db

# For production (PostgreSQL)
DATABASE_URL=postgresql://user:pass@localhost:5432/ea_importer

# OCR settings
OCR_LANGUAGE=eng
OCR_DPI=300

# Processing limits
MIN_CLAUSE_COUNT=20
```

### Directory Structure
```
data/
├── eas/raw/           # ← Place your PDF files here
├── eas/text/          # ← Cleaned text output
├── eas/clauses/       # ← Structured clause data
├── reports/           # ← Analysis reports
└── examples/          # ← Sample CSV files
```

## 🐛 Troubleshooting

### Issue: "No module named 'pydantic_settings'"
```bash
pip install pydantic-settings
```

### Issue: OCR not working
```bash
# Install Tesseract
# macOS: brew install tesseract
# Ubuntu: sudo apt-get install tesseract-ocr
pip install pytesseract opencv-python
```

### Issue: Database connection failed
```bash
# Use SQLite for development
echo "DATABASE_URL=sqlite:///ea_importer.db" > .env

# Or install PostgreSQL adapter
pip install psycopg2-binary
```

### Issue: MinHash clustering fails
```bash
pip install datasketch
# Or use basic threshold clustering (built-in)
```

## 📚 Next Steps

1. **Add more PDFs**: Copy additional EA files to `data/eas/raw/`
2. **Try clustering**: Run similarity analysis on multiple documents
3. **Explore outputs**: Examine the generated JSONL clause files
4. **Set up database**: For large-scale processing, configure PostgreSQL
5. **Customize processing**: Modify text cleaning and segmentation rules
6. **Build families**: Group similar EAs and extract common patterns

## 📞 Getting Help

- **Check logs**: Look for error messages in console output
- **Run tests**: `python3 scripts/basic_test.py`
- **Read README**: Full documentation in `README.md`
- **Example scripts**: See `scripts/` directory for more examples

---

🎉 **You're ready to start processing Enterprise Agreements!**