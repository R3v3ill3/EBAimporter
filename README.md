# EA Importer

**Australian Enterprise Agreement Ingestion & Corpus Builder**

A comprehensive system for ingesting, processing, clustering, and querying Australian Enterprise Agreements (EAs) with human-in-the-loop validation and quality assurance.

## 🎯 Overview

EA Importer is designed to transform a collection of 150+ Enterprise Agreement PDFs into a versioned, auditable, query-ready corpus. The system provides:

- **Batch PDF ingestion** with OCR support
- **Automatic text cleaning** and clause segmentation  
- **Document fingerprinting** and similarity analysis
- **Intelligent clustering** into EA families
- **Human-in-the-loop validation** workflows
- **Structured data extraction** (rates, rules, allowances)
- **Quality assurance** with synthetic testing
- **Version control** and audit trails

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   PDF Files     │───▶│   Ingest         │───▶│  Text & Clauses │
│   (Raw EAs)     │    │   Pipeline       │    │   (Segmented)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Clustering    │◀───│   Fingerprints   │───▶│    Database     │
│   (Families)    │    │   (Similarity)   │    │   (Metadata)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
        │
        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Gold Text     │───▶│   Rates & Rules  │───▶│   QA Testing    │
│  (Family Base)  │    │   (Structured)   │    │  (Validation)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- PostgreSQL 12+ (or SQLite for development)
- Tesseract OCR
- Git

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd EBAimporter
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Tesseract OCR:**
   
   **macOS:**
   ```bash
   brew install tesseract
   ```
   
   **Ubuntu/Debian:**
   ```bash
   sudo apt-get install tesseract-ocr
   ```
   
   **Windows:**
   Download from: https://github.com/UB-Mannheim/tesseract/wiki

4. **Set up environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your database settings
   ```

5. **Initialize database:**
   ```bash
   python -m ea_importer.cli db init
   ```

### Quick Demo

```bash
# Place some EA PDFs in data/eas/raw/
mkdir -p data/eas/raw
# Copy your PDF files here

# Run the demo script
python scripts/demo.py
```

## 📖 Usage

### Command Line Interface

The system provides a comprehensive CLI through the `ea-agent` command:

```bash
# Install the package to enable the ea-agent command
pip install -e .

# Or run directly
python -m ea_importer.cli --help
```

### Core Commands

#### 1. **Ingestion Pipeline**

Process PDF files into structured text and clauses:

```bash
# Process all PDFs in a directory
ea-agent ingest run data/eas/raw/

# With options
ea-agent ingest run data/eas/raw/ \
  --force-ocr \
  --max-files 10 \
  --pattern "*.pdf"

# Check ingestion status
ea-agent ingest status
```

#### 2. **Clustering Analysis**

Group similar EAs into families:

```bash
# Run adaptive clustering
ea-agent cluster run

# With specific algorithm and threshold
ea-agent cluster run \
  --algorithm minhash \
  --threshold 0.9 \
  --out reports/clusters/custom
```

#### 3. **Database Management**

```bash
# Initialize database
ea-agent db init

# Check database status
ea-agent db status

# Reset database (careful!)
ea-agent db init --drop
```

### Python API

```python
from ea_importer.pipeline import create_ingest_pipeline
from ea_importer.pipeline.clustering import create_ea_clusterer

# Create pipeline
pipeline = create_ingest_pipeline()

# Process documents
stats = pipeline.batch_ingest(
    input_dir=Path("data/eas/raw"),
    force_ocr=False
)

# Run clustering
clusterer = create_ea_clusterer()
result = clusterer.adaptive_clustering(fingerprints)
```

## 📁 Data Structure

The system organizes data in a structured hierarchy:

```
data/
├── eas/
│   ├── raw/           # Original PDF files
│   ├── text/          # Cleaned text files (.txt)
│   ├── clauses/       # Segmented clauses (.jsonl)
│   ├── fp/            # Fingerprints (.sha256, .minhash)
│   └── emb/           # Embeddings (.npy) [optional]
├── families/          # EA family definitions
│   └── {family_id}/
│       ├── gold/      # Gold standard text
│       ├── rates.csv  # Extracted rates
│       └── rules.json # Business rules
├── instances/         # Specific EA instances
│   └── {instance_id}/
│       └── overlay.json
├── reports/           # Analysis reports
│   ├── clusters/      # Clustering results
│   └── qa/           # Quality assurance
└── versions/          # Version control
    └── {version}/
        └── manifest.json
```

## 🔧 Configuration

Key configuration options in `.env`:

```bash
# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/ea_importer

# File Processing
OCR_LANGUAGE=eng
OCR_DPI=300
MIN_CLAUSE_COUNT=60

# Clustering
CLUSTER_THRESHOLD_HIGH=0.95
CLUSTER_THRESHOLD_MEDIUM=0.90
CLUSTER_THRESHOLD_LOW=0.85

# Jurisdiction
JURISDICTION="NSW + Federal (FWC/Fair Work)"
TARGET_VERSION="2025.08.v1"
```

## 🔍 Features

### Document Processing
- **Smart OCR**: Automatic text layer detection with OCR fallback
- **Text cleaning**: Header/footer removal, hyphenation repair, Unicode normalization
- **Clause segmentation**: Hierarchical numbering pattern recognition
- **Fingerprinting**: SHA256 + MinHash for similarity detection

### Clustering & Families
- **Multi-algorithm clustering**: MinHash, HDBSCAN, DBSCAN, Agglomerative
- **Adaptive thresholds**: Automatic parameter tuning
- **Family detection**: Group similar EAs with confidence scoring
- **Outlier detection**: Identify unique or problematic documents

### Quality Assurance
- **Validation metrics**: Processing success rates, clause counts, text quality
- **Similarity analysis**: Document and clause-level comparisons
- **Error reporting**: Detailed failure analysis and recommendations

### Human-in-the-Loop
- **Review workflows**: Cluster validation, family confirmation
- **Diff visualization**: Side-by-side comparison of similar documents
- **Manual overrides**: Expert curation of automatic decisions

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=ea_importer --cov-report=html

# Run specific test categories
pytest tests/test_pdf_processor.py
pytest tests/test_clustering.py
```

## 📊 Monitoring & Metrics

The system tracks comprehensive metrics:

- **Processing metrics**: Success rates, processing times, file sizes
- **Quality metrics**: Clause counts, text lengths, cleaning statistics  
- **Clustering metrics**: Silhouette scores, cluster sizes, confidence levels
- **Database metrics**: Record counts, query performance, storage usage

## 🔒 Security & Privacy

- **No PII in embeddings**: Personal information is never embedded
- **Audit trails**: Complete provenance tracking for all operations
- **Access controls**: Role-based permissions for sensitive operations
- **Data retention**: Configurable retention policies

## 🛠️ Development

### Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes
4. Add tests for new functionality
5. Run the test suite: `pytest`
6. Commit your changes: `git commit -m 'Add amazing feature'`
7. Push to the branch: `git push origin feature/amazing-feature`
8. Open a Pull Request

### Code Style

- Use `black` for code formatting
- Use `isort` for import sorting
- Use `flake8` for linting
- Use `mypy` for type checking

```bash
# Format code
black src/ tests/
isort src/ tests/

# Check linting
flake8 src/ tests/
mypy src/
```

## 📚 Documentation

- **API Documentation**: Generated with Sphinx (coming soon)
- **User Guide**: Detailed usage examples and workflows
- **Developer Guide**: Architecture and extension points
- **FAQ**: Common questions and troubleshooting

## 🗺️ Roadmap

### Phase 1: Core Pipeline ✅
- [x] PDF processing and OCR
- [x] Text cleaning and segmentation
- [x] Fingerprinting and clustering
- [x] Database schema and CLI

### Phase 2: Advanced Features (In Progress)
- [ ] Rates and rules extraction
- [ ] Instance management and overlays
- [ ] Web interface for human review
- [ ] Quality assurance framework

### Phase 3: Production Features (Planned)
- [ ] Batch CSV import from URLs
- [ ] Version control and corpus locking
- [ ] Advanced analytics and reporting
- [ ] REST API for integration

### Phase 4: Enterprise Features (Future)
- [ ] Multi-tenant support
- [ ] Advanced search and retrieval
- [ ] Machine learning improvements
- [ ] Compliance and audit features

## 💡 Support

- **Documentation**: Check the `docs/` directory
- **Issues**: Report bugs and feature requests on GitHub
- **Discussions**: Community discussions and Q&A
- **Email**: For security issues and private inquiries

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Fair Work Commission for EA data standards
- Australian Department of Employment and Workplace Relations
- Open source libraries that make this project possible
- The industrial relations community for feedback and testing

---

**EA Importer** - Transforming Enterprise Agreements into Actionable Intelligence