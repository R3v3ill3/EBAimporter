#!/bin/bash
"""
Installation script for EA Importer system.
"""

set -e  # Exit on any error

echo "🚀 EA Importer Installation Script"
echo "=================================="

# Check Python version
echo "Checking Python version..."
python3 --version || {
    echo "❌ Python 3 is required but not found"
    echo "Please install Python 3.9+ and try again"
    exit 1
}

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️  It's recommended to use a virtual environment"
    echo "Do you want to create one? (y/n)"
    read -r create_venv
    
    if [[ $create_venv == "y" || $create_venv == "Y" ]]; then
        echo "Creating virtual environment..."
        python3 -m venv venv
        source venv/bin/activate
        echo "✅ Virtual environment created and activated"
    fi
fi

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install --upgrade pip

# Install core dependencies without the ones that might fail
pip install \
    PyPDF2 \
    pdfplumber \
    Pillow \
    scikit-learn \
    numpy \
    pandas \
    scipy \
    sqlalchemy \
    click \
    rich \
    typer \
    pydantic \
    python-dotenv \
    requests \
    tqdm \
    jsonschema \
    pyyaml

echo "✅ Core dependencies installed"

# Try to install optional dependencies
echo "📦 Installing optional dependencies..."

# Try OCR dependencies
pip install pytesseract opencv-python || {
    echo "⚠️  OCR dependencies failed to install"
    echo "You may need to install Tesseract manually"
}

# Try ML dependencies
pip install datasketch || {
    echo "⚠️  MinHash library failed to install"
}

pip install hdbscan || {
    echo "⚠️  HDBSCAN failed to install"
}

# Try to install Pydantic settings
pip install pydantic-settings || {
    echo "⚠️  pydantic-settings failed to install"
}

# Try to install PostgreSQL adapter
pip install psycopg2-binary || {
    echo "⚠️  PostgreSQL adapter failed to install"
    echo "You can use SQLite for development"
}

# Check for Tesseract OCR
echo "🔍 Checking for Tesseract OCR..."
tesseract --version >/dev/null 2>&1 && {
    echo "✅ Tesseract OCR found"
} || {
    echo "⚠️  Tesseract OCR not found"
    echo "Install it with:"
    echo "  macOS: brew install tesseract"
    echo "  Ubuntu: sudo apt-get install tesseract-ocr"
    echo "  Windows: Download from GitHub releases"
}

# Create data directories
echo "📁 Creating data directories..."
mkdir -p data/eas/{raw,text,clauses,fp,emb}
mkdir -p data/{instances,families,reports/{clusters,qa},versions}
echo "✅ Data directories created"

# Copy example environment file
if [[ ! -f .env ]]; then
    if [[ -f .env.example ]]; then
        cp .env.example .env
        echo "✅ Environment file created from example"
        echo "📝 Please edit .env to configure your settings"
    fi
fi

# Try to install the package in development mode
echo "📦 Installing EA Importer package..."
pip install -e . || {
    echo "⚠️  Package installation failed"
    echo "You can still run scripts directly"
}

# Run basic tests
echo "🧪 Running basic tests..."
python3 scripts/basic_test.py || {
    echo "⚠️  Basic tests failed"
    echo "Check the error messages above"
}

echo ""
echo "🎉 Installation completed!"
echo ""
echo "📚 Next steps:"
echo "1. Configure your settings in .env"
echo "2. Set up a database (PostgreSQL recommended, SQLite for development)"
echo "3. Initialize the database: python -m ea_importer.cli db init"
echo "4. Add PDF files to data/eas/raw/"
echo "5. Run the demo: python scripts/demo.py"
echo ""
echo "📖 For more information, see README.md"