#!/usr/bin/env python3
"""
EA Importer - Australian Enterprise Agreement Ingestion & Corpus Builder
"""

from setuptools import setup, find_packages

setup(
    name="ea-importer",
    version="0.1.0",
    description="Australian Enterprise Agreement Ingestion & Corpus Builder",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="EA Importer Team",
    python_requires=">=3.9",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        # PDF Processing
        "PyPDF2>=3.0.0",
        "pdfplumber>=0.9.0",
        "pdf2image>=1.16.0",
        
        # OCR
        "pytesseract>=0.3.10",
        "opencv-python>=4.8.0",
        "Pillow>=10.0.0",
        
        # Text Processing
        "spacy>=3.6.0",
        "nltk>=3.8.0",
        "regex>=2023.6.3",
        "beautifulsoup4>=4.12.0",
        "lxml>=4.9.0",
        
        # Machine Learning & Clustering
        "scikit-learn>=1.3.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.11.0",
        "hdbscan>=0.8.29",
        "datasketch>=1.6.0",
        
        # Embeddings (optional)
        "sentence-transformers>=2.2.0",
        "torch>=2.0.0",
        
        # Database
        "psycopg2-binary>=2.9.0",
        "sqlalchemy>=2.0.0",
        "alembic>=1.11.0",
        
        # Web Framework
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "jinja2>=3.1.0",
        "python-multipart>=0.0.6",
        
        # CLI
        "click>=8.1.0",
        "rich>=13.4.0",
        "typer>=0.9.0",
        
        # Utilities
        "pydantic>=2.0.0",
        "pydantic-settings>=2.0.0",
        "python-dotenv>=1.0.0",
        "requests>=2.31.0",
        "tqdm>=4.65.0",
        "jsonschema>=4.18.0",
        "pyyaml>=6.0.0",
        "httpx>=0.24.0",
        
        # Testing
        "pytest>=7.4.0",
        "pytest-asyncio>=0.21.0",
        "pytest-cov>=4.1.0",
    ],
    extras_require={
        "dev": [
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
            "pre-commit>=3.3.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "ea-agent=ea_importer.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Legal Industry",
        "Topic :: Text Processing :: Legal Documents",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)