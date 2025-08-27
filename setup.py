#!/usr/bin/env python3
"""
Setup script for EBAimporter - Australian Enterprise Agreement Ingestion & Corpus Building System
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
requirements = []
with open('requirements.txt', 'r') as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#'):
            # Remove version pinning for setup.py
            if '>=' in line:
                line = line.split('>=')[0]
            elif '==' in line:
                line = line.split('==')[0]
            requirements.append(line)

setup(
    name="ea-importer",
    version="1.0.0",
    description="Australian Enterprise Agreement Ingestion & Corpus Building System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="EA Importer Team",
    author_email="contact@ea-importer.com",
    url="https://github.com/your-org/EBAimporter",
    license="MIT",
    
    # Package discovery
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    
    # Include non-Python files
    include_package_data=True,
    package_data={
        "ea_importer": [
            "web/static/*",
            "web/templates/*",
            "config/*.yaml",
            "config/*.json",
        ]
    },
    
    # Dependencies
    install_requires=requirements,
    
    # Optional dependencies
    extras_require={
        'dev': [
            'pytest>=7.4.0',
            'pytest-asyncio>=0.21.0',
            'pytest-cov>=4.1.0',
            'black>=23.7.0',
            'isort>=5.12.0',
            'flake8>=6.0.0',
            'mypy>=1.5.0',
        ],
        'analysis': [
            'jupyter>=1.0.0',
            'matplotlib>=3.7.0',
            'seaborn>=0.12.0',
        ],
        'gpu': [
            'torch>=2.0.0',
            'transformers>=4.20.0',
        ]
    },
    
    # Entry points for CLI commands
    entry_points={
        'console_scripts': [
            'ea-importer=ea_importer.cli:main',
            'ea-web=ea_importer.web.app:run_server',
        ],
    },
    
    # Classifiers
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Legal Industry",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Text Processing",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Office/Business :: Legal",
    ],
    
    # Python version requirement
    python_requires=">=3.8",
    
    # Keywords
    keywords=[
        "enterprise-agreements", 
        "legal-documents", 
        "pdf-processing", 
        "document-clustering", 
        "ocr", 
        "australian-law",
        "industrial-instruments"
    ],
    
    # Project URLs
    project_urls={
        "Bug Reports": "https://github.com/your-org/EBAimporter/issues",
        "Source": "https://github.com/your-org/EBAimporter",
        "Documentation": "https://ea-importer.readthedocs.io/",
    },
)