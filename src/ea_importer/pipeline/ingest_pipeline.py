"""
Ingestion pipeline orchestrator for EA Importer.

Provides reusable functions to process PDFs from a directory or a single file
and persist artifacts (text, clauses, fingerprint) to the configured data dirs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Dict, Any
import json

from ..core.logging import get_logger, log_function_call
from ..core.config import get_settings
from ..utils import PDFProcessor, TextCleaner, TextSegmenter, Fingerprinter


logger = get_logger(__name__)


@log_function_call
def ingest_file(pdf_path: Path, output_root: Optional[Path] = None, force_ocr: bool = False) -> Dict[str, Any]:
    settings = get_settings()
    output_root = output_root or settings.paths.data_dir / "eas"

    processor = PDFProcessor()
    cleaner = TextCleaner()
    segmenter = TextSegmenter()
    fingerprinter = Fingerprinter()

    document = processor.process_pdf(pdf_path, force_ocr=force_ocr)
    cleaned = cleaner.clean_document(document)
    clauses = segmenter.segment_document(cleaned)
    fingerprint = fingerprinter.fingerprint_document(document)

    ea_id = document.metadata.get("ea_id")

    text_dir = output_root / "text"
    text_dir.mkdir(parents=True, exist_ok=True)
    with open(text_dir / f"{ea_id}.txt", "w", encoding="utf-8") as f:
        f.write(document.full_text)

    clauses_dir = output_root / "clauses"
    clauses_dir.mkdir(parents=True, exist_ok=True)
    with open(clauses_dir / f"{ea_id}.jsonl", "w", encoding="utf-8") as f:
        for c in clauses:
            f.write(
                json.dumps(
                    {
                        "ea_id": c.ea_id,
                        "clause_id": c.clause_id,
                        "heading": c.heading,
                        "text": c.text,
                        "path": c.path,
                        "hash_sha256": c.hash_sha256,
                        "token_count": c.token_count,
                    }
                )
                + "\n"
            )

    fp_dir = output_root / "fp"
    fingerprinter.save_fingerprint(fingerprint, fp_dir)

    return {
        "ea_id": ea_id,
        "pages": len(document.pages),
        "clauses": len(clauses),
        "ocr_used": bool(document.metadata.get("ocr_used")),
    }


@log_function_call
def ingest_directory(
    input_dir: Path,
    output_root: Optional[Path] = None,
    force_ocr: bool = False,
    max_files: Optional[int] = None,
) -> Dict[str, Any]:
    pdfs = sorted([p for p in Path(input_dir).glob("*.pdf") if p.is_file()])
    if max_files:
        pdfs = pdfs[:max_files]

    successes: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    for pdf in pdfs:
        try:
            info = ingest_file(pdf, output_root=output_root, force_ocr=force_ocr)
            successes.append({"file": str(pdf), **info})
            logger.info(f"Ingested {pdf.name} -> {info['ea_id']}")
        except Exception as e:
            logger.error(f"Ingestion failed for {pdf}: {e}")
            failures.append({"file": str(pdf), "error": str(e)})

    return {
        "processed": len(successes),
        "failed": len(failures),
        "successes": successes,
        "failures": failures,
    }


__all__ = ["ingest_file", "ingest_directory"]

