"""
Scan all PDFs in data/raw_pdfs/ and produce an audit spreadsheet (CSV).
Use this BEFORE ingesting to decide which documents to include.

Usage:
    python -m scripts.audit_pdfs
    python -m scripts.audit_pdfs --output my_audit.csv
"""
import argparse
import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.settings import RAW_PDF_DIR
from scripts.ingest_pdfs import detect_type, infer_title


def count_words(text: str) -> int:
    return len(text.split())


def extract_preview(text: str, max_chars: int = 300) -> str:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    preview = ' '.join(lines[:8])
    return preview[:max_chars]


def audit_pdf(pdf_path: Path) -> dict:
    try:
        import pymupdf4llm
        text = pymupdf4llm.to_markdown(str(pdf_path), show_progress=False)
    except ImportError:
        try:
            import pymupdf
            doc = pymupdf.open(str(pdf_path))
            text = '\n'.join(page.get_text() for page in doc)
            doc.close()
        except Exception as e:
            return {
                "filename": pdf_path.name,
                "title": "",
                "type": detect_type(pdf_path.name),
                "words": 0,
                "pages": 0,
                "preview": "",
                "include": "",
                "notes": f"ERROR: {e}",
            }

    # Clean minimal
    text = re.sub(r'\n{3,}', '\n\n', text)
    words = count_words(text)

    try:
        import pymupdf
        doc = pymupdf.open(str(pdf_path))
        pages = doc.page_count
        doc.close()
    except Exception:
        pages = 0

    title   = infer_title(pdf_path, text)
    preview = extract_preview(text)

    # Auto-suggest: include if enough words and not a pure dictionary
    doc_type   = detect_type(pdf_path.name)
    auto_include = "si" if words >= 300 and doc_type != "diccionario" else "revisar"

    return {
        "filename": pdf_path.name,
        "title":    title,
        "type":     doc_type,
        "words":    words,
        "pages":    pages,
        "preview":  preview,
        "include":  auto_include,
        "notes":    "",
    }


def main():
    parser = argparse.ArgumentParser(description="Audit PDFs for Kawsay corpus")
    parser.add_argument('--input-dir', type=str, default=None)
    parser.add_argument('--output', type=str, default="audit.csv")
    args = parser.parse_args()

    input_dir = Path(args.input_dir) if args.input_dir else RAW_PDF_DIR

    if not input_dir.exists():
        print(f"Directory not found: {input_dir}")
        sys.exit(1)

    pdfs = sorted(input_dir.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {input_dir}")
        sys.exit(0)

    print(f"Auditing {len(pdfs)} PDFs from {input_dir}...")

    rows = []
    for i, pdf in enumerate(pdfs, 1):
        print(f"  [{i:3d}/{len(pdfs)}] {pdf.name[:60]}")
        rows.append(audit_pdf(pdf))

    out_path = Path(args.output)
    fieldnames = ["filename", "title", "type", "words", "pages",
                  "include", "notes", "preview"]

    with out_path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Stats
    total_words = sum(r['words'] for r in rows)
    by_type = {}
    for r in rows:
        by_type[r['type']] = by_type.get(r['type'], 0) + 1

    print(f"\n── Audit complete ───────────────────────")
    print(f"  Total PDFs   : {len(rows)}")
    print(f"  Total words  : {total_words:,}")
    print(f"  By type:")
    for t, count in sorted(by_type.items()):
        print(f"    {t:15s}: {count}")
    print(f"\n  Spreadsheet saved to: {out_path.resolve()}")
    print(f"\n  Next steps:")
    print(f"  1. Open {out_path.name} in Excel or LibreOffice")
    print(f"  2. Review 'include' column (si/revisar) and adjust")
    print(f"  3. Add notes for partial documents (e.g. 'solo cap 2-4')")
    print(f"  4. Run: python -m scripts.ingest_pdfs")


if __name__ == "__main__":
    main()
