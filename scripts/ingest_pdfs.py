"""
Convert PDF files from data/raw_pdfs/ into chunked Markdown documents
ready for indexing in data/documents/.

Usage:
    python -m scripts.ingest_pdfs              # process all PDFs
    python -m scripts.ingest_pdfs --dry-run    # preview without writing
    python -m scripts.ingest_pdfs --file my.pdf
"""
import argparse
import re
import sys
from pathlib import Path

# Allow running as script or module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.settings import DOCUMENT_DIR, RAW_PDF_DIR, CHUNK_MAX_WORDS, CHUNK_OVERLAP

# Document type detection by filename keywords
TYPE_KEYWORDS = {
    "tesis":    ["tesis", "thesis", "disertacion", "dissertation"],
    "articulo": ["articulo", "article", "paper", "revista", "journal"],
    "proyecto": ["proyecto", "project", "informe", "report", "programa"],
    "libro":    ["libro", "book", "capitulo", "chapter", "manual"],
    "material": ["material", "didactico", "cuento", "story", "guia", "guide"],
    "diccionario": ["diccionario", "dictionary", "glosario", "glossary", "vocabulario"],
}


def detect_type(filename: str) -> str:
    name = filename.lower()
    for doc_type, keywords in TYPE_KEYWORDS.items():
        if any(kw in name for kw in keywords):
            return doc_type
    return "otro"


def extract_text(pdf_path: Path) -> str:
    """Extract text from PDF using pymupdf4llm (preserves structure)."""
    try:
        import pymupdf4llm
        return pymupdf4llm.to_markdown(str(pdf_path), show_progress=False)
    except ImportError:
        # Fallback to basic pymupdf if pymupdf4llm not available
        import pymupdf
        doc = pymupdf.open(str(pdf_path))
        pages = [page.get_text() for page in doc]
        doc.close()
        return "\n\n".join(pages)


def clean_text(text: str) -> str:
    """Remove common PDF artifacts."""
    # Remove page numbers (isolated digits on a line)
    text = re.sub(r'(?m)^\s*\d{1,4}\s*$', '', text)
    # Collapse 3+ blank lines into 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Remove horizontal rules
    text = re.sub(r'[-_=]{4,}', '', text)
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    # Normalize whitespace within lines
    lines = [' '.join(line.split()) for line in text.splitlines()]
    return '\n'.join(lines).strip()


def split_into_chunks(text: str, max_words: int = CHUNK_MAX_WORDS,
                      overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping word-based chunks.
    Each chunk fits within multilingual-e5-base's 512-token limit (~400 words).
    """
    # Split on double newlines first to respect paragraph boundaries
    paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]

    chunks = []
    buffer_words: list[str] = []

    for para in paragraphs:
        words = para.split()
        if not words:
            continue

        # If adding this paragraph exceeds limit, flush buffer first
        if buffer_words and len(buffer_words) + len(words) > max_words:
            chunk = ' '.join(buffer_words)
            if len(chunk.split()) >= 20:   # discard tiny chunks
                chunks.append(chunk)
            # Keep overlap words for context continuity
            buffer_words = buffer_words[-overlap:] if overlap else []

        buffer_words.extend(words)

        # If buffer alone exceeds limit, split it
        while len(buffer_words) > max_words:
            chunk = ' '.join(buffer_words[:max_words])
            chunks.append(chunk)
            buffer_words = buffer_words[max_words - overlap:]

    # Flush remaining
    if buffer_words and len(buffer_words) >= 20:
        chunks.append(' '.join(buffer_words))

    return chunks


def clean_title(raw: str) -> str:
    """Strip HTML tags, entities and markdown formatting from a title candidate."""
    raw = re.sub(r'<[^>]+>', '', raw)           # remove HTML tags
    raw = re.sub(r'<!--.*?-->', '', raw)         # remove HTML comments
    raw = raw.replace('&gt;', '>').replace('&lt;', '<').replace('&amp;', '&')
    raw = raw.replace('&quot;', '"').replace('&#39;', "'").replace('&nbsp;', ' ')
    raw = re.sub(r'&#\d+;', '', raw)            # remove remaining numeric entities
    raw = re.sub(r'&[a-z]+;', '', raw)          # remove remaining named entities
    raw = re.sub(r'[*_`~]+', '', raw)           # remove markdown bold/italic
    return ' '.join(raw.split()).strip()


def infer_title(pdf_path: Path, text: str) -> str:
    """
    Try to extract a meaningful title from the first lines of the document.
    Falls back to the filename stem.
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for line in lines[:15]:
        candidate = clean_title(line)
        if 10 < len(candidate) < 150 and not candidate.startswith('#'):
            if not re.match(r'^(página|page|\d|isbn|doi|http|©|autor|author|start of)', candidate.lower()):
                return candidate
    return pdf_path.stem.replace('_', ' ').replace('-', ' ').title()


def build_markdown(title: str, doc_type: str, source_file: str,
                   chunks: list[str]) -> str:
    """Build a Markdown document with the required # / ## structure."""
    lines = [
        f"# {title}",
        f"",
        f"<!-- type:{doc_type} source:{source_file} -->",
        "",
    ]
    for i, chunk in enumerate(chunks, 1):
        lines += [f"## Fragmento {i}", "", chunk, ""]
    return '\n'.join(lines)


def next_doc_id(output_dir: Path) -> str:
    """Return the next available DXXX prefix (supports up to D999)."""
    existing = sorted(output_dir.glob("D[0-9]*_*.md"))
    if not existing:
        return "D001"
    last = existing[-1].stem.split('_')[0]
    num = int(last[1:]) + 1
    return f"D{num:03d}"


def safe_stem(title: str) -> str:
    """Convert a title to a safe filename stem."""
    stem = title.lower()
    stem = re.sub(r'[áàä]', 'a', stem)
    stem = re.sub(r'[éèë]', 'e', stem)
    stem = re.sub(r'[íìï]', 'i', stem)
    stem = re.sub(r'[óòö]', 'o', stem)
    stem = re.sub(r'[úùü]', 'u', stem)
    stem = re.sub(r'[ñ]', 'n', stem)
    stem = re.sub(r'[^a-z0-9]+', '_', stem)
    return stem[:40].strip('_')


def process_pdf(pdf_path: Path, output_dir: Path,
                dry_run: bool = False) -> dict:
    """Process a single PDF and write its Markdown file."""
    print(f"  Processing: {pdf_path.name}")

    try:
        raw = extract_text(pdf_path)
    except Exception as e:
        return {"file": pdf_path.name, "status": "error", "reason": str(e)}

    text = clean_text(raw)
    if len(text.split()) < 50:
        return {"file": pdf_path.name, "status": "skipped",
                "reason": "too little text after cleaning"}

    title     = infer_title(pdf_path, text)
    doc_type  = detect_type(pdf_path.name)
    chunks    = split_into_chunks(text)

    if not chunks:
        return {"file": pdf_path.name, "status": "skipped",
                "reason": "no chunks generated"}

    doc_id   = next_doc_id(output_dir)
    stem     = safe_stem(title)
    out_name = f"{doc_id}_{stem}.md"
    out_path = output_dir / out_name
    content  = build_markdown(title, doc_type, pdf_path.name, chunks)

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path.write_text(content, encoding='utf-8')

    return {
        "file":    pdf_path.name,
        "status":  "ok",
        "output":  out_name,
        "title":   title,
        "type":    doc_type,
        "chunks":  len(chunks),
        "words":   len(text.split()),
    }


def main():
    parser = argparse.ArgumentParser(description="Ingest PDFs into Kawsay corpus")
    parser.add_argument('--dry-run', action='store_true',
                        help='Preview without writing files')
    parser.add_argument('--file', type=str, default=None,
                        help='Process a single PDF by filename')
    parser.add_argument('--input-dir', type=str, default=None,
                        help='Override input directory (default: data/raw_pdfs)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Override output directory (default: data/documents)')
    args = parser.parse_args()

    input_dir  = Path(args.input_dir)  if args.input_dir  else RAW_PDF_DIR
    output_dir = Path(args.output_dir) if args.output_dir else DOCUMENT_DIR

    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}")
        sys.exit(1)

    if args.file:
        pdfs = [input_dir / args.file]
    else:
        pdfs = sorted(input_dir.glob("*.pdf"))

    if not pdfs:
        print(f"No PDF files found in {input_dir}")
        sys.exit(0)

    print(f"\nKawsay PDF Ingestion Pipeline")
    print(f"  Input:   {input_dir}  ({len(pdfs)} file(s))")
    print(f"  Output:  {output_dir}")
    print(f"  Dry run: {args.dry_run}\n")

    results = []
    for pdf in pdfs:
        result = process_pdf(pdf, output_dir, dry_run=args.dry_run)
        results.append(result)

    # Summary
    ok      = [r for r in results if r['status'] == 'ok']
    skipped = [r for r in results if r['status'] == 'skipped']
    errors  = [r for r in results if r['status'] == 'error']

    print(f"\n── Summary ──────────────────────────────")
    print(f"  Processed : {len(ok)}")
    print(f"  Skipped   : {len(skipped)}")
    print(f"  Errors    : {len(errors)}")

    if ok:
        total_chunks = sum(r['chunks'] for r in ok)
        print(f"  Total chunks generated: {total_chunks}")
        print(f"\n  Files created:")
        for r in ok:
            print(f"    {r['output']:45s} {r['chunks']:3d} chunks  [{r['type']}]  {r['title'][:50]}")

    if skipped:
        print(f"\n  Skipped:")
        for r in skipped:
            print(f"    {r['file']:45s} {r['reason']}")

    if errors:
        print(f"\n  Errors:")
        for r in errors:
            print(f"    {r['file']:45s} {r['reason']}")

    if args.dry_run:
        print("\n  (Dry run — no files written)")


if __name__ == "__main__":
    main()
