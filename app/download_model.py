"""
Download the embedding model to models/embedding/ before indexing.
Only needs to run once; subsequent runs use the local cache.

Usage:
    python -m app.download_model
"""
from app.retrieval import load_encoder
from app.settings import MODEL_DIR, EMBEDDING_ID


def main() -> None:
    print(f"Downloading embedding model: {EMBEDDING_ID}")
    print(f"Destination: {MODEL_DIR / 'embedding'}")
    print("This may take a few minutes on first run...\n")
    encoder = load_encoder(local_only=False)
    print(f"\nModel ready. Max sequence length: {encoder.max_seq_length} tokens")
    print("You can now run: python -m app.build_index")


if __name__ == "__main__":
    main()
