"""
Check that Ollama is running and the required model is available.

Usage:
    python -m scripts.check_ollama
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.settings import OLLAMA_MODEL, OLLAMA_BASE_URL


def main() -> None:
    print(f"Checking Ollama at {OLLAMA_BASE_URL}...")

    try:
        import ollama
    except ImportError:
        print("  ✗ ollama package not installed. Run: pip install ollama")
        sys.exit(1)

    try:
        client = ollama.Client(host=OLLAMA_BASE_URL)
        models = [m.model for m in client.list().models]
    except Exception as e:
        print(f"  ✗ Cannot connect to Ollama: {e}")
        print("  Make sure Ollama is running: https://ollama.com/download")
        sys.exit(1)

    print(f"  ✓ Ollama is running")
    print(f"  Available models: {models or '(none)'}")

    if any(OLLAMA_MODEL in m for m in models):
        print(f"  ✓ Model '{OLLAMA_MODEL}' is ready")
    else:
        print(f"  ✗ Model '{OLLAMA_MODEL}' not found.")
        print(f"  Run: ollama pull {OLLAMA_MODEL}")
        sys.exit(1)

    # Quick generation test
    print(f"\nRunning a quick generation test...")
    try:
        response = client.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": "Reply with just: OK"}],
            options={"temperature": 0, "num_predict": 10},
        )
        print(f"  ✓ Generation works. Response: {response.message.content.strip()}")
    except Exception as e:
        print(f"  ✗ Generation failed: {e}")
        sys.exit(1)

    print("\nOllama is ready for Kawsay.")


if __name__ == "__main__":
    main()
