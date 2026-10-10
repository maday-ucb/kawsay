"""
Start the Kawsay server.

Usage:
    python run.py
    python run.py --port 8997
    python run.py --host 0.0.0.0   # expose on local network
    python run.py --rebuild        # rebuild index before starting
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT       = Path(__file__).resolve().parent
INDEX_PATH = ROOT / "artifacts" / "index.joblib"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Kawsay server")
    parser.add_argument("--port",    type=int, default=8995)
    parser.add_argument("--host",    default="127.0.0.1")
    parser.add_argument("--rebuild", action="store_true",
                        help="Rebuild the index before starting")
    args = parser.parse_args()

    if args.rebuild:
        print("Rebuilding index...")
        result = subprocess.run(
            [sys.executable, "-m", "app.build_index"], cwd=ROOT
        )
        if result.returncode != 0:
            print("Index build failed. Fix the errors above and try again.")
            sys.exit(1)
    elif not INDEX_PATH.exists():
        print("Index not found. Build it first:")
        print("  python -m app.download_model")
        print("  python -m app.build_index")
        print("Or run with --rebuild to do it automatically:")
        print("  python run.py --rebuild")
        sys.exit(1)

    print(f"\nStarting Kawsay on http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.\n")

    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
