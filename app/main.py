"""
Kawsay — FastAPI backend.
Serves the frontend, exposes the search/ask API and evaluation metrics.
"""
import json
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator

from app.generation import LocalGenerator, REFUSAL, SYSTEM_PROMPT
from app.retrieval import SearchEngine
from app.settings import ARTIFACT_DIR, FRONTEND_DIR, MAX_QUERY_LENGTH


# ── Request schema ────────────────────────────────────────────────────────────

class QueryInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: StrictStr = Field(min_length=10, max_length=MAX_QUERY_LENGTH)

    @field_validator("query")
    @classmethod
    def clean_query(cls, value: str) -> str:
        value = " ".join(value.split())
        if len(value) < 10:
            raise ValueError("La consulta debe tener al menos 10 caracteres.")
        if sum(c.isalpha() for c in value) < 5:
            raise ValueError("La consulta debe contener al menos 5 letras.")
        return value


# ── Lifespan: load models once at startup ────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.engine       = None
    app.state.generator    = None
    app.state.report       = {}
    app.state.request_lock = threading.Lock()

    index_path   = ARTIFACT_DIR / "index.joblib"
    metrics_path = ARTIFACT_DIR / "metrics.json"

    if index_path.exists() and metrics_path.exists():
        try:
            app.state.engine = SearchEngine()
            app.state.report = json.loads(metrics_path.read_text(encoding="utf-8"))

            if app.state.report.get("version") != app.state.engine.index["version"]:
                raise RuntimeError(
                    "Index and metrics.json versions differ. "
                    "Run: python -m app.build_index"
                )
        except Exception as e:
            print(f"[kawsay] Warning: could not load index — {e}")
            app.state.engine = None

        # Generator is optional: system works without Ollama (search only)
        try:
            app.state.generator = LocalGenerator()
            print("[kawsay] Generator ready.")
        except Exception as e:
            print(f"[kawsay] Generator not available (search-only mode): {e}")
    else:
        print("[kawsay] Index not found. Run: python -m app.build_index")

    yield


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Kawsay — Recuperación semántica lingüística",
    version="1.0.0",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


# ── Helpers ───────────────────────────────────────────────────────────────────

def require_engine() -> SearchEngine:
    if app.state.engine is None:
        raise HTTPException(
            503,
            detail=(
                "Índice no disponible. "
                "Ejecuta: python -m app.build_index"
            ),
        )
    return app.state.engine


def run_search(query: str) -> dict:
    try:
        return require_engine().search(query)
    except ValueError as e:
        raise HTTPException(422, detail=str(e)) from e


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {
        "status":        "ok" if app.state.engine else "not_ready",
        "generator":     app.state.generator is not None,
        "search_only":   app.state.generator is None,
    }


@app.get("/api/documents")
def documents() -> list:
    return require_engine().chunks


@app.get("/api/metrics")
def metrics() -> dict:
    require_engine()
    return app.state.report


@app.post("/api/search")
def search_only(payload: QueryInput) -> dict:
    """Return ranked results from both semantic and BM25 search."""
    return run_search(payload.query)


@app.post("/api/ask")
def ask(payload: QueryInput) -> dict:
    """Search, apply gate, and generate a grounded answer if threshold is met."""
    if not app.state.request_lock.acquire(blocking=False):
        raise HTTPException(
            429,
            detail="Hay una consulta en proceso. Intenta en unos segundos.",
        )
    try:
        start   = time.perf_counter()
        results = run_search(payload.query)

        threshold = app.state.report.get(
            "threshold_selection", {}
        ).get("threshold", 0.30)

        top_score = results["semantic"][0]["score"]
        accepted  = top_score >= threshold

        if not accepted:
            answer = {
                "status":             "below_threshold",
                "answer":             REFUSAL,
                "evidence":           [],
                "generation_seconds": 0.0,
            }
        elif app.state.generator is None:
            # Search-only mode: no generator available
            answer = {
                "status":             "search_only",
                "answer":             (
                    "El generador no está disponible. "
                    "Revisa los fragmentos recuperados como fuentes."
                ),
                "evidence":           [],
                "generation_seconds": 0.0,
            }
        else:
            answer = app.state.generator.generate(
                payload.query, results["semantic"]
            )
            answer.pop("raw_output", None)  # never expose unverified draft

        return {
            **answer,
            "results":            results,
            "threshold":          threshold,
            "gate_passed":        accepted,
            "top_score":          round(top_score, 6),
            "version":            app.state.engine.index["version"],
            "total_seconds":      round(time.perf_counter() - start, 3),
        }
    finally:
        app.state.request_lock.release()
