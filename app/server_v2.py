"""Web server: static pages and the scientific API (no neural network, no torch).

Run:  python -m uvicorn app.server_v2:app --port 8000
"""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from app.api_v2 import router  # noqa: E402

STATIC = REPO / "app" / "static"
app = FastAPI(title="Crater2Impactor",
              description="Impactor Parameter Estimation from Lunar Impact Craters. Explicit-prior Bayesian inference of impactor properties from simple lunar craters.")
app.include_router(router)
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


def _page(name: str):
    def handler() -> FileResponse:
        return FileResponse(STATIC / name)
    return handler


for route, page in (("/", "index.html"), ("/explorer", "forward.html"), ("/forward", "forward.html"),
                    ("/model", "model.html"), ("/about", "about.html"), ("/feedback", "feedback.html")):
    app.add_api_route(route, _page(page), methods=["GET"], include_in_schema=False)


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> FileResponse:
    return FileResponse(STATIC / "favicon.ico")
