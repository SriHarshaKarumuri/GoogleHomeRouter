"""Vercel serverless entry point for FastAPI."""
import os
os.environ.setdefault("PYTHONPATH", "..")

from mangum import Mangum
from main import app  # noqa: E402

# Wrap ASGI app for Vercel's HTTP event handler
handler = Mangum(app)
