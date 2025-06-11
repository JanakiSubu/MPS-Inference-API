#!/usr/bin/env bash
# Launch FastAPI server using Uvicorn
uvicorn app.main:app --host 0.0.0.0 --port 8000