import os
import time
import asyncio
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from app.model import load_model, infer, DEVICE
from app.utils import preprocess
from app.batcher import enqueue, batch_worker

# Initialize model and app
model = load_model()
app = FastAPI()

@app.on_event("startup")
async def on_startup():
    # Launch the batching worker
    asyncio.create_task(batch_worker(model, infer))

@app.post("/infer")
async def infer_single(file: UploadFile = File(...)):
    try:
        start_time = time.time()
        tensor = await preprocess(file, DEVICE)
        pred = infer(model, tensor)[0].item()
        latency = (time.time() - start_time) * 1000
        return {"predictions": [pred], "latency_ms": round(latency, 2)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/infer-quant")
async def infer_quant(file: UploadFile = File(...)):
    try:
        # Enable quantization and reload model
        os.environ["QUANTIZE"] = "1"
        quant_model = load_model()
        start_time = time.time()
        tensor = await preprocess(file, DEVICE)
        pred = infer(quant_model, tensor)[0].item()
        latency = (time.time() - start_time) * 1000
        return {"predictions": [pred], "latency_ms": round(latency, 2)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/infer-batch")
async def infer_batch(files: list[UploadFile] = File(...)):
    try:
        start_time = time.time()
        results = []
        for f in files:
            tensor = await preprocess(f, DEVICE)
            p = await enqueue(tensor)
            results.append(p)
        latency = (time.time() - start_time) * 1000
        return {"predictions": results, "latency_ms": round(latency, 2)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
