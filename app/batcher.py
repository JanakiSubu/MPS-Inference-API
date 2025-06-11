import asyncio
import torch
from collections import deque
from prometheus_client import Gauge

QUEUE = deque()
QUEUE_GAUGE = Gauge('inference_queue_size', 'Size of inference queue')

async def batch_worker(model, infer_fn, batch_size=8, wait_ms=50):
    while True:
        await asyncio.sleep(wait_ms/1000)
        if not QUEUE:
            continue
        batch = []
        futures = []
        while QUEUE and len(batch) < batch_size:
            req, fut = QUEUE.popleft()
            batch.append(req)
            futures.append(fut)
        # update gauge
        QUEUE_GAUGE.set(len(QUEUE))
        inputs = torch.cat(batch, dim=0)
        preds = infer_fn(model, inputs)
        for fut, p in zip(futures, preds):
            fut.set_result(p.item())

async def enqueue(request_tensor):
    loop = asyncio.get_event_loop()
    fut = loop.create_future()
    QUEUE.append((request_tensor, fut))
    QUEUE_GAUGE.set(len(QUEUE))
    return await fut