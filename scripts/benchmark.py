import time
import numpy as np
import requests

# Warm-up
def warmup(n=10):
    for _ in range(n):
        requests.post('http://localhost:8000/infer', files={'files': open('test.jpg','rb')})

# Benchmark
def benchmark(iters=100):
    latencies = []
    for _ in range(iters):
        start = time.time()
        requests.post('http://localhost:8000/infer', files={'files': open('test.jpg','rb')})
        latencies.append((time.time() - start)*1000)
    print(f"Avg latency: {np.mean(latencies):.2f} ms | P99: {np.percentile(latencies,99):.2f} ms")

if __name__ == '__main__':
    warmup()
    benchmark()