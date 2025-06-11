# MPS Inference API 🔥

A multi-mode image classification API running entirely on Apple Silicon (M1/2) via PyTorch’s MPS backend—no cloud GPUs needed!

## Features:

* FP32 inference (`/infer`)
* Dynamic quantized inference (`/infer-quant`)
* Intelligent batching (`/infer-batch`)
* Steady-state latency reporting in JSON
* Extensible: swap ResNet-18 & ViT-B/16 backbones via `MODEL_TYPE`

## 🚀 Quickstart

```bash
# 1. Clone repository
git clone https://github.com/JanakiSubu/MPS-Inference-API.git
cd MPS-Inference-API

# 2. Setup environment
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 3. Choose model type
export MODEL_TYPE=resnet   # or "vit"

# 4. Run server
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**Test endpoints:**

```bash
# Full-precision
curl -F "file=@test.jpg" http://localhost:8000/infer
# Quantized
curl -F "file=@test.jpg" http://localhost:8000/infer-quant
# Batched
curl -F "files=@test.jpg" http://localhost:8000/infer-batch
```

## 📈 Benchmarks (steady-state)

| Endpoint       | Latency (ms) |              Speedup vs FP32 |
| -------------- | -----------: | ---------------------------: |
| `/infer`       |     **17.0** |                           1× |
| `/infer-quant` |     **13.1** |                     **1.3×** |
| `/infer-batch` |     **44.8** | (4× batch → \~11.2 ms/image) |

## 🔧 Features

* Zero setup: runs out-of-the-box on M1/2 with PyTorch MPS
* Dynamic Quantization: speed vs. accuracy tradeoff
* Async Batching: groups requests via asyncio queue
* Lightweight: <300 lines of code total
* Plug-and-Play: swap backbones with `MODEL_TYPE`

## 📁 Repository Structure

```text
mps-inference-api/
├── app/
│   ├── main.py       # API endpoints & batching
│   ├── model.py      # load & quantize logic
│   ├── utils.py      # async MPS preprocessing
│   └── batcher.py    # queue + worker
├── scripts/
│   ├── benchmark.py  # warm-up & steady-state tests
│   └── map_labels.py # map ImageNet index → label
├── benchmark_chart.png  # bar chart of latencies
├── requirements.txt   # dependencies list
├── README.md          # this file
└── LICENSE            # MIT License
```

## 📂 How it works

1. Upload images via FastAPI `UploadFile` objects
2. Preprocess (resize, normalize) on the MPS device
3. Infer through ResNet-18 or ViT-B/16
4. Return JSON with `predictions` and `latency_ms` fields
5. Batch endpoint uses an asyncio queue + background worker to group requests

## 🛠️ Contributing

1. ⭐ Star the repo
2. 🔀 Fork & create a Pull Request
3. 🐛 Report issues
4. 🚀 Submit enhancements (e.g., explainability, new backbones)

