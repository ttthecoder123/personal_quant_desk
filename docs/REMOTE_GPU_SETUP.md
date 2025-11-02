# Remote GPU Training Setup Guide
## M1 MacBook + Remote GPU Server Architecture

This guide shows how to leverage remote GPU servers for training while using your M1 MacBook for development and inference.

---

## 🎯 Overview

**Cost Comparison:**
- M4 Max MacBook: **$3,500-4,000**
- M1 MacBook + Remote GPU: **$50-200/year** (for training)

**Performance:**
- M4 Max: Good for inference, limited for training
- Remote GPU (RTX 4090/A100): **10-50x faster** training

---

## 📋 Prerequisites

### On Your M1 MacBook:

```bash
# Python 3.10+
python3 --version

# Install dependencies
pip install onnxruntime numpy pandas pyyaml

# Optional: For monitoring
pip install wandb mlflow
```

### Remote GPU Server Options:

#### Option 1: Vast.ai (Recommended for Bang/Buck)
- **Cost**: $0.30-0.80/hr for RTX 4090
- **Setup**: 5 minutes
- **Link**: https://vast.ai

#### Option 2: RunPod
- **Cost**: $0.50-1.00/hr for RTX 4090
- **Setup**: 5 minutes
- **Link**: https://runpod.io

#### Option 3: AWS/GCP/Azure
- **Cost**: $1-3/hr (more expensive)
- **Setup**: More complex
- **Use case**: Production deployments

#### Option 4: Self-Hosted GPU Server
- **Cost**: $2,000-4,000 upfront + $50-100/mo electricity
- **Setup**: 1-2 hours
- **Use case**: Frequent training needs

---

## 🚀 Quick Start

### Step 1: Setup Remote GPU Server

**Using Vast.ai:**

```bash
# 1. Go to https://vast.ai
# 2. Search for instances with:
#    - GPU: RTX 4090 or A100
#    - Disk: 100GB+
#    - Upload speed: >100 Mbps

# 3. Rent instance with Docker image:
pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# 4. Note the SSH command provided, e.g.:
#    ssh -p 12345 root@ssh.vast.ai -L 8080:localhost:8080
```

**Using RunPod:**

```bash
# 1. Go to https://runpod.io
# 2. Click "Rent GPU Instances"
# 3. Select RTX 4090 or A100
# 4. Choose "PyTorch" template
# 5. Click "Deploy"
# 6. Copy SSH connection details
```

### Step 2: Configure Remote Connection

Edit `config/remote_gpu.yaml`:

```yaml
remote_gpu:
  host: "ssh4.vast.ai"  # Your instance host
  user: "root"
  port: 12345           # Your SSH port
  remote_path: "/workspace/personal_quant_desk"
```

### Step 3: Test Connection

```bash
# Test SSH connection
ssh -p 12345 root@ssh4.vast.ai

# If successful, you'll see the remote shell
```

### Step 4: First Training Run

```bash
# From your M1 MacBook, run:
python scripts/remote_train.py \
    --config config/remote_gpu.yaml \
    --model lstm

# This will:
# 1. Sync code to remote server
# 2. Run training on GPU
# 3. Download trained model
# 4. Shutdown remote instance (to save costs)
```

---

## 📊 Workflow Examples

### Example 1: Train LSTM Model

```bash
# One-line command from your M1 MacBook:
python scripts/remote_train.py --config config/remote_gpu.yaml --model lstm

# What happens:
# [M1] Sync code → [GPU] Train LSTM → [GPU] Export to ONNX → [M1] Download model
# Time: ~30 minutes on RTX 4090
# Cost: $0.25 (30 min × $0.50/hr)
```

### Example 2: Train All Models

```bash
# Train LSTM, Transformer, and RL agent sequentially
python scripts/remote_train.py --config config/remote_gpu.yaml --model all

# Time: ~3-4 hours
# Cost: $2-3 (4 hrs × $0.50/hr)
```

### Example 3: Hyperparameter Optimization

```bash
# Modify config/lstm_config.yaml:
training:
  hyperopt:
    enabled: true
    max_trials: 100

# Run hyperparameter search
python scripts/remote_train.py --config config/remote_gpu.yaml --model lstm

# Time: ~8-10 hours (100 trials × 5 min)
# Cost: $4-5
```

### Example 4: Monitor Training Live

```bash
# Terminal 1: Start training
python scripts/remote_train.py --config config/remote_gpu.yaml --model lstm

# Terminal 2: Monitor GPU usage
python scripts/remote_train.py --config config/remote_gpu.yaml --monitor

# Or use WandB dashboard:
https://wandb.ai/your-username/quant-desk
```

---

## 🎮 Daily Workflow

### Morning: Development on M1

```bash
# Work on strategies, backtest, develop features
cd /home/user/personal_quant_desk

# Run backtests (fast on M1)
python backtesting/backtest_engine.py

# Develop new features
jupyter notebook notebooks/feature_engineering.ipynb
```

### Afternoon: Submit Training Jobs

```bash
# Finished developing new features? Train models on remote GPU
python scripts/remote_train.py --config config/remote_gpu.yaml --model all

# Go get coffee ☕
# Training happens on remote GPU while you do other work
```

### Evening: Deploy New Models

```bash
# Models are automatically downloaded to models/trained/
# Run inference on M1 using ONNX
python models/deep_learning/inference_onnx.py

# Benchmark inference speed (should be <10ms on M1)
```

---

## 💰 Cost Optimization Strategies

### 1. Auto-Shutdown After Training

```yaml
# In config/remote_gpu.yaml
remote_gpu:
  auto_shutdown: true  # ✓ Enabled by default
```

This saves you from forgetting to shut down instances!

### 2. Use Spot Instances (50-70% cheaper)

On Vast.ai/RunPod, enable "Interruptible" instances:
- Cost: $0.15-0.30/hr (vs $0.50-0.80/hr)
- Risk: May be interrupted, but training checkpoints every 10 epochs

### 3. Train During Off-Peak Hours

- Peak hours (9am-5pm): Higher prices
- Off-peak (nights/weekends): 20-30% cheaper

### 4. Batch Training Jobs

Instead of:
```bash
# Train LSTM (30 min) → Shutdown → Start new instance → Train Transformer (60 min)
# Cost: 2× startup overhead
```

Do this:
```bash
# Train all models in one session (90 min total)
python scripts/remote_train.py --model all
# Cost: No startup overhead, saves ~$0.50
```

### 5. Use Model Distillation

Train large models on GPU, then distill to smaller models for M1 inference:
- Large LSTM (512 hidden): Train on GPU
- Small LSTM (128 hidden): Distill from large, run on M1
- Performance: 90% of large model, 4x faster inference

---

## 🔍 Monitoring & Debugging

### Check GPU Utilization

```bash
# From M1, check remote GPU usage
python scripts/remote_train.py --config config/remote_gpu.yaml --monitor

# Output:
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 525.105.17   Driver Version: 525.105.17   CUDA Version: 12.0     |
# |-------------------------------+----------------------+----------------------+
# | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
# | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
# |===============================+======================+======================|
# |   0  NVIDIA RTX 4090     On   | 00000000:01:00.0 Off |                  Off |
# | 30%   62C    P2   320W / 450W |  22000MiB / 24564MiB |     98%      Default |
# +-------------------------------+----------------------+----------------------+
```

### View Training Logs

```bash
# SSH into remote server
ssh -p 12345 root@ssh4.vast.ai

# View training logs
cd /workspace/personal_quant_desk
tail -f training.log
```

### WandB Dashboard

1. Enable in `config/lstm_config.yaml`:
```yaml
tracking:
  wandb: true
```

2. Visit: https://wandb.ai/your-username/quant-desk

3. View:
   - Training/validation loss curves
   - GPU utilization
   - Learning rate schedules
   - Model predictions vs actuals

---

## 🐛 Troubleshooting

### Issue: SSH Connection Refused

```bash
# Check if instance is running
# On Vast.ai: Go to https://vast.ai/console/instances/
# Status should be "Running"

# Test connection
ping ssh4.vast.ai
```

### Issue: Training OOM (Out of Memory)

```yaml
# Reduce batch size in config/lstm_config.yaml
training:
  batch_size: 64  # Was 128
```

### Issue: Slow Training

```bash
# Check GPU utilization
python scripts/remote_train.py --monitor

# If <50% utilization:
# 1. Increase batch size
# 2. Enable mixed precision (already enabled in train_lstm.py)
# 3. Use data parallelism (multi-GPU)
```

### Issue: Model Download Fails

```bash
# Manually download models
rsync -avz -e "ssh -p 12345" \
    root@ssh4.vast.ai:/workspace/personal_quant_desk/models/trained/ \
    ./models/trained/
```

---

## 📈 Expected Performance

### Training Speed (vs M1 MacBook)

| Model | M1 MacBook (CPU) | RTX 4090 (GPU) | A100 (GPU) | Speedup |
|-------|------------------|----------------|------------|---------|
| LSTM | 4 hours | 20 minutes | 10 minutes | **12-24x** |
| Transformer | 12 hours | 45 minutes | 20 minutes | **16-36x** |
| RL Agent | 24 hours | 2 hours | 1 hour | **12-24x** |

### Inference Speed (on M1 MacBook with ONNX)

| Model | Latency | Throughput |
|-------|---------|------------|
| LSTM | 5-8ms | 125-200 samples/sec |
| Transformer | 10-15ms | 65-100 samples/sec |
| Ensemble (3 models) | 20-30ms | 33-50 samples/sec |

All inference runs on M1 CPU, fast enough for real-time trading!

---

## 🎓 Next Steps

### 1. Advanced Features

**Multi-GPU Training:**
```python
# In train_lstm.py, enable DataParallel:
model = nn.DataParallel(model)  # Uses all available GPUs
```

**Distributed Training:**
```bash
# Use multiple instances in parallel
python scripts/remote_train.py --instances 4 --model lstm
```

**Quantization:**
```yaml
# In config/lstm_config.yaml
export:
  quantization: true  # INT8 quantization (4x smaller, 2x faster)
```

### 2. Add More Models

Create `models/deep_learning/train_transformer.py` following the same pattern:
- GPU training script
- ONNX export
- M1 inference

### 3. Alternative Data

Add sentiment analysis:
```bash
# Train FinBERT on GPU
python scripts/remote_train.py --model finbert

# Run inference on M1 (fast)
```

### 4. Production Deployment

**Option A: Keep models on M1 (simple)**
- All inference on M1 MacBook
- Retrain weekly on remote GPU

**Option B: Cloud deployment (scalable)**
- Deploy models to AWS Lambda / GCP Cloud Run
- Use M1 for development only

---

## 💡 Pro Tips

1. **Use tmux on remote server** - Training continues even if SSH disconnects
   ```bash
   ssh -p 12345 root@ssh4.vast.ai
   tmux new -s training
   python models/deep_learning/train_lstm.py
   # Press Ctrl+B, then D to detach
   ```

2. **Version control models** - Use MLflow or DVC
   ```bash
   pip install mlflow
   mlflow ui  # View all model versions
   ```

3. **Cache data on remote server** - Don't re-download market data every time
   ```bash
   # First run: Download data (slow)
   # Subsequent runs: Use cached data (fast)
   ```

4. **Use Jupyter on remote GPU** - For interactive development
   ```bash
   ssh -p 12345 -L 8888:localhost:8888 root@ssh4.vast.ai
   jupyter notebook --no-browser --port 8888
   # Open http://localhost:8888 on M1
   ```

---

## 📚 Resources

- **Vast.ai Docs**: https://vast.ai/docs
- **RunPod Docs**: https://docs.runpod.io
- **ONNX Runtime**: https://onnxruntime.ai
- **PyTorch ONNX Export**: https://pytorch.org/docs/stable/onnx.html
- **WandB**: https://docs.wandb.ai

---

## 🤝 Support

If you encounter issues:

1. Check troubleshooting section above
2. Review logs: `tail -f training.log`
3. Test with smaller model: `batch_size: 32`, `epochs: 10`

---

**Happy Trading! 🚀📈**
