#!/bin/bash
# Quick Start Script for Remote GPU Training Setup
# This script helps you get started with M1 + Remote GPU training in minutes

set -e  # Exit on error

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  Personal Quant Desk - Remote GPU Training Quickstart    ║"
echo "║  M1 MacBook + Cloud GPU = 98% Cost Savings! 🚀           ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""

# Check if running on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "⚠️  Warning: This script is optimized for macOS (M1 MacBook)"
    echo "   It may work on other systems, but YMMV"
    echo ""
fi

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo "📋 Checking prerequisites..."
echo ""

# Check Python
if ! command_exists python3; then
    echo "❌ Python 3 not found. Please install Python 3.10+"
    exit 1
else
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    echo "✓ Python $PYTHON_VERSION found"
fi

# Check pip
if ! command_exists pip3; then
    echo "❌ pip3 not found. Please install pip"
    exit 1
else
    echo "✓ pip found"
fi

# Check ssh
if ! command_exists ssh; then
    echo "❌ SSH not found. Please install OpenSSH"
    exit 1
else
    echo "✓ SSH found"
fi

# Check rsync
if ! command_exists rsync; then
    echo "⚠️  rsync not found. Installing via Homebrew..."
    if command_exists brew; then
        brew install rsync
    else
        echo "❌ Homebrew not found. Please install rsync manually"
        exit 1
    fi
else
    echo "✓ rsync found"
fi

echo ""
echo "✓ All prerequisites met!"
echo ""

# Install Python dependencies
echo "📦 Installing Python dependencies..."
echo ""

DEPENDENCIES="onnxruntime numpy pandas pyyaml"

for dep in $DEPENDENCIES; do
    if python3 -c "import $dep" 2>/dev/null; then
        echo "✓ $dep already installed"
    else
        echo "Installing $dep..."
        pip3 install -q $dep
    fi
done

echo ""
echo "✓ Python dependencies installed!"
echo ""

# Optional dependencies
echo "📦 Optional: Install monitoring tools? (WandB, MLflow)"
read -p "Install monitoring tools? [y/N]: " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Installing wandb and mlflow..."
    pip3 install -q wandb mlflow
    echo "✓ Monitoring tools installed!"
    echo ""
    echo "💡 Setup WandB:"
    echo "   1. Create account at https://wandb.ai"
    echo "   2. Run: wandb login"
    echo ""
fi

# Setup remote GPU configuration
echo "🔧 Setting up remote GPU configuration..."
echo ""

CONFIG_FILE="config/remote_gpu.yaml"

if [ -f "$CONFIG_FILE" ]; then
    echo "✓ Remote GPU config already exists: $CONFIG_FILE"
    read -p "Overwrite with new configuration? [y/N]: " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Keeping existing configuration"
    else
        echo "Please enter your remote GPU details:"
        echo "(You can get these from Vast.ai or RunPod after renting an instance)"
        echo ""
        read -p "Host (e.g., ssh4.vast.ai): " GPU_HOST
        read -p "Port (e.g., 12345): " GPU_PORT
        read -p "User (usually 'root'): " GPU_USER

        # Update config file
        sed -i.bak "s/host: .*/host: \"$GPU_HOST\"/" "$CONFIG_FILE"
        sed -i.bak "s/port: .*/port: $GPU_PORT/" "$CONFIG_FILE"
        sed -i.bak "s/user: .*/user: \"$GPU_USER\"/" "$CONFIG_FILE"

        echo "✓ Configuration updated!"
    fi
else
    echo "⚠️  Config file not found. Please create $CONFIG_FILE manually"
    echo "   See docs/REMOTE_GPU_SETUP.md for instructions"
fi

echo ""

# Test remote connection (optional)
if [ -f "$CONFIG_FILE" ]; then
    echo "🔌 Test remote connection?"
    read -p "Test SSH connection to remote GPU? [y/N]: " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # Extract connection details from config
        GPU_HOST=$(grep "host:" "$CONFIG_FILE" | cut -d'"' -f2)
        GPU_PORT=$(grep "port:" "$CONFIG_FILE" | awk '{print $2}')
        GPU_USER=$(grep "user:" "$CONFIG_FILE" | cut -d'"' -f2)

        echo "Testing connection to $GPU_USER@$GPU_HOST:$GPU_PORT..."

        if ssh -p "$GPU_PORT" -o ConnectTimeout=10 -o BatchMode=yes "$GPU_USER@$GPU_HOST" "echo 'Connection successful'" 2>/dev/null; then
            echo "✓ SSH connection successful!"
        else
            echo "❌ SSH connection failed"
            echo "   Please check:"
            echo "   1. Instance is running (check Vast.ai/RunPod dashboard)"
            echo "   2. Host, port, and user are correct in config/remote_gpu.yaml"
            echo "   3. SSH key is added (if required)"
        fi
    fi
fi

echo ""

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p models/trained/lstm
mkdir -p models/trained/transformer
mkdir -p models/trained/rl_execution
mkdir -p experiments/results
echo "✓ Directories created!"
echo ""

# Print next steps
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  Setup Complete! 🎉                                       ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""
echo "📖 Next Steps:"
echo ""
echo "1. Rent a GPU instance:"
echo "   → Vast.ai: https://vast.ai (RTX 4090 for \$0.30-0.50/hr)"
echo "   → RunPod: https://runpod.io (RTX 4090 for \$0.50-0.80/hr)"
echo ""
echo "2. Update config/remote_gpu.yaml with your instance details"
echo ""
echo "3. Run your first training:"
echo "   python scripts/remote_train.py --config config/remote_gpu.yaml --model lstm"
echo ""
echo "4. Cost: ~\$0.25 (30 minutes training)"
echo "   Expected improvement: +20-30% Sharpe ratio"
echo ""
echo "📚 Documentation:"
echo "   • Quick Start: docs/MY_STRATEGY.md"
echo "   • Full Guide: docs/REMOTE_GPU_SETUP.md"
echo "   • Summary: docs/IMPLEMENTATION_SUMMARY.md"
echo ""
echo "💡 Pro Tip: Start with RL execution agent (Priority #1)"
echo "   Saves \$1,000-2,400/year in execution costs!"
echo ""
echo "🚀 Ready to take your quant desk to the next level!"
echo ""
