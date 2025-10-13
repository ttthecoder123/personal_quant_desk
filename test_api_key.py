#!/usr/bin/env python3
"""
Quick API Key Test
Tests the Alpha Vantage API key without requiring full installation.
"""

import os
import sys
from pathlib import Path

def test_env_loading():
    """Test loading the .env file."""
    print("🔧 Testing environment configuration...")

    # Check if .env exists
    env_file = Path(__file__).parent / ".env"
    env_template = Path(__file__).parent / ".env.template"

    if not env_file.exists():
        if env_template.exists():
            print("📝 .env file not found, copying from .env.template...")
            with open(env_template, 'r') as src, open(env_file, 'w') as dst:
                dst.write(src.read())
            print("✅ Created .env file")
        else:
            print("❌ Neither .env nor .env.template found")
            return False

    # Read the API key
    with open(env_file, 'r') as f:
        content = f.read()

    # Extract Alpha Vantage API key
    for line in content.split('\n'):
        if line.startswith('ALPHA_VANTAGE_API_KEY='):
            api_key = line.split('=', 1)[1].strip()
            if api_key and api_key != 'your_alpha_vantage_api_key':
                print(f"✅ Alpha Vantage API key found: {api_key[:8]}...")
                return api_key

    print("❌ Alpha Vantage API key not found or not set")
    return False

def test_api_key_simple(api_key):
    """Test the API key with a simple request."""
    print("\n🌐 Testing Alpha Vantage API key...")

    try:
        import urllib.request
        import json
    except ImportError:
        print("❌ Required modules not available")
        return False

    # Test API call
    test_url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=IBM&interval=5min&apikey={api_key}"

    try:
        print("📡 Making test API call...")
        with urllib.request.urlopen(test_url, timeout=10) as response:
            data = json.loads(response.read().decode())

        if "Error Message" in data:
            print(f"❌ API Error: {data['Error Message']}")
            return False
        elif "Note" in data:
            print(f"⚠️  API Note: {data['Note']}")
            return True  # Note usually means rate limit, but key works
        elif "Time Series (5min)" in data:
            print("✅ API key working! Successfully retrieved data")
            return True
        elif "Information" in data:
            print(f"ℹ️  API Info: {data['Information']}")
            return True
        else:
            print("⚠️  Unexpected response format, but no error")
            print("Response keys:", list(data.keys())[:3])
            return True

    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False

def main():
    """Run the API key test."""
    print("=" * 50)
    print("🔑 ALPHA VANTAGE API KEY TEST")
    print("=" * 50)

    # Test environment loading
    api_key = test_env_loading()
    if not api_key:
        print("\n❌ Environment test failed")
        return False

    # Test API key
    api_works = test_api_key_simple(api_key)

    print("\n" + "=" * 50)
    if api_works:
        print("🎉 SUCCESS! Alpha Vantage API key is working")
        print("\nYou can now:")
        print("1. Install requirements: pip install -r requirements.txt")
        print("2. Test data download: cd data && python main.py update --symbols AUDUSD=X --days 3")
        print("3. View the Quick Start guide: cat QUICKSTART.md")
    else:
        print("❌ API key test failed")
        print("\nOptions:")
        print("1. Get a free API key at: https://www.alphavantage.co/support/#api-key")
        print("2. Use yfinance only: cd data && python main.py update --no-hybrid")
    print("=" * 50)

    return api_works

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)