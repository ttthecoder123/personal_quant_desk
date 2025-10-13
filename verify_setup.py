#!/usr/bin/env python3
"""
Setup Verification Script
Verifies that the consolidated Personal Quant Desk structure is correct and functional.
"""

import os
import sys
from pathlib import Path

def verify_structure():
    """Verify the project structure is correct."""
    print("🔍 Verifying Personal Quant Desk Structure...")

    base_path = Path(__file__).parent

    # Required directories
    required_dirs = [
        "config",
        "data/ingestion",
        "data/processed",
        "data/cache",
        "data/catalog",
        "data/quality_reports",
        "data/config",
        "data/logs",
        "models",
        "strategies",
        "risk",
        "execution",
        "backtesting",
        "monitoring",
        "notebooks",
        "tests",
        "utils",
        "logs"
    ]

    # Required files
    required_files = [
        ".env.template",
        ".gitignore",
        "requirements.txt",
        "README.md",
        "config/config.yaml",
        "data/config/data_sources.yaml",
        "data/config/instruments.yaml",
        "data/main.py",
        "data/ingestion/__init__.py",
        "data/ingestion/alpha_vantage.py",
        "data/ingestion/downloader.py",
        "data/ingestion/validator.py",
        "data/ingestion/quality_scorer.py",
        "data/ingestion/storage.py",
        "data/ingestion/catalog.py"
    ]

    print("\n📁 Directory Structure:")
    missing_dirs = []
    for dir_path in required_dirs:
        full_path = base_path / dir_path
        if full_path.exists():
            print(f"  ✅ {dir_path}")
        else:
            print(f"  ❌ {dir_path} - MISSING")
            missing_dirs.append(dir_path)

    print("\n📄 Required Files:")
    missing_files = []
    for file_path in required_files:
        full_path = base_path / file_path
        if full_path.exists():
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path} - MISSING")
            missing_files.append(file_path)

    return len(missing_dirs) == 0 and len(missing_files) == 0

def verify_configuration():
    """Verify configuration files are properly set up."""
    print("\n🔧 Configuration Verification:")

    base_path = Path(__file__).parent

    # Check .env.template for Alpha Vantage
    env_template = base_path / ".env.template"
    if env_template.exists():
        content = env_template.read_text()
        if "ALPHA_VANTAGE_API_KEY" in content:
            print("  ✅ .env.template contains Alpha Vantage configuration")
        else:
            print("  ❌ .env.template missing Alpha Vantage configuration")
            return False

    # Check data_sources.yaml for Alpha Vantage config
    data_sources = base_path / "data/config/data_sources.yaml"
    if data_sources.exists():
        content = data_sources.read_text()
        if "alpha_vantage:" in content and "rate_limits:" in content:
            print("  ✅ data_sources.yaml contains Alpha Vantage configuration")
        else:
            print("  ❌ data_sources.yaml missing proper Alpha Vantage configuration")
            return False

    return True

def verify_data_ingestion():
    """Verify data ingestion modules can be imported."""
    print("\n🔄 Data Ingestion Module Verification:")

    # Add data directory to path for testing
    data_path = Path(__file__).parent / "data"
    sys.path.insert(0, str(data_path))

    modules_to_test = [
        ("ingestion", "Ingestion package"),
        ("ingestion.alpha_vantage", "Alpha Vantage adapter"),
        ("ingestion.downloader", "Data downloader with HybridDataManager"),
        ("ingestion.validator", "Data validator with corporate actions"),
        ("ingestion.quality_scorer", "Quality scoring system"),
        ("ingestion.storage", "Parquet storage system"),
        ("ingestion.catalog", "Data catalog system")
    ]

    all_passed = True
    for module_name, description in modules_to_test:
        try:
            __import__(module_name)
            print(f"  ✅ {description}")
        except ImportError as e:
            print(f"  ❌ {description} - Import failed: {e}")
            all_passed = False
        except Exception as e:
            print(f"  ⚠️  {description} - Import succeeded but error: {e}")

    return all_passed

def verify_cli():
    """Verify the CLI is accessible."""
    print("\n🖥️  CLI Verification:")

    main_py = Path(__file__).parent / "data/main.py"
    if main_py.exists():
        print("  ✅ data/main.py exists")
        # Try to check if it has the basic structure
        content = main_py.read_text()
        if "cli.command()" in content and "HybridDataManager" in content:
            print("  ✅ CLI includes hybrid data management")
        else:
            print("  ⚠️  CLI may be missing hybrid features")
        return True
    else:
        print("  ❌ data/main.py missing")
        return False

def main():
    """Run all verification checks."""
    print("=" * 60)
    print("🚀 PERSONAL QUANT DESK - SETUP VERIFICATION")
    print("=" * 60)

    checks = [
        ("Project Structure", verify_structure),
        ("Configuration", verify_configuration),
        ("Data Ingestion Modules", verify_data_ingestion),
        ("CLI Interface", verify_cli)
    ]

    all_passed = True
    for check_name, check_func in checks:
        try:
            passed = check_func()
            if not passed:
                all_passed = False
        except Exception as e:
            print(f"\n❌ {check_name} verification failed with error: {e}")
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 VERIFICATION PASSED!")
        print("✅ Personal Quant Desk is ready for use!")
        print("\nNext Steps:")
        print("1. Copy .env.template to .env (Alpha Vantage API key already included)")
        print("2. Install requirements: pip install -r requirements.txt")
        print("3. Test data ingestion: cd data && python main.py --help")
        print("4. Download sample data: cd data && python main.py update --symbols SPY --days 5")
        print("5. Try hybrid mode: cd data && python main.py update --symbols AUDUSD=X --days 5")
    else:
        print("❌ VERIFICATION FAILED!")
        print("Please check the issues above and fix them before proceeding.")
    print("=" * 60)

    return all_passed

if __name__ == "__main__":
    sys.exit(0 if main() else 1)