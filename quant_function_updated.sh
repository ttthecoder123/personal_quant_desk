# Personal Quant Desk - Enhanced Navigation with Virtual Environment
quant() {
    local project_dir="/Users/alexandergeorgieff/Desktop/Trading Project/personal_quant_desk"
    local venv_path="$project_dir/venv"

    case "$1" in
        desk|"")
            cd "$project_dir"
            if [[ "$VIRTUAL_ENV" != "$venv_path" ]]; then
                echo "🔧 Activating virtual environment..."
                source "$venv_path/bin/activate"
            fi
            echo "🚀 Welcome to Personal Quant Desk!"
            echo "📁 Directory: $(pwd)"
            echo "🐍 Virtual env: $(basename "$VIRTUAL_ENV" 2>/dev/null || echo "Not activated")"
            echo ""
            echo "Quick commands:"
            echo "  quant data    - Go to data directory + activate venv"
            echo "  quant verify  - Run setup verification"
            echo "  quant test    - Test API key"
            echo "  quant start   - Quick start guide"
            echo "  quant work    - Ready for Python development"
            ;;
        data)
            cd "$project_dir/data"
            if [[ "$VIRTUAL_ENV" != "$venv_path" ]]; then
                echo "🔧 Activating virtual environment..."
                source "$venv_path/bin/activate"
            fi
            echo "📊 Data directory - Ready for ingestion commands"
            echo "🐍 Virtual env: Active ($(basename "$VIRTUAL_ENV"))"
            echo ""
            echo "Try: python main.py --help"
            ;;
        work)
            cd "$project_dir"
            if [[ "$VIRTUAL_ENV" != "$venv_path" ]]; then
                echo "🔧 Activating virtual environment..."
                source "$venv_path/bin/activate"
            fi
            echo "💻 Development environment ready!"
            echo "📁 Directory: $(pwd)"
            echo "🐍 Virtual env: Active ($(basename "$VIRTUAL_ENV"))"
            echo ""
            echo "Python packages available:"
            echo "  • yfinance, alpha_vantage, pandas, numpy"
            echo "  • Personal Quant Desk components"
            echo ""
            echo "Quick tests:"
            echo "  python test_installation.py"
            echo "  python -c \"import yfinance; print('✅ yfinance ready')\""
            ;;
        verify)
            cd "$project_dir"
            if [[ "$VIRTUAL_ENV" != "$venv_path" ]]; then
                source "$venv_path/bin/activate" 2>/dev/null
            fi
            python3 verify_setup.py
            ;;
        test)
            cd "$project_dir"
            if [[ "$VIRTUAL_ENV" != "$venv_path" ]]; then
                source "$venv_path/bin/activate" 2>/dev/null
            fi
            python3 test_api_key.py
            ;;
        install)
            cd "$project_dir"
            if [[ "$VIRTUAL_ENV" != "$venv_path" ]]; then
                source "$venv_path/bin/activate" 2>/dev/null
            fi
            python test_installation.py
            ;;
        start)
            cd "$project_dir"
            cat QUICKSTART.md
            ;;
        deactivate)
            if [[ -n "$VIRTUAL_ENV" ]]; then
                deactivate
                echo "🔌 Virtual environment deactivated"
            else
                echo "ℹ️  No virtual environment is currently active"
            fi
            ;;
        status)
            echo "📍 Current directory: $(pwd)"
            if [[ -n "$VIRTUAL_ENV" ]]; then
                echo "🐍 Virtual env: Active ($(basename "$VIRTUAL_ENV"))"
                echo "📦 Python: $(python --version 2>/dev/null || echo "Not available")"
            else
                echo "🐍 Virtual env: Not active"
            fi
            if [[ -d "$project_dir" ]]; then
                echo "📁 Project dir: ✅ Found"
                if [[ -d "$venv_path" ]]; then
                    echo "📦 Virtual env: ✅ Available at venv/"
                else
                    echo "📦 Virtual env: ❌ Not found"
                fi
            else
                echo "📁 Project dir: ❌ Not found"
            fi
            ;;
        *)
            echo "Usage: quant [command]"
            echo ""
            echo "Navigation:"
            echo "  quant desk      - Go to main project + activate venv (default)"
            echo "  quant data      - Go to data directory + activate venv"
            echo "  quant work      - Development environment + activate venv"
            echo ""
            echo "Development:"
            echo "  quant verify    - Run setup verification"
            echo "  quant test      - Test Alpha Vantage API key"
            echo "  quant install   - Test installation"
            echo "  quant start     - Show quick start guide"
            echo ""
            echo "Utility:"
            echo "  quant status    - Show current status"
            echo "  quant deactivate - Deactivate virtual environment"
            ;;
    esac
}