# Terminal Shortcuts for Personal Quant Desk

## 🚀 Quick Navigation with Virtual Environment

You can now type `quant` commands from any terminal to quickly access your Personal Quant Desk **with automatic virtual environment activation**!

### 📋 Available Commands

| Command | Action | Description |
|---------|--------|-------------|
| `quant` or `quant desk` | Navigate + activate venv | Opens to main project with Python packages |
| `quant data` | Data directory + activate venv | Ready for `python main.py` commands |
| `quant work` | Development environment | Perfect for Python development work |
| `quant verify` | Run setup verification | Checks project structure |
| `quant test` | Test Alpha Vantage API | Verifies API key works |
| `quant install` | Test installation | Comprehensive package verification |
| `quant status` | Show current status | Directory, venv, and project status |
| `quant deactivate` | Deactivate virtual environment | Exit Python environment |
| `quant start` | Show quick start guide | Displays setup instructions |

### 🎮 Usage Examples

```bash
# From anywhere in terminal - jump to main project with Python ready
quant desk
# Virtual environment is automatically activated!

# Development work - everything ready for coding
quant work
python -c "import yfinance, pandas; print('✅ Ready for development')"

# Jump directly to data directory with venv activated
quant data
python main.py update --symbols "SPY" --days 5  # All packages available!

# Quick Python testing from anywhere
quant work
python test_installation.py

# API testing from anywhere
quant test

# Check everything is working
quant status
```

### 📋 What You'll See

**When you type `quant desk`:**
```
🔧 Activating virtual environment...
🚀 Welcome to Personal Quant Desk!
📁 Directory: /Users/.../personal_quant_desk
🐍 Virtual env: venv

Quick commands:
  quant data    - Go to data directory + activate venv
  quant verify  - Run setup verification
  quant test    - Test API key
  quant start   - Quick start guide
  quant work    - Ready for Python development
```

**When you type `quant work`:**
```
🔧 Activating virtual environment...
💻 Development environment ready!
📁 Directory: /Users/.../personal_quant_desk
🐍 Virtual env: Active (venv)

Python packages available:
  • yfinance, alpha_vantage, pandas, numpy
  • Personal Quant Desk components

Quick tests:
  python test_installation.py
  python -c "import yfinance; print('✅ yfinance ready')"
```

**When you type `quant data`:**
```
🔧 Activating virtual environment...
📊 Data directory - Ready for ingestion commands
🐍 Virtual env: Active (venv)

Try: python main.py --help
```

## ⚙️ How It Works

The commands are implemented as a zsh function in your `~/.zshrc` file:

```bash
quant() {
    case "$1" in
        desk|"")
            cd "/path/to/personal_quant_desk"
            echo "🚀 Welcome to Personal Quant Desk!"
            # ... helpful info
            ;;
        data)
            cd "/path/to/personal_quant_desk/data"
            echo "📊 Data directory - Ready for ingestion commands"
            ;;
        # ... other commands
    esac
}
```

## 🔧 Manual Setup (if needed)

If you're using a different shell or want to set this up manually:

### For Bash (.bashrc or .bash_profile):
```bash
quant() {
    case "$1" in
        desk|"")
            cd "/Users/alexandergeorgieff/Desktop/Trading Project/personal_quant_desk"
            echo "🚀 Welcome to Personal Quant Desk!"
            ;;
        data)
            cd "/Users/alexandergeorgieff/Desktop/Trading Project/personal_quant_desk/data"
            echo "📊 Data directory - Ready for ingestion commands"
            ;;
        *)
            echo "Usage: quant [desk|data|verify|test|start]"
            ;;
    esac
}
```

### For Fish shell (~/.config/fish/config.fish):
```fish
function quant
    switch $argv[1]
        case desk ''
            cd "/Users/alexandergeorgieff/Desktop/Trading Project/personal_quant_desk"
            echo "🚀 Welcome to Personal Quant Desk!"
        case data
            cd "/Users/alexandergeorgieff/Desktop/Trading Project/personal_quant_desk/data"
            echo "📊 Data directory - Ready for ingestion commands"
        case '*'
            echo "Usage: quant [desk|data|verify|test|start]"
    end
end
```

## 🔄 Reload Your Shell

After setup, reload your shell configuration:

```bash
# For zsh
source ~/.zshrc

# For bash
source ~/.bashrc
# or
source ~/.bash_profile

# Or just open a new terminal
```

## 🎯 Quick Workflows

### Daily Development Workflow:
```bash
quant desk          # Jump to project
quant verify        # Check everything's working
quant data          # Go to data directory
python main.py update --symbols "SPY,AUDUSD=X" --days 1
```

### Quick API Testing:
```bash
quant test          # Test API key from anywhere
```

### Project Setup Verification:
```bash
quant verify        # Full system check
```

### View Documentation:
```bash
quant start         # Quick start guide
quant desk && cat README.md  # Full documentation
```

## 🚨 Troubleshooting

### Command not found
- Make sure you've reloaded your shell: `source ~/.zshrc`
- Or open a new terminal window
- Check that the function exists: `type quant`

### Wrong directory
- The function uses absolute paths, so it should work from anywhere
- If the path is wrong, edit `~/.zshrc` and update the paths

### Function conflicts
- If you have other `quant` commands, rename this function in `~/.zshrc`
- Example: rename to `qdesk` or `quantdesk`

---

**Now you can access your Personal Quant Desk from anywhere with just `quant desk`! 🎉**