#!/bin/bash
# TA ULS Training System Setup Script

echo "=========================================="
echo "TA ULS Training System Setup"
echo "=========================================="

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed"
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"

# Check Julia
if ! command -v julia &> /dev/null; then
    echo "⚠️  Julia not found. Julia integration will be disabled."
    echo "   To enable Julia features, install Julia from: https://julialang.org/"
    JULIA_AVAILABLE=false
else
    echo "✅ Julia found: $(julia --version)"
    JULIA_AVAILABLE=true
fi

# Install Python dependencies
echo ""
echo "Installing Python dependencies..."
pip3 install -r requirements.txt
if [ $? -eq 0 ]; then
    echo "✅ Python dependencies installed"
else
    echo "❌ Failed to install Python dependencies"
    exit 1
fi

# Install Julia dependencies (if Julia is available)
if [ "$JULIA_AVAILABLE" = true ]; then
    echo ""
    echo "Installing Julia dependencies..."
    julia install_julia_deps.jl
    if [ $? -eq 0 ]; then
        echo "✅ Julia dependencies installed"
    else
        echo "⚠️  Julia dependency installation had issues"
    fi
fi

# Make scripts executable
chmod +x test_ta_uls.py
chmod +x al-ULs.Py

echo ""
echo "=========================================="
echo "Running System Tests"
echo "=========================================="

# Run tests
python3 test_ta_uls.py
TEST_RESULT=$?

echo ""
echo "=========================================="
echo "Setup Complete"
echo "=========================================="

if [ $TEST_RESULT -eq 0 ]; then
    echo "🎉 Setup completed successfully!"
    echo ""
    echo "Quick Start:"
    echo "1. To run full training demo:"
    echo "   python3 al-ULs.Py"
    echo ""
    echo "2. To start Julia server manually:"
    echo "   julia -e 'include(\"Al-uLS.jl\"); start_http_server(8000)'"
    echo ""
    echo "3. To run tests:"
    echo "   python3 test_ta_uls.py"
else
    echo "⚠️  Setup completed with some test failures"
    echo "   Core Python functionality should work"
    echo "   Check test output for specific issues"
fi

echo ""
echo "For detailed documentation, see: TA_ULS_README.md"