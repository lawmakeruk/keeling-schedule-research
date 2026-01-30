#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status

echo "Setting up the Keeling Schedule Flask App..."

# Create the virtual environment
if [ ! -d .venv ]; then
    if command -v python3.12 &>/dev/null; then
        python3.12 -m venv .venv
        echo "✅ Virtual environment created with python3.12"
    elif command -v py &>/dev/null; then
        py -3.12 -m venv .venv
        echo "✅ Virtual environment created with py -3.12"
    else
        echo "❌ Python 3.12 not found"
        exit 1
    fi
fi

# Activate the virtual environment
if [[ -f ".venv/bin/activate" ]]; then
    source .venv/bin/activate
    echo "✅ Activated virtual environment (.venv/bin/activate)"
elif [[ -f ".venv/Scripts/activate" ]]; then
    source .venv/Scripts/activate
    echo "✅ Activated virtual environment (.venv/Scripts/activate)"
else
    echo "❌ Could not find virtual environment activation script"
    exit 1
fi

# Confirm venv is active
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "❌ Failed to activate virtual environment"
    exit 1
else
    echo "✅ Virtual environment activated: $VIRTUAL_ENV"
fi

echo "Installing dependencies..."
python -m pip install --upgrade pip
python -m pip install -q -r requirements.txt
echo "✅ Dependencies installed!"

echo "Running auto-formatting with Black..."
python -m black app/ tests/
echo "✅ Code formatting complete!"

echo "Running linting checks..."
if ! python -m flake8 app/ tests/; then
    echo ""
    echo "❌ Linting failed even after auto-formatting!"
    echo "Please fix the above issues manually before proceeding."
    exit 1
fi

echo "✅ Linting passed!"

echo "Running unit tests with coverage..."
if ! python -m pytest --cov=app tests/ --cov-report=term-missing --cov-report=html; then
    echo "❌ Tests failed! Please review the above errors."
    exit 1
fi

echo "✅ All unit tests passed!"

echo "Starting Flask server..."
export FLASK_APP="app"
export FLASK_ENV=development
python -m flask run --debug