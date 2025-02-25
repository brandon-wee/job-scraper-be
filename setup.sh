#!/bin/bash

# Exit on any error
set -e

# Activate virtual environment (if applicable)
if [ -d "venv" ]; then
    source venv/bin/activate  # For Linux/macOS
    # source venv/Scripts/activate  # For Windows (Git Bash)
fi

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Ensure requirements.txt exists before installing
if [ -f "requirements.txt" ]; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
else
    echo "Error: requirements.txt not found!"
    exit 1
fi

# Download required NLTK data
echo "Downloading NLTK data..."
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet')"

# Run database migrations (if using Flask-Migrate)
# Uncomment the line below if your app requires migrations
# echo "Running database migrations..."
# flask db upgrade || echo "No migrations found."

# Ensure the app runs on the correct port
export PORT=5000
echo "Port set to $PORT"

echo "Setup complete!"
