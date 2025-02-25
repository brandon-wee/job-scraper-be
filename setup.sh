#!/bin/bash

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet')"

# Run database migrations (if using Flask-Migrate)
# flask db upgrade

# Ensure the app runs on the right port
export PORT=5000
