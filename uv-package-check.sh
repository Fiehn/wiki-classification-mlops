#!/bin/bash

# Check if the UV package is installed
if python -c "import UV" 2>/dev/null; then
    echo "UV package is installed. Proceeding with the hook."
    # Add any commands you want to run if UV is installed
    uv run pip freeze > requirements.txt
    uv run --dev pip freeze > requirements-dev.txt
    git add requirements.txt requirements-dev.txt
else
    echo "UV package is not installed. Skipping the hook."
fi
