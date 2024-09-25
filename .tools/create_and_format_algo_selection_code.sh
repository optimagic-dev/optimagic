#!/bin/bash

# Check if the project is already installed locally
if ! pip show optimagic &> /dev/null; then
    # Install the project locally
    pip install -e .
fi

# Check if ruff is installed
if ! ruff --version &> /dev/null; then
    # Install ruff
    pip install ruff
fi

# Run the Python script to create algo_selection.py
python .tools/create_algo_selection_code.py

# Run ruff format on the created file
ruff format src/optimagic/algo_selection.py --silent --config pyproject.toml

# Run ruff lint with fixes on the created file
ruff check src/optimagic/algo_selection.py --fix --silent --config pyproject.toml
