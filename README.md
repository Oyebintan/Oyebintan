# Hybrid Feature Selection Spam Classifier

This project provides a FastAPI backend and a static frontend for a hybrid feature selection
pipeline (Chi-Square + Mutual Information) paired with a lightweight deep learning model for
email spam classification.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn backend.main:app --reload
```

Open `http://localhost:8000` to access the UI. Use the **Train** button to build the model
from the bundled sample dataset and then classify messages.
