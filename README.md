# Fashion MNIST Production

This project is a production-style machine learning workflow for the Fashion MNIST dataset.

The goal is to train a model, test the project code, and serve the trained model with a FastAPI application.

## What The Project Does

The model classifies grayscale clothing images from Fashion MNIST into one of 10 classes, for example:

- T-shirt/top
- Trouser
- Dress
- Sneaker
- Bag

Each image has size `28x28`, so the API expects `784` pixel values.

## Project Structure

```text
app/
  main.py        FastAPI application
  model.py       model architecture and loading
  predict.py     prediction logic
  schemas.py     API input and output schemas

app/ml/
  dataset.py             dataset class
  preprocessing.py       preprocessing helpers
  feature_selection.py   feature selection helpers
  training.py            training loop
  class_names.py         Fashion MNIST class names

notebooks/
  01_eda.ipynb
  02_feature_engineering.ipynb
  03_training_pipeline.ipynb

tests/
  unit and API tests

models/
  saved model and preprocessing parameters
```

## Notebooks

The project started with three notebooks:

1. `01_eda.ipynb` - exploratory data analysis
2. `02_feature_engineering.ipynb` - data split and preprocessing
3. `03_training_pipeline.ipynb` - model training and saving the best model

## Setup

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run Tests

Run tests locally:

```bash
python -m pytest
```

Run Ruff:

```bash
python -m ruff check .
```

## Docker

Build the Docker image:

```bash
docker build -t fashion-mnist-checks .
```

Run tests and linting inside Docker:

```bash
docker run --rm fashion-mnist-checks
```

## FastAPI

Run the API locally:

```bash
uvicorn app.main:app --reload
```

Open Swagger UI in the browser:

```text
http://127.0.0.1:8000/docs
```

## Prediction Endpoint

Endpoint:

```text
POST /predict
```

Example request body:

```json
{
  "pixels": [0, 0, 0, 0]
}
```

The example is shortened. A real request must contain exactly `784` pixel values between `0` and `255`.

Example response:

```json
{
  "predicted_class": 3,
  "class_name": "Dress",
  "confidence": 0.98
}
```

## CI

GitHub Actions runs Ruff and Pytest inside Docker on pushes and pull requests.

## Branch Workflow

The project uses feature branches.

Typical workflow:

```bash
git checkout develop
git pull origin develop
git checkout -b feature/task-name
```
