# Real-Time ML Inference REST API & Capstone

    Production-style FastAPI microservice wrapping the **Task 3 champion Logistic Regression model**.

    ## Important project boundary

    This capstone is a **new repository/project**. It does not modify or replace the previously submitted Task 3 repository.

    The model artifact was independently reconstructed from the same Task 3 champion configuration:
    - Dataset: scikit-learn Breast Cancer Wisconsin Diagnostic dataset
    - Target: `1 = malignant`, `0 = benign`
    - Train/test split: 80/20, stratified, `random_state=42`
    - Preprocessing: median imputation + `StandardScaler`
    - Classifier: `LogisticRegression(C=1.0, solver="liblinear")`
    - Artifact: `models/champion_model.joblib`

    **Educational use only. This service is not a clinical diagnostic system.**

    ## Architecture

    ```text
    Client
       |
       | POST /predict
       | JSON: { "features": [30 numeric values] }
       v
    FastAPI + Pydantic validation
       |
       v
    Serialized sklearn Pipeline
       |-- Median imputation
       |-- StandardScaler
       `-- Logistic Regression
       |
       v
    Prediction + probability response
    ```

    ## Project structure

    ```text
    ML-Inference-REST-API/
    ├── app/
    │   ├── __init__.py
    │   └── main.py
    ├── models/
    │   └── champion_model.joblib
    ├── tests/
    │   └── test_api.py
    ├── Dockerfile
    ├── requirements.txt
    ├── README.md
    └── .gitignore
    ```

    ## API endpoints

    | Method | Endpoint | Purpose |
    |---|---|---|
    | GET | `/health` | Service health check |
    | GET | `/model-info` | Model and feature metadata |
    | POST | `/predict` | Return class prediction and probabilities |

    FastAPI automatically exposes interactive documentation at `/docs` and an OpenAPI schema at `/openapi.json`.

    ## Prediction request

    The request must contain exactly 30 numeric values in this order:

    | Index | Feature | Type |
    |---:|---|---|
    | `0` | `mean radius` | numeric |
| `1` | `mean texture` | numeric |
| `2` | `mean perimeter` | numeric |
| `3` | `mean area` | numeric |
| `4` | `mean smoothness` | numeric |
| `5` | `mean compactness` | numeric |
| `6` | `mean concavity` | numeric |
| `7` | `mean concave points` | numeric |
| `8` | `mean symmetry` | numeric |
| `9` | `mean fractal dimension` | numeric |
| `10` | `radius error` | numeric |
| `11` | `texture error` | numeric |
| `12` | `perimeter error` | numeric |
| `13` | `area error` | numeric |
| `14` | `smoothness error` | numeric |
| `15` | `compactness error` | numeric |
| `16` | `concavity error` | numeric |
| `17` | `concave points error` | numeric |
| `18` | `symmetry error` | numeric |
| `19` | `fractal dimension error` | numeric |
| `20` | `worst radius` | numeric |
| `21` | `worst texture` | numeric |
| `22` | `worst perimeter` | numeric |
| `23` | `worst area` | numeric |
| `24` | `worst smoothness` | numeric |
| `25` | `worst compactness` | numeric |
| `26` | `worst concavity` | numeric |
| `27` | `worst concave points` | numeric |
| `28` | `worst symmetry` | numeric |
| `29` | `worst fractal dimension` | numeric |

    Example:

    ```json
    {
      "features": [17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]
    }
    ```

    Example response:

    ```json
    {
      "prediction": 1,
      "prediction_label": "malignant",
      "probability_benign": 0.01,
      "probability_malignant": 0.99
    }
    ```

    ## Run locally

    Create and activate a virtual environment:

    ```powershell
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    ```

    Install pinned dependencies:

    ```powershell
    pip install -r requirements.txt
    ```

    Start the API:

    ```powershell
    uvicorn app.main:app --reload
    ```

    Open:

    - API docs: http://127.0.0.1:8000/docs
    - Health: http://127.0.0.1:8000/health

    ## Unit tests

    ```powershell
    pytest -q
    ```

    Tests validate:
    - health response
    - model metadata
    - prediction response schema
    - probability bounds and sum
    - invalid feature-count handling

    ## Docker

    Build:

    ```powershell
    docker build -t ml-inference-api .
    ```

    Run:

    ```powershell
    docker run --rm -p 8000:8000 ml-inference-api
    ```

    Then open `http://127.0.0.1:8000/docs`.

    ## Production considerations

    For a real deployment, add authentication/authorization, HTTPS/TLS, structured logging, request IDs, rate limiting, monitoring, model/data drift checks, resource limits, image vulnerability scanning, CI/CD, and a model/version registry.

    ## Certification evidence

    This repository provides the expected capstone proof:
    - serialized champion model
    - FastAPI REST service
    - `/predict` endpoint with probabilities
    - Dockerfile
    - pinned dependencies
    - automated unit tests
    - interactive API documentation
    - end-to-end ML architecture documentation
