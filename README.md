# Mobile Churn Prediction — End-to-End ML Project

## Folder structure

```
churn_project/
├── data/               raw + cleaned + engineered data at every stage
├── scripts/            01–04: the pipeline, run in order
├── artifacts/           saved model/pipeline objects (.joblib)
├── plots/               every chart generated along the way
├── notes/               Week 8 concept notes (clustering, PCA, metrics, etc.)
└── api/                 Step 5 — the FastAPI app
    ├── main.py
    ├── requirements.txt
    └── artifacts/churn_model_pipeline.joblib   (a copy — see note below)
```

## How to run Steps 1–4 (in VS Code, from a terminal)

These are plain Python scripts, not notebooks — run them one at a time
from the `churn_project/` folder, in order, since each one reads the
previous step's output from `data/`:

```
pip install pandas numpy scikit-learn matplotlib seaborn joblib
cd churn_project
python scripts/01_eda_cleaning.py
python scripts/02_feature_engineering.py
python scripts/03_pca_clustering.py
python scripts/04_model_training.py
```

Each script prints its findings to the terminal and writes its outputs into
`data/`, `artifacts/`, and `plots/`. If you'd rather work cell-by-cell,
you can paste each script into a Jupyter/Colab notebook — nothing about
Steps 1–4 requires the terminal specifically. Step 5 does (see below).

## How to run Step 5 (the API) — VS Code, not Colab

```
cd churn_project/api
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Then open **http://127.0.0.1:8000/docs** for interactive Swagger docs,
or test `POST /predict-churn` from Postman.

### How the API actually gets its model — no "notebook fetching" involved

`main.py` never touches any notebook or script at runtime. The connection
is just one file:

```
scripts/04_model_training.py  --(joblib.dump)-->  churn_model_pipeline.joblib  --(joblib.load)-->  api/main.py
```

`04_model_training.py` trains the pipeline once and serializes the
*fitted* object — preprocessing steps and all — to a single `.joblib`
file. `api/main.py` loads that file at server startup
(`joblib.load("artifacts/churn_model_pipeline.joblib")`) and just calls
`.predict_proba()` on it for each request. That's the entire link: a
saved file, not a live connection to Steps 1–4. This is why the same
copy of `churn_model_pipeline.joblib` sits in both `artifacts/` (from
training) and `api/artifacts/` (what the API actually loads) — the API
folder is meant to be deployable on its own, without the rest of the
project.

If you retrain the model later (edit `04_model_training.py`, rerun it),
you'll need to copy the new `churn_model_pipeline.joblib` into
`api/artifacts/` and restart the server for the API to pick up the change.
