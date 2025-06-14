import pickle
import shap
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import time
import xgboost as xgb
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

with open("xg_model.pkl", "rb") as f:
    model = pickle.load(f)

with open('x_columns.pkl', 'rb') as f:
    x_columns = pickle.load(f)

background = pd.read_parquet("shap_background.parquet")

def predict_proba_wrapper(X):
    if isinstance(X, (np.ndarray, list)):
        X = pd.DataFrame(X, columns=x_columns)
    return model.predict_proba(X)

explainer = shap.KernelExplainer(predict_proba_wrapper, background)

app = FastAPI()

class PredictRequest(BaseModel):
    features: list

@app.post("/predict")
def predict(req: PredictRequest):
    start_time = time.time()

    X = np.array([req.features])

    if isinstance(X, (np.ndarray, list)):
        X = pd.DataFrame(X, columns=x_columns)
    prediction = model.predict_proba(X)[0]

    sample_np = X.to_numpy().reshape(1, -1)
    shap_values_single = explainer.shap_values(sample_np)[0]
    abs_shap_single = np.abs(shap_values_single[:,1])
    abs_shap_single /= abs_shap_single.sum()

    explanation_items = [
        (
            feature,
            {
                "contribution": float(contribution),
                "impact": 1 if shap[1] > 0 else 0
            }
        )
        for feature, shap, contribution in zip(x_columns, shap_values_single, abs_shap_single)
    ]

    explanation_items_sorted = sorted(explanation_items, key=lambda x: x[1]["contribution"], reverse=True)
    explanation_dict_sorted = dict(explanation_items_sorted)

    elapsed_time = time.time() - start_time

    return {
        "prediction": float(prediction[1]),
        "explanation": explanation_dict_sorted,
        "elapsed_time_seconds": round(elapsed_time, 4)
    }