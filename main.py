import pickle
import shap
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import time

with open("xg_model.pkl", "rb") as f:
    model = pickle.load(f)

with open('x_columns.pkl', 'rb') as f:
    x_columns = pickle.load(f)

background = pd.read_parquet("shap_background.parquet")

# explainer = shap.KernelExplainer(predict_proba_wrapper, background)
explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")

app = FastAPI()

class PredictRequest(BaseModel):
    features: list

@app.post("/predict")
def predict(req: PredictRequest):
    start_time = time.time()

    X = pd.DataFrame([req.features], columns=x_columns)
    prediction = model.predict_proba(X)[0]
    shap_values_single = explainer.shap_values(X)
    abs_shap_single = np.abs(shap_values_single[0])
    abs_shap_single /= abs_shap_single.sum()

    explanation_items = [
        (
            feature,
            {
                "shap_value": float(shap),
                "contribution": float(contribution),
                "impact": 1 if shap > 0 else 0
            }
        )
        for feature, shap, contribution in zip(x_columns, shap_values_single[0], abs_shap_single)
    ]
    explanation_items_sorted = sorted(explanation_items, key=lambda x: x[1]["contribution"], reverse=True)
    explanation_dict_sorted = dict(explanation_items_sorted)

    elapsed_time = time.time() - start_time

    return {
        "prediction": float(prediction[1]),
        "explanation": explanation_dict_sorted,
        "elapsed_time_seconds": round(elapsed_time, 4)
    }