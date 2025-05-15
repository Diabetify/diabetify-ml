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

X_test = pd.read_csv("./X_test.csv")
y_test = pd.read_csv("./y_test.csv")

background = pd.read_parquet("shap_background.parquet")

def predict_proba_wrapper(X):
    if isinstance(X, (np.ndarray, list)):
        X = pd.DataFrame(X, columns=x_columns)
    return model.predict_proba(X)

explainer = shap.KernelExplainer(predict_proba_wrapper, background)

app = FastAPI()

class PredictRequest(BaseModel):
    features: list

class UpdateRequest(BaseModel):
    X_new: list[list] = X_test
    y_new: list = y_test
    X_val: list[list] = X_test
    y_val: list = y_test
    epochs: int = 5

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

@app.post("/update")
def update_model(req: UpdateRequest):
    global model

    start_time = time.time()

    X_new = X_test
    y_new = y_test
    X_val = X_test
    y_val = y_test

    model.save_model("temp_backup_model.json")

    probs_before = model.predict_proba(X_val)
    auc_before = roc_auc_score(y_val, probs_before[:, 1])
    prec, recall, _ = precision_recall_curve(y_val, probs_before[:, 1])
    pr_auc_before = auc(recall, prec)

    new_model = xgb.XGBClassifier(**model.get_params())
    new_model.fit(
        X_new,
        y_new,
        xgb_model=model.get_booster(),
        verbose=False,
    )

    probs_after = new_model.predict_proba(X_val)
    auc_after = roc_auc_score(y_val, probs_after[:, 1])
    prec, recall, _ = precision_recall_curve(y_val, probs_after[:, 1])
    pr_auc_after = auc(recall, prec)

    if auc_after < auc_before or pr_auc_after < pr_auc_before:
        model.load_model("temp_backup_model.json")
        status = "reverted"
    else:
        with open("xg_model.pkl", "wb") as f:
            pickle.dump(new_model, f)
        status = "updated"
        model = new_model

    elapsed_time = time.time() - start_time
    return {
        "status": status,
        "auc_before": auc_before,
        "auc_after": auc_after,
        "pr_auc_before": pr_auc_before,
        "pr_auc_after": pr_auc_after,
        "elapsed_time_seconds": round(elapsed_time, 4)
    }