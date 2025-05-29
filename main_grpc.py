import pickle
import shap
import numpy as np
import pandas as pd
import time
import xgboost as xgb
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import grpc
import concurrent.futures
import datetime
import logging
import os
from grpc_reflection.v1alpha import reflection

import prediction_pb2
import prediction_pb2_grpc

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load the model and data
try:
    with open("xg_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open('x_columns.pkl', 'rb') as f:
        x_columns = pickle.load(f)

    X_test = pd.read_csv("./X_test.csv")
    y_test = pd.read_csv("./y_test.csv")

    background = pd.read_parquet("shap_background.parquet")
    
    logger.info("Successfully loaded model and data files")
except Exception as e:
    logger.error(f"Error loading model or data files: {e}")
    sys.exit(1)

def predict_proba_wrapper(X):
    if isinstance(X, (np.ndarray, list)):
        X = pd.DataFrame(X, columns=x_columns)
    return model.predict_proba(X)

# Create SHAP explainer
explainer = shap.KernelExplainer(predict_proba_wrapper, background)
logger.info("SHAP explainer initialized")

def make_prediction(features):
    start_time = time.time()

    X = np.array([features])

    if isinstance(X, (np.ndarray, list)):
        X = pd.DataFrame(X, columns=x_columns)
    prediction = model.predict_proba(X)[0]

    sample_np = X.to_numpy().reshape(1, -1)
    try:
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
    except Exception as e:
        logger.error(f"Error generating SHAP explanations: {e}")
        explanation_dict_sorted = {}

    elapsed_time = time.time() - start_time

    return {
        "prediction": float(prediction[1]),
        "explanation": explanation_dict_sorted,
        "elapsed_time_seconds": round(elapsed_time, 4)
    }

def update_model_function(X_new, y_new, X_val, y_val, epochs=5):
    global model

    start_time = time.time()

    X_new_df = pd.DataFrame(X_new, columns=x_columns)
    y_new_series = pd.Series(y_new)
    X_val_df = pd.DataFrame(X_val, columns=x_columns)
    y_val_series = pd.Series(y_val)

    model.save_model("temp_backup_model.json")

    probs_before = model.predict_proba(X_val_df)
    auc_before = roc_auc_score(y_val_series, probs_before[:, 1])
    prec, recall, _ = precision_recall_curve(y_val_series, probs_before[:, 1])
    pr_auc_before = auc(recall, prec)

    new_model = xgb.XGBClassifier(**model.get_params())
    new_model.fit(
        X_new_df,
        y_new_series,
        xgb_model=model.get_booster(),
        verbose=False,
    )

    probs_after = new_model.predict_proba(X_val_df)
    auc_after = roc_auc_score(y_val_series, probs_after[:, 1])
    prec, recall, _ = precision_recall_curve(y_val_series, probs_after[:, 1])
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
        "auc_before": float(auc_before),
        "auc_after": float(auc_after),
        "pr_auc_before": float(pr_auc_before),
        "pr_auc_after": float(pr_auc_after),
        "elapsed_time_seconds": round(elapsed_time, 4)
    }

class PredictionServicer(prediction_pb2_grpc.PredictionServiceServicer):
    def Predict(self, request, context):
        logger.info(f"gRPC predict request received with {len(request.features)} features")
        features = list(request.features)
        
        try:
            result = make_prediction(features)
            
            explanation_pb = {}
            for feature_name, explanation_data in result["explanation"].items():
                feature_explanation = prediction_pb2.FeatureExplanation(
                    contribution=explanation_data["contribution"],
                    impact=explanation_data["impact"]
                )
                explanation_pb[feature_name] = feature_explanation
            
            return prediction_pb2.PredictionResponse(
                prediction=result["prediction"],
                explanation=explanation_pb,
                elapsed_time=result["elapsed_time_seconds"],
                timestamp=datetime.datetime.now().isoformat()
            )
        except Exception as e:
            logger.error(f"Error in Predict: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Prediction failed: {str(e)}")
            return prediction_pb2.PredictionResponse()
        
    def UpdateModel(self, request, context):
        logger.info("gRPC update model request received")
        
        try:
            X_new = [[float(val) for val in feature.values] for feature in request.X_new]
            y_new = list(request.y_new)
            X_val = [[float(val) for val in feature.values] for feature in request.X_val]
            y_val = list(request.y_val)
            epochs = request.epochs
            
            result = update_model_function(X_new, y_new, X_val, y_val, epochs)
            
            return prediction_pb2.UpdateModelResponse(
                status=result["status"],
                auc_before=result["auc_before"],
                auc_after=result["auc_after"],
                pr_auc_before=result["pr_auc_before"],
                pr_auc_after=result["pr_auc_after"],
                elapsed_time=result["elapsed_time_seconds"],
                timestamp=datetime.datetime.now().isoformat()
            )
        except Exception as e:
            logger.error(f"Error in UpdateModel: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Model update failed: {str(e)}")
            return prediction_pb2.UpdateModelResponse()
        
    def HealthCheck(self, request, context):
        return prediction_pb2.HealthCheckResponse(
            status="healthy",
            timestamp=datetime.datetime.now().isoformat()
        )

def serve():
    port = os.getenv("GRPC_PORT", "50051")
    
    server = grpc.server(concurrent.futures.ThreadPoolExecutor(max_workers=10))
    prediction_pb2_grpc.add_PredictionServiceServicer_to_server(
        PredictionServicer(), server
    )
    
    service_names = (
        prediction_pb2.DESCRIPTOR.services_by_name['PredictionService'].full_name,
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(service_names, server)
    
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    logger.info(f"gRPC server running on port {port}")
    
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down server")
        server.stop(0)

if __name__ == "__main__":
    serve()