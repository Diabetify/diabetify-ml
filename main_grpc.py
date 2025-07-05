import pickle
import shap
import numpy as np
import pandas as pd
import time
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
    
    logger.info("Successfully loaded model and data files")
except Exception as e:
    logger.error(f"Error loading model or data files: {e}")
    sys.exit(1)

# Create SHAP explainer
explainer = shap.TreeExplainer(model)
logger.info("SHAP explainer initialized")

def make_prediction(features):
    start_time = time.time()

    X = pd.DataFrame([features], columns=x_columns)
    prediction = model.predict_proba(X)[0]
    try:
        shap_values_single = explainer.shap_values(X)
        abs_shap_single = np.abs(shap_values_single[0])
        abs_shap_single /= abs_shap_single.sum()

        explanation_items = [
            (
                feature,
                {
                    "shap": float(shap),
                    "contribution": float(contribution),
                    "impact": 1 if shap > 0 else 0
                }
            )
            for feature, shap, contribution in zip(x_columns, shap_values_single[0], abs_shap_single)
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

class PredictionServicer(prediction_pb2_grpc.PredictionServiceServicer):
    def Predict(self, request, context):
        logger.info(f"gRPC predict request received with {len(request.features)} features")
        features = list(request.features)
        
        try:
            result = make_prediction(features)
            
            explanation_pb = {}
            for feature_name, explanation_data in result["explanation"].items():
                feature_explanation = prediction_pb2.FeatureExplanation(
                    shap=explanation_data["shap"],
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