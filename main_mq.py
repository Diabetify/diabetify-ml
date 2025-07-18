import pickle
import shap
import numpy as np
import pandas as pd
import time
import json
import datetime
import os
import sys
import threading
import signal
import math
from typing import Dict, Any, List

import pika
from pika.exceptions import AMQPConnectionError

# --- Basic Configuration ---
RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://admin:password123@localhost:5672/")
MAX_RABBITMQ_RETRIES = 5
RABBITMQ_RETRY_DELAY = 5


class AsyncMLService:
    """
    An async-only ML service that handles RabbitMQ requests for predictions.
    """
    def __init__(self):
        self.model = None
        self.explainer = None
        self.x_columns = None
        self.is_running = threading.Event()

        # RabbitMQ attributes
        self.rabbit_connection = None
        self.rabbit_channel = None
        self.rabbit_consumer_thread = None
        self.prediction_request_queue = "ml.prediction.request"
        self.health_request_queue = "ml.health.request"

        # Debug counters
        self.messages_received = 0
        self.messages_processed = 0
        self.messages_failed = 0

        # Load ML model and related assets
        self._load_model_and_data()

    def _load_model_and_data(self):
        """Loads the ML model, columns, and initializes the SHAP explainer."""
        try:
            with open("xg_model.pkl", "rb") as f:
                self.model = pickle.load(f)
            with open('x_columns.pkl', 'rb') as f:
                self.x_columns = pickle.load(f)
            
            self.explainer = shap.TreeExplainer(self.model)
        except FileNotFoundError:
            sys.exit(1)
        except Exception:
            sys.exit(1)

    def safe_float(self, value):
        """Convert value to safe float, handling NaN and infinity"""
        try:
            val = float(value)
            if math.isnan(val) or math.isinf(val):
                return 0.0
            return val
        except (ValueError, TypeError):
            return 0.0

    def safe_string(self, value):
        """Convert value to safe string"""
        try:
            result = str(value).encode('ascii', 'ignore').decode('ascii')
            if not result:
                result = "unknown_feature"
            return result
        except:
            return "unknown_feature"

    def validate_float_for_json(self, value, field_name):
        """Validate that a float value is safe for JSON serialization"""
        if not isinstance(value, (int, float)):
            return False
        
        if math.isnan(value):
            return False
        
        if math.isinf(value):
            return False
        
        if abs(value) > 1e308:
            return False
        
        return True

    def make_prediction(self, features: List[float]) -> Dict[str, Any]:
        """
        Performs prediction and generates SHAP explanations for a given set of features.
        Returns response in JSON-compatible format for RabbitMQ.
        """
        start_time = time.time()
        
        if len(features) != len(self.x_columns):
            raise ValueError(f"Feature mismatch: Expected {len(self.x_columns)} features, but received {len(features)}.")

        try:
            converted_features = [self.safe_float(f) for f in features]
            X = pd.DataFrame([converted_features], columns=self.x_columns)
            prediction = self.model.predict_proba(X)[0]
            
            shap_values_single = self.explainer.shap_values(X)
            abs_shap_single = np.abs(shap_values_single[0])
            
            total_shap = abs_shap_single.sum()
            if total_shap > 0:
                abs_shap_single /= total_shap
            else:
                abs_shap_single = np.zeros_like(abs_shap_single)

            explanation_dict = {}
            for i, (feature, shap, contribution) in enumerate(zip(self.x_columns, shap_values_single[0], abs_shap_single)):
                safe_feature_name = self.safe_string(feature)
                safe_shap = self.safe_float(shap)
                safe_contribution = self.safe_float(contribution)
                safe_impact = 1 if safe_shap > 0 else 0
                
                if not self.validate_float_for_json(safe_shap, f"shap_{i}"):
                    safe_shap = 0.0
                
                if not self.validate_float_for_json(safe_contribution, f"contribution_{i}"):
                    safe_contribution = 0.0
                
                explanation_dict[safe_feature_name] = {
                    "shap": safe_shap,
                    "contribution": safe_contribution,
                    "impact": safe_impact,
                    "value": features[i]  # Include original feature value
                }

            elapsed_time = time.time() - start_time

            safe_prediction = self.safe_float(prediction[1])
            if not self.validate_float_for_json(safe_prediction, "prediction"):
                safe_prediction = 0.0

            safe_elapsed = self.safe_float(elapsed_time)
            if not self.validate_float_for_json(safe_elapsed, "elapsed_time"):
                safe_elapsed = 0.0

            return {
                "prediction": safe_prediction,
                "explanation": explanation_dict,
                "elapsed_time": safe_elapsed,
                "timestamp": datetime.datetime.now().isoformat()
            }
        except Exception:
            raise

    # --- RabbitMQ Methods ---
    def _setup_rabbitmq(self):
        """Sets up the RabbitMQ connection, channel, and queues with retry logic."""
        for attempt in range(MAX_RABBITMQ_RETRIES):
            try:
                self.rabbit_connection = pika.BlockingConnection(pika.URLParameters(RABBITMQ_URL))
                self.rabbit_channel = self.rabbit_connection.channel()
                
                queues = [
                    self.prediction_request_queue, 
                    "ml.prediction.response",
                    "ml.prediction.hybrid_response",
                    self.health_request_queue,
                    "ml.health.response"
                ]
                
                for queue in queues:
                    self.rabbit_channel.queue_declare(queue=queue, durable=True)
                
                self.rabbit_channel.basic_qos(prefetch_count=1)
                return True
            except AMQPConnectionError:
                time.sleep(RABBITMQ_RETRY_DELAY)
        
        return False

    def _handle_rabbitmq_message(self, ch, method, properties, body, handler_func):
        """Generic message handler that uses the 'reply_to' property for responses."""
        correlation_id = properties.correlation_id
        reply_to_queue = properties.reply_to
        
        self.messages_received += 1
        
        if not reply_to_queue:
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
            self.messages_failed += 1
            return
            
        actual_correlation_id = correlation_id # Default value

        try:
            request_data = json.loads(body.decode('utf-8'))
            
            if 'correlation_id' in request_data and request_data['correlation_id']:
                actual_correlation_id = request_data['correlation_id']
            
            response_data = handler_func(request_data)
            
            response_data['correlation_id'] = actual_correlation_id
            
            ch.basic_publish(
                exchange='',
                routing_key=reply_to_queue,
                body=json.dumps(response_data),
                properties=pika.BasicProperties(
                    correlation_id=actual_correlation_id,
                    content_type='application/json',
                    delivery_mode=2,
                )
            )
            ch.basic_ack(delivery_tag=method.delivery_tag)
            self.messages_processed += 1

        except (json.JSONDecodeError, Exception) as e:
            error_response = {
                "error": str(e),
                "correlation_id": actual_correlation_id,
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            try:
                ch.basic_publish(
                    exchange='',
                    routing_key=reply_to_queue,
                    body=json.dumps(error_response),
                    properties=pika.BasicProperties(correlation_id=error_response["correlation_id"])
                )
            except Exception:
                pass
            
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
            self.messages_failed += 1

    def _process_prediction_request(self, data: Dict) -> Dict:
        features = data.get('features', [])
        result = self.make_prediction(features)
        return result

    def _process_health_request(self, data: Dict) -> Dict:
        result = self.get_health_status()
        return result

    def _start_consuming(self):
        """The main loop for the RabbitMQ consumer thread."""
        self.rabbit_channel.basic_consume(
            queue=self.prediction_request_queue,
            on_message_callback=lambda c, m, p, b: self._handle_rabbitmq_message(c, m, p, b, self._process_prediction_request)
        )
        
        self.rabbit_channel.basic_consume(
            queue=self.health_request_queue,
            on_message_callback=lambda c, m, p, b: self._handle_rabbitmq_message(c, m, p, b, self._process_health_request)
        )

        try:
            self.rabbit_channel.start_consuming()
        except Exception:
            if self.is_running.is_set():
                pass

    # --- Health Check Logic ---
    def get_health_status(self) -> Dict[str, Any]:
        """Returns the health status of the service."""
        is_healthy = self.model is not None and self.explainer is not None
        status = "healthy" if is_healthy else "unhealthy"
        
        return {
            "status": status,
            "timestamp": datetime.datetime.now().isoformat(),
            "details": {
                "model_loaded": self.model is not None,
                "explainer_ready": self.explainer is not None,
                "rabbitmq_connected": self.rabbit_connection and self.rabbit_connection.is_open,
                "service_type": "async_only",
                "messages_received": self.messages_received,
                "messages_processed": self.messages_processed,
                "messages_failed": self.messages_failed
            }
        }

    # --- Service Start/Stop ---
    def start(self):
        """Starts RabbitMQ service and keeps the script alive."""
        self.is_running.set()
        
        if self._setup_rabbitmq():
            self.rabbit_consumer_thread = threading.Thread(target=self._start_consuming, name="RabbitMQConsumer")
            self.rabbit_consumer_thread.daemon = True
            self.rabbit_consumer_thread.start()
        else:
            sys.exit(1)

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        """Stops the RabbitMQ service gracefully."""
        if not self.is_running.is_set():
            return
        
        self.is_running.clear()

        if self.rabbit_channel and self.rabbit_channel.is_open:
            self.rabbit_channel.stop_consuming()
        if self.rabbit_connection and self.rabbit_connection.is_open:
            self.rabbit_connection.close()
        
        if self.rabbit_consumer_thread and self.rabbit_consumer_thread.is_alive():
             self.rabbit_consumer_thread.join()


def main():
    """Initializes and starts the async-only service."""
    service = AsyncMLService()
    
    def signal_handler(signum, frame):
        service.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    service.start()

if __name__ == '__main__':
    main()