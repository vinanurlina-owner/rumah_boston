import json
import pickle
import os
import time
import logging
from functools import wraps

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from werkzeug.security import check_password_hash

from prometheus_client import (
    Counter,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

# ============================================================
# Logging Configuration
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

# ============================================================
# Flask Application
# ============================================================

app = Flask(__name__)

# ============================================================
# Secret Key / Session
# ============================================================

# WAJIB di-set sebagai environment variable APP_SECRET.
APP_SECRET = os.getenv("APP_SECRET")
if not APP_SECRET:
    raise RuntimeError("APP_SECRET environment variable is not set")

app.config["SECRET_KEY"] = APP_SECRET
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = True

# ============================================================
# Application Configuration
# ============================================================

MODEL_FILE = os.getenv("MODEL_FILE", "regmodel.pkl")
SCALER_FILE = os.getenv("SCALER_FILE", "scaling.pkl")

PORT = int(os.getenv("PORT", "5000"))
FLASK_ENV = os.getenv("FLASK_ENV", "production")

# ============================================================
# Login Configuration
# ============================================================

# Credentials login disimpan sebagai environment variables.
# ADMIN_PASSWORD_HASH harus berupa Werkzeug password hash.
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD_HASH = os.getenv("ADMIN_PASSWORD_HASH")

if not ADMIN_PASSWORD_HASH:
    raise RuntimeError("ADMIN_PASSWORD_HASH environment variable is not set")

# ============================================================
# Prometheus Metrics
# ============================================================

HTTP_REQUEST_TOTAL = Counter(
    "http_requests_total",
    "Total HTTP Requests",
    ["method", "endpoint", "status"],
)

PREDICTION_COUNT = Counter(
    "prediction_count_total",
    "Total Prediction Requests",
)

PREDICTION_ERRORS = Counter(
    "prediction_errors_total",
    "Total Prediction Errors",
)

PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Prediction Latency",
)

HTTP_RESPONSE_TIME = Histogram(
    "http_response_time_seconds",
    "HTTP Response Time",
    ["endpoint"],
)

# ============================================================
# Load Machine Learning Model
# ============================================================

logger.info("==============================================")
logger.info(" Flask Machine Learning Application")
logger.info("==============================================")
logger.info(f"Environment : {FLASK_ENV}")
logger.info(f"Port        : {PORT}")
logger.info(f"Model File  : {MODEL_FILE}")
logger.info(f"Scaler File : {SCALER_FILE}")

try:
    with open(SCALER_FILE, "rb") as f:
        scalar = pickle.load(f)

    with open(MODEL_FILE, "rb") as f:
        regmodel = pickle.load(f)

    MODEL_READY = True
    logger.info("Machine Learning model loaded successfully.")

except Exception:
    MODEL_READY = False
    scalar = None
    regmodel = None
    logger.exception("Failed to load Machine Learning model.")

# ============================================================
# Monitoring Decorator
# ============================================================

def monitor_endpoint(endpoint_name):
    """
    Mengukur jumlah request, status HTTP, response time,
    dan menulis informasi request ke log.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            status_code = "200"

            try:
                response = func(*args, **kwargs)

                # Flask route dapat mengembalikan:
                # Response, (Response, status), atau (body, status, headers)
                if isinstance(response, tuple):
                    if len(response) >= 2:
                        status_code = str(response[1])
                else:
                    status_code = str(getattr(response, "status_code", 200))

                return response

            except Exception:
                status_code = "500"
                raise

            finally:
                duration = time.perf_counter() - start_time

                HTTP_REQUEST_TOTAL.labels(
                    method=request.method,
                    endpoint=endpoint_name,
                    status=status_code,
                ).inc()

                HTTP_RESPONSE_TIME.labels(
                    endpoint=endpoint_name,
                ).observe(duration)

                logger.info(
                    "%s %s %s %.4f sec",
                    request.method,
                    endpoint_name,
                    status_code,
                    duration,
                )

        return wrapper

    return decorator

# ============================================================
# Login
# ============================================================

def login_required(view):
    @wraps(view)
    def wrapped_view(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login"))
        return view(*args, **kwargs)

    return wrapped_view


@app.route("/login", methods=["GET", "POST"])
@monitor_endpoint("/login")
def login():
    if session.get("logged_in"):
        return redirect(url_for("home"))

    error = None

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        try:
            valid_password = check_password_hash(
                ADMIN_PASSWORD_HASH,
                password,
            )
        except Exception:
            valid_password = False
            logger.exception("Password hash validation failed.")

        if username == ADMIN_USERNAME and valid_password:
            session.clear()
            session["logged_in"] = True
            session["username"] = username

            logger.info("Successful login for user: %s", username)

            return redirect(url_for("home"))

        logger.warning(
            "Failed login attempt for username: %s",
            username,
        )
        error = "Username atau password salah."

    return render_template("login.html", error=error)


# ============================================================
# Logout
# ============================================================

@app.route("/logout")
@monitor_endpoint("/logout")
def logout():
    username = session.get("username")

    session.clear()

    logger.info("Logout for user: %s", username)

    return redirect(url_for("login"))


# ============================================================
# Home
# ============================================================

@app.route("/")
@login_required
@monitor_endpoint("/")
def home():
    return render_template(
        "home.html",
        username=session.get("username"),
    )


# ============================================================
# Prediction API - JSON
# ============================================================

@app.route("/predict_api", methods=["POST"])
@login_required
@monitor_endpoint("/predict_api")
def predict_api():
    if not MODEL_READY:
        logger.error("Prediction requested but model is unavailable.")
        return jsonify(
            status="error",
            message="Machine Learning model is not available.",
        ), 503

    start_time = time.perf_counter()

    try:
        if not request.is_json:
            raise ValueError("Request body must be JSON.")

        payload = request.get_json(silent=True)

        if payload is None:
            raise ValueError("JSON body is empty.")

        if "data" not in payload:
            raise ValueError("Missing key 'data'.")

        data = payload["data"]

        if not isinstance(data, dict):
            raise ValueError("'data' must be an object.")

        values = list(data.values())

        if not values:
            raise ValueError("'data' must contain at least one feature.")

        features = np.array(
            values,
            dtype=float,
        ).reshape(1, -1)

        # Validasi jumlah feature jika scaler menyediakan informasi ini.
        expected_features = getattr(scalar, "n_features_in_", None)
        if (
            expected_features is not None
            and features.shape[1] != expected_features
        ):
            raise ValueError(
                f"Expected {expected_features} features, "
                f"but received {features.shape[1]}."
            )

        scaled = scalar.transform(features)
        prediction = regmodel.predict(scaled)
        prediction_value = float(prediction[0])

        PREDICTION_COUNT.inc()

        logger.info(
            "Prediction API Success : %s",
            prediction_value,
        )

        return jsonify(
            prediction=prediction_value,
            status="success",
        ), 200

    except Exception as e:
        PREDICTION_ERRORS.inc()

        logger.exception("Prediction API failed.")

        return jsonify(
            status="error",
            message=str(e),
        ), 400

    finally:
        PREDICTION_LATENCY.observe(
            time.perf_counter() - start_time,
        )


# ============================================================
# Prediction via HTML Form
# ============================================================

@app.route("/predict", methods=["POST"])
@login_required
@monitor_endpoint("/predict")
def predict():
    if not MODEL_READY:
        logger.error("Prediction requested but model is unavailable.")

        return render_template(
            "home.html",
            username=session.get("username"),
            prediction_text="Machine Learning model is unavailable.",
        ), 503

    start_time = time.perf_counter()

    try:
        data = [
            float(x)
            for x in request.form.values()
        ]

        expected_features = getattr(scalar, "n_features_in_", None)
        if (
            expected_features is not None
            and len(data) != expected_features
        ):
            raise ValueError(
                f"Expected {expected_features} features, "
                f"but received {len(data)}."
            )

        features = np.array(
            data,
            dtype=float,
        ).reshape(1, -1)

        final_input = scalar.transform(features)
        output = regmodel.predict(final_input)[0]
        prediction_value = float(output)

        PREDICTION_COUNT.inc()

        logger.info(
            "Prediction HTML Success : %s",
            prediction_value,
        )

        return render_template(
            "home.html",
            username=session.get("username"),
            prediction_text=(
                f"The House price prediction is "
                f"{prediction_value}"
            ),
        )

    except Exception as e:
        PREDICTION_ERRORS.inc()

        logger.exception("HTML prediction failed.")

        return render_template(
            "home.html",
            username=session.get("username"),
            prediction_text=f"Error : {str(e)}",
        ), 400

    finally:
        PREDICTION_LATENCY.observe(
            time.perf_counter() - start_time,
        )


# ============================================================
# Health Check
# ============================================================

@app.route("/health")
@monitor_endpoint("/health")
def health():
    if MODEL_READY:
        return jsonify(
            status="UP",
            application="Flask ML API",
            model_loaded=True,
            environment=FLASK_ENV,
        ), 200

    return jsonify(
        status="DOWN",
        application="Flask ML API",
        model_loaded=False,
        environment=FLASK_ENV,
    ), 503


# ============================================================
# CPU Stress Endpoint
# ============================================================

@app.route("/stress")
@monitor_endpoint("/stress")
def stress():
    logger.info("CPU Stress Started")

    start = time.perf_counter()

    for _ in range(200):
        matrix_a = np.random.rand(600, 600)
        matrix_b = np.random.rand(600, 600)
        np.dot(matrix_a, matrix_b)

    elapsed = round(
        time.perf_counter() - start,
        2,
    )

    logger.info(
        "CPU Stress Finished (%s sec)",
        elapsed,
    )

    return jsonify(
        status="completed",
        duration_seconds=elapsed,
    ), 200


# ============================================================
# Prometheus Metrics Endpoint
# ============================================================

@app.route("/metrics", methods=["GET"])
def metrics():
    """
    Endpoint yang akan di-scrape oleh Prometheus.
    Endpoint ini sengaja tidak menggunakan login_required
    agar Prometheus dapat melakukan scraping langsung.
    """
    return (
        generate_latest(),
        200,
        {
            "Content-Type": CONTENT_TYPE_LATEST,
        },
    )


# ============================================================
# Error Handlers
# ============================================================

@app.errorhandler(404)
def not_found(error):
    HTTP_REQUEST_TOTAL.labels(
        method=request.method,
        endpoint=request.path,
        status="404",
    ).inc()

    logger.warning(
        "404 Not Found : %s",
        request.path,
    )

    return jsonify(
        status="error",
        code=404,
        message="Endpoint not found.",
    ), 404


@app.errorhandler(405)
def method_not_allowed(error):
    HTTP_REQUEST_TOTAL.labels(
        method=request.method,
        endpoint=request.path,
        status="405",
    ).inc()

    logger.warning(
        "405 Method Not Allowed : %s",
        request.path,
    )

    return jsonify(
        status="error",
        code=405,
        message="Method not allowed.",
    ), 405


@app.errorhandler(500)
def internal_error(error):
    HTTP_REQUEST_TOTAL.labels(
        method=request.method,
        endpoint=request.path,
        status="500",
    ).inc()

    logger.exception("Internal Server Error")

    return jsonify(
        status="error",
        code=500,
        message="Internal server error.",
    ), 500


# ============================================================
# Startup Banner
# ============================================================

logger.info("==============================================")
logger.info(" Flask Machine Learning Application")
logger.info("==============================================")
logger.info("Environment     : %s", FLASK_ENV)
logger.info("Port            : %s", PORT)
logger.info("Model File      : %s", MODEL_FILE)
logger.info("Scaler File     : %s", SCALER_FILE)
logger.info("Model Ready     : %s", MODEL_READY)

if MODEL_READY:
    logger.info("Application started successfully.")
else:
    logger.warning(
        "Application started WITHOUT Machine Learning model."
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    logger.info("Starting Flask server...")

    app.run(
        host="0.0.0.0",
        port=PORT,
        debug=False,
        threaded=True,
    )
