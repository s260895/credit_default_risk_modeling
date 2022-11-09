# credit_default_risk_modeling
MVP for predicting probability of default for a loan applicant

Step 1: Spin up MLFlow Production Server (Model Lifecycle Manager)
mlflow server --backend-store-uri=sqlite:///mlflow.db --default-artifact-root=./mlruns

Step 2: Spin up Flask Server to test if Model Server works properly
python predict.py

Step 3: Spin up gunicorn Server to deploy Model Server in production environment
gunicorn --bind=0.0.0.0:9696 predict:prediction_endpoint