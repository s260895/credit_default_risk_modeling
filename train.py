data_clean = __import__('clean')
from matplotlib.scale import LogisticTransform
import pandas as pd
import numpy as np

# data preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer

# models
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor

# model performance metrics
from sklearn.metrics import mean_squared_error
amex_metric = __import__('amex_metric')

# saving model to file
import pickle

# mlflow for experiment tracking
import mlflow

# hyper-parameter optimization
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope

# sklearn pipeline creation
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# misc utilities
import copy

# intel sklearn optimization library
from sklearnex import patch_sklearn
patch_sklearn()

def initialize_regression_model(params,categoric_feat,numeric_feat,model_type):

    numeric_pipeline = Pipeline(steps=[
        ('impute', SimpleImputer(strategy='mean')),
        ('scale', StandardScaler())
    ])
    categorical_pipeline = Pipeline(steps=[
        ('impute', SimpleImputer(strategy='most_frequent',missing_values=pd.NA)),
        ('one-hot', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])

    preprocessor_pipeline = ColumnTransformer(transformers=[
        ('numeric', numeric_pipeline, numeric_feat)
        # ('categoric', categorical_pipeline, categoric_feat)
    ])

    if model_type == 'gradientboost':
        regressor = GradientBoostingRegressor(**params)
    if model_type == 'randomforest':
        regressor = RandomForestRegressor(**params)
    if model_type == 'adaboost':
        regressor = AdaBoostRegressor(**params)

    else:
        regressor = None

    regression_model = Pipeline(steps=[
        ('preprocess', preprocessor_pipeline),
        ('model', regressor)
    ])

    return regression_model

def hyperparameter_optimizer(
    X_train,
    X_val,
    y_train,
    y_val,
    numeric_feat,
    # categoric_feat,
    model_type
):

    # define hyper-parameter search space
    search_space_gradient_boost = {
        # 'n_estimators':hp.choice('n_estimators',np.arange(10,101,1)),
        'learning_rate':hp.loguniform('learning_rate',-5,1),
        # 'min_samples_split':hp.loguniform('min_child_weight',-4,0),
        # 'max_depth':scope.int(hp.quniform('max_depth',5,100,5)),        
        'random_state':42
    }
    search_space_random_forest = {
        'max_depth':hp.choice('max_depth',np.arange(10,1000,10)),
        'random_state': 42
    }
    search_space_ada_boost = {
        'learning_rate':hp.loguniform('learning_rate',-5,1),    
        'random_state':42
    }

    if model_type == 'gradientboost':
        search_space = search_space_gradient_boost
    if model_type == 'adaboost':
        search_space = search_space_ada_boost
    if model_type == 'randomforest':
        search_space = search_space_random_forest
    

    # define objective function
    def objective(params):
        
        with mlflow.start_run():
            mlflow.set_tag('model',model_type)
            # mlflow.log_params(params)
            mlflow.sklearn.autolog()
            pipe = initialize_regression_model(params=params,categoric_feat=None,numeric_feat=numeric_feat,model_type=model_type)
            pipe.fit(X_train,y_train)
            y_pred = pipe.predict(X_val)
            custom_amex_metric = amex_metric.amex_metric(pd.DataFrame(y_val.values,columns=['target']),pd.DataFrame(y_pred,columns=['prediction']))
            mlflow.log_metric('custom_amex_metric',custom_amex_metric)
            # mlflow.log_artifact(scaler_path,artifact_path="preprocessor")
            # mlflow.log_artifact(vectorizer_path,artifact_path="preprocessor")
            # mlflow.xgboost.log_model(xgb_model,artifact_path="models_mlflow")

        return {'loss':custom_amex_metric,'status':STATUS_OK}


    # Perform hyper-parameter optimization
    best_result = fmin(
        fn = objective,
        space = search_space,
        algo = tpe.suggest,
        max_evals=100,
        trials = Trials()
    )   

    return best_result
    

def main(
    train_path = 'amex_data.csv',
    tracking_uri = 'sqlite:///mlflow.db',
    experiment = 'adaboost-amex-default-experiment',
    target_feat = 'target',
    model_type = 'adaboost'
):
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    df = pd.read_csv('amex_data.csv',index_col=0)
    X_train,X_val,y_train,y_val = data_clean.cleaned_train_and_target(df,target_feat)
    categoric_feat = ['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_63', 'D_64', 'D_68']
    numeric_feat = [feat for feat in df.columns if feat not in categoric_feat and feat not in ['customer_ID','target','S_2']]
    best_result = hyperparameter_optimizer(X_train,X_val,y_train,y_val,numeric_feat,model_type=model_type)

# Run 3 experiments with three different models: adaboost, gradientboost and randomforest
main(experiment='adaboost-amex-default-experiment',model_type='adaboost')
main(experiment='randomforest-amex-default-experiment',model_type='randomforest')
main(experiment='gradientboost-amex-default-experiment',model_type='gradientboost')