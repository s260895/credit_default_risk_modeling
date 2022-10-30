data_clean = __import__('clean')
# data manipulation and storage
from matplotlib.scale import LogisticTransform
import pandas as pd
import numpy as np

# data preprocessing
# from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer

# models
# from sklearn.linear_model import LinearRegression
# from sklearn.linear_model import Lasso
# from sklearn.linear_model import Ridge
# import xgboost as xgb
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor


# model performance metrics
from sklearn.metrics import mean_squared_error

# saving model to file
import pickle

# mlflow for experiment tracking
import mlflow

# hyper-parameter optimization
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope

# sklearn pipeline creation
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# misc utilities
import copy

# intel sklearn optimization library
from sklearnex import patch_sklearn
patch_sklearn()

def initialize_regression_model(params,categoric_feat,numeric_feat,model_type='gradientboost'):

    numeric_pipeline = Pipeline(steps=[
        ('impute', SimpleImputer(strategy='mean')),
        ('scale', StandardScaler())
    ])
    categorical_pipeline = Pipeline(steps=[
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('one-hot', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])

    preprocessor_pipeline = ColumnTransformer(transformers=[
        ('numeric', numeric_pipeline, numeric_feat),
        ('categoric', categorical_pipeline, categoric_feat)
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
    y_train,
    X_val,
    y_val,
    categoric_feat,
    numeric_feat,
    model_type
):

    # define hyper-parameter search space
    search_space_gradient_boost = {
        # 'n_estimators':hp.choice('n_estimators',np.arange(10,101,1)),
        'learning_rate':hp.loguniform('learning_rate',-3,1),
        # 'min_samples_split':hp.loguniform('min_child_weight',-4,0),
        # 'max_depth':scope.int(hp.quniform('max_depth',5,100,5)),        
        'random_state':42
    }
    search_space_random_forest = {
        'max_depth':hp.choice('max_depth',np.arange(10,100,10)),
        'random_state': 42
    }
    search_space_ada_boost = {
        'learning_rate':hp.loguniform('learning_rate',-3,1),    
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
            pipe = initialize_regression_model(params=params,categoric_feat=categoric_feat,numeric_feat=numeric_feat,model_type=model_type)
            pipe.fit(X_train,y_train)
            y_pred = pipe.predict(X_val)
            rmse = mean_squared_error(y_val,y_pred,squared=False)
            mlflow.log_metric('validation_rmse',rmse)
            # mlflow.log_artifact(scaler_path,artifact_path="preprocessor")
            # mlflow.log_artifact(vectorizer_path,artifact_path="preprocessor")
            # mlflow.xgboost.log_model(xgb_model,artifact_path="models_mlflow")

        return {'loss':rmse,'status':STATUS_OK}


    # Perform hyper-parameter optimization
    best_result = fmin(
        fn = objective,
        space = search_space,
        algo = tpe.suggest,
        max_evals=5,
        trials = Trials()
    )   

    return best_result

def main(
    train_path = 'data/train_data.csv',
    val_path = 'na',
    tracking_uri = 'sqlite:///mlflow.db',
    experiment = 'ada-boost-amex-default-experiment',
    target_feat = 'target',
    train_feat = ['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68','customer_ID', 'S_2', 'P_2', 'D_39', 'B_1', 'B_2', 'R_1', 'S_3', 'D_41', 'B_3', 'D_42', 'D_43', 'D_44', 'B_4', 'D_45', 'B_5', 'R_2', 'D_46', 'D_47', 'D_48', 'D_49', 'B_6', 'B_7', 'B_8', 'D_50', 'D_51', 'B_9', 'R_3', 'D_52', 'P_3', 'B_10', 'D_53', 'S_5', 'B_11', 'S_6', 'D_54', 'R_4', 'S_7', 'B_12', 'S_8', 'D_55', 'D_56', 'B_13', 'R_5', 'D_58', 'S_9', 'B_14', 'D_59', 'D_60', 'D_61', 'B_15', 'S_11', 'D_62', 'D_65', 'B_16', 'B_17', 'B_18', 'B_19', 'B_20', 'S_12', 'R_6', 'S_13', 'B_21', 'D_69', 'B_22', 'D_70', 'D_71', 'D_72', 'S_15', 'B_23', 'D_73', 'P_4', 'D_74', 'D_75', 'D_76', 'B_24', 'R_7', 'D_77', 'B_25', 'B_26', 'D_78', 'D_79', 'R_8', 'R_9', 'S_16', 'D_80', 'R_10', 'R_11', 'B_27', 'D_81', 'D_82', 'S_17', 'R_12', 'B_28', 'R_13', 'D_83', 'R_14', 'R_15', 'D_84', 'R_16', 'B_29', 'S_18', 'D_86', 'D_87', 'R_17', 'R_18', 'D_88', 'B_31', 'S_19', 'R_19', 'B_32', 'S_20', 'R_20', 'R_21', 'B_33', 'D_89', 'R_22', 'R_23', 'D_91', 'D_92', 'D_93', 'D_94', 'R_24', 'R_25', 'D_96', 'S_22', 'S_23', 'S_24', 'S_25', 'S_26', 'D_102', 'D_103', 'D_104', 'D_105', 'D_106', 'D_107', 'B_36', 'B_37', 'R_26', 'R_27', 'D_108', 'D_109', 'D_110', 'D_111', 'B_39', 'D_112', 'B_40', 'S_27', 'D_113', 'D_115', 'D_118', 'D_119', 'D_121', 'D_122', 'D_123', 'D_124', 'D_125', 'D_127', 'D_128', 'D_129', 'B_41', 'B_42', 'D_130', 'D_131', 'D_132', 'D_133', 'R_28', 'D_134', 'D_135', 'D_136', 'D_137', 'D_138', 'D_139', 'D_140', 'D_141', 'D_142', 'D_143', 'D_144', 'D_145'],
    categoric_feat = ['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68'],
    numeric_feat = ['customer_ID', 'S_2', 'P_2', 'D_39', 'B_1', 'B_2', 'R_1', 'S_3', 'D_41', 'B_3', 'D_42', 'D_43', 'D_44', 'B_4', 'D_45', 'B_5', 'R_2', 'D_46', 'D_47', 'D_48', 'D_49', 'B_6', 'B_7', 'B_8', 'D_50', 'D_51', 'B_9', 'R_3', 'D_52', 'P_3', 'B_10', 'D_53', 'S_5', 'B_11', 'S_6', 'D_54', 'R_4', 'S_7', 'B_12', 'S_8', 'D_55', 'D_56', 'B_13', 'R_5', 'D_58', 'S_9', 'B_14', 'D_59', 'D_60', 'D_61', 'B_15', 'S_11', 'D_62', 'D_65', 'B_16', 'B_17', 'B_18', 'B_19', 'B_20', 'S_12', 'R_6', 'S_13', 'B_21', 'D_69', 'B_22', 'D_70', 'D_71', 'D_72', 'S_15', 'B_23', 'D_73', 'P_4', 'D_74', 'D_75', 'D_76', 'B_24', 'R_7', 'D_77', 'B_25', 'B_26', 'D_78', 'D_79', 'R_8', 'R_9', 'S_16', 'D_80', 'R_10', 'R_11', 'B_27', 'D_81', 'D_82', 'S_17', 'R_12', 'B_28', 'R_13', 'D_83', 'R_14', 'R_15', 'D_84', 'R_16', 'B_29', 'S_18', 'D_86', 'D_87', 'R_17', 'R_18', 'D_88', 'B_31', 'S_19', 'R_19', 'B_32', 'S_20', 'R_20', 'R_21', 'B_33', 'D_89', 'R_22', 'R_23', 'D_91', 'D_92', 'D_93', 'D_94', 'R_24', 'R_25', 'D_96', 'S_22', 'S_23', 'S_24', 'S_25', 'S_26', 'D_102', 'D_103', 'D_104', 'D_105', 'D_106', 'D_107', 'B_36', 'B_37', 'R_26', 'R_27', 'D_108', 'D_109', 'D_110', 'D_111', 'B_39', 'D_112', 'B_40', 'S_27', 'D_113', 'D_115', 'D_118', 'D_119', 'D_121', 'D_122', 'D_123', 'D_124', 'D_125', 'D_127', 'D_128', 'D_129', 'B_41', 'B_42', 'D_130', 'D_131', 'D_132', 'D_133', 'R_28', 'D_134', 'D_135', 'D_136', 'D_137', 'D_138', 'D_139', 'D_140', 'D_141', 'D_142', 'D_143', 'D_144', 'D_145', 'target']
):
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    train_path = '/media/ub/sec/train_data.csv'
    val_path = '/media/ub/sec/train_labels.csv'
    df_train = pd.read_parquet(train_path)
    df_val = pd.read_parquet(val_path)
    X_train,y_train = data_clean.cleaned_train_and_target(df_train,train_feat,target_feat)
    X_val, y_val = data_clean.cleaned_train_and_target(df_val,train_feat,target_feat)
    best_result = hyperparameter_optimizer(X_train,y_train,X_val,y_val,categoric_feat,numeric_feat)
    # train_best_model(train,valid,y_val,dv,scaler)

# main()