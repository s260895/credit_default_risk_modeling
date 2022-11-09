from flask import Flask, request, jsonify
import mlflow
import pandas as pd
data_clean = __import__('clean')

tracking_uri = '127.0.0.1:5000',
exp_name = 'gradientboost-amex-default-experiment'
mlflow.set_tracking_uri(tracking_uri)
# mlflow.set_experiment(exp_name)

# define function for prediction using saved model
def predict(raw_data,train_feat,target_feat):
    # Load model from mlflow as a PyFuncModel.
    logged_model = 'runs:/9a28033fdd914d0eb4ae48fb11fde866/model'
    loaded_model = mlflow.pyfunc.load_model(logged_model)
    X,y = data_clean.cleaned_train_and_target(df=pd.DataFrame(raw_data),train_feat=train_feat,target_feat=target_feat,inference=True)
    # X = loaded_model.transform(X)
    y_pred = loaded_model.predict(X)
    # rmse = mean_squared_error(y,y_pred,squared=False)
    return y_pred



app = Flask('duration-prediction')

@app.route('/predict',methods=['POST'])
def prediction_endpoint():
    data = request.get_json()
    # print(data)
    prediction = predict.predict(data,train_feat=['PU_DO_pair','trip_distance','total_amount','passenger_count'],target_feat='duration')
    result = {
        'duration': prediction[0]
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=9696)
    