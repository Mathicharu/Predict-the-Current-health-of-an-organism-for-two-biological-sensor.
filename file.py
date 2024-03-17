import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd

def linearregression():
    train_data= pd.read_csv(r'C:\Users\charu\Downloads\p1_train.csv',header=None)
    test_data=pd.read_csv(r"C:\Users\charu\Downloads\p1_test.csv",header=None)
    column=["Sensor1_measurement", "Sensor2_measurement", "Target_Value"]
    train_data.columns=column
    test_data.columns=column
    X_train = train_data[['Sensor1_measurement',"Sensor2_measurement"]]
    y_train = train_data['Target_Value']
    X_test = test_data[['Sensor1_measurement',"Sensor2_measurement"]]
    y_test = test_data['Target_Value']
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    regr = linear_model.LinearRegression()
    regr.fit(X_train,y_train)
    regr_pred = regr.predict(X_test)
    linear_mean_square = mean_squared_error(y_test,regr_pred)
    linear_mean_absolute=mean_absolute_error(y_test,regr_pred)

    train_data1 = pd.read_csv(r'C:\Users\charu\Downloads\p1_train.csv', header=None)
    train_data1.columns = ["Sensor1_measurement", "Sensor2_measurement", "Target_Value"]
    X_train1 = train_data1[["Sensor1_measurement", "Sensor2_measurement"]]
    y_train1 = train_data1["Target_Value"]
    svr_model = SVR(kernel='linear')  
    svr_model.fit(X_train1, y_train1)
    test_data1 = pd.read_csv(r'C:\Users\charu\Downloads\p1_test.csv', header=None)
    test_data1.columns = ["Sensor1_measurement", "Sensor2_measurement", "Target_Value"]
    X_test = test_data1[["Sensor1_measurement", "Sensor2_measurement"]]
    y_test = test_data1["Target_Value"]
    predictions = svr_model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)


    predicted_value = pd.DataFrame({ "Target_Value": y_test,"Predicted_linear": regr_pred,"Predicted_svr": predictions})
    print(predicted_value)
    Row=["Linear_regression_model","Support_vector_regression"]
    
    Mean_Square_Absolute=pd.DataFrame({"Mean Squared Error":[linear_mean_square,mse],"Mean Absolute Error":[linear_mean_absolute,mae] },index=Row)

    print(Mean_Square_Absolute)

linearregression()
