import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import time

from matplotlib import style
style.use("seaborn-v0_8")
import plotly.express as px

import warnings
warnings.filterwarnings('ignore')

st.write("## Calories burned Prediction")
# st.image("img.png" , use_column_width=True)
st.write("In this WebApp you will be able to observe your predicted calories burned in your body.Only thing you have to do is pass your parameters such as `Age` , `Gender` , `BMI` , etc into this WebApp and then you will be able to see the predicted value of kilocalories that burned in your body.")


st.sidebar.header("User Input Parameters : ")

def user_input_features():
    global Age , BMI , Duration , Heart_Rate , Body_Temp
    Age = st.sidebar.slider("Age : " , 10 , 100 , 30)
    BMI = st.sidebar.slider("BMI : " , 15 , 40 , 20)
    Duration = st.sidebar.slider("Duration (min) : " , 0 , 35 , 15)
    Heart_Rate = st.sidebar.slider("Heart Rate : " , 60 , 130 , 80)
    Body_Temp = st.sidebar.slider("Body Temperature (C) : " , 36 , 42 , 38)
    Gender_button = st.sidebar.radio("Gender : ", ("Male" , "Female"))

    if Gender_button == "Male":
        Gender_male = 1
    else:
        Gender_male = 0

    data = {
    "Age" : Age,
    "BMI" : BMI,
    "Duration" : Duration,
    "Heart_Rate" : Heart_Rate,
    "Body_Temp" : Body_Temp,
    "Gender_male" : ["Male" if Gender_button == "Male" else "Female"]
    }

    data_model = {
    "Age" : Age,
    "BMI" : BMI,
    "Duration" : Duration,
    "Heart_Rate" : Heart_Rate,
    "Body_Temp" : Body_Temp,
    "Gender_male" : Gender_male
    }

    features = pd.DataFrame(data_model, index=[0])
    data = pd.DataFrame(data, index=[0])
    return features , data

df , data = user_input_features()

st.write("---")
st.header("Your Parameters : ")
latest_iteration = st.empty()
bar = st.progress(0)
for i in range(100):
  # Update the progress bar with each iteration
  bar.progress(i + 1)
  time.sleep(0.01)
st.write(data)

calories = pd.read_csv("calories.csv")
exercise = pd.read_csv("exercise.csv")

exercise_df = exercise.merge(calories , on = "User_ID")
# st.write(exercise_df.head())
exercise_df.drop(columns = "User_ID" , inplace = True)

exercise_train_data , exercise_test_data = train_test_split(exercise_df , test_size = 0.2 , random_state = 1)

for data in [exercise_train_data , exercise_test_data]:         # adding BMI column to both training and test sets
    data["BMI"] = data["Weight"] / ((data["Height"] / 100) ** 2)
    data["BMI"] = round(data["BMI"] , 2)

exercise_train_data = exercise_train_data[["Gender" , "Age" , "BMI" , "Duration" , "Heart_Rate" , "Body_Temp" , "Calories"]]
exercise_test_data = exercise_test_data[["Gender" , "Age" , "BMI"  , "Duration" , "Heart_Rate" , "Body_Temp" , "Calories"]]
exercise_train_data = pd.get_dummies(exercise_train_data, drop_first = True)
exercise_test_data = pd.get_dummies(exercise_test_data, drop_first = True)

X_train = exercise_train_data.drop("Calories" , axis = 1)
y_train = exercise_train_data["Calories"]

X_test = exercise_test_data.drop("Calories" , axis = 1)
y_test = exercise_test_data["Calories"]

random_reg = RandomForestRegressor(n_estimators = 1000 , max_features = 3 , max_depth = 6)
random_reg.fit(X_train , y_train)
random_reg_prediction = random_reg.predict(X_test)

#st.write("RandomForest Mean Absolute Error(MAE) : " , round(metrics.mean_absolute_error(y_test , random_reg_prediction) , 2))
#st.write("RandomForest Mean Squared Error(MSE) : " , round(metrics.mean_squared_error(y_test , random_reg_prediction) , 2))
#st.write("RandomForest Root Mean Squared Error(RMSE) : " , round(np.sqrt(metrics.mean_squared_error(y_test , random_reg_prediction)) , 2))
prediction = random_reg.predict(df)

st.write("---")
st.header("Prediction : ")
latest_iteration = st.empty()
bar = st.progress(0)
for i in range(100):
  # Update the progress bar with each iteration
  bar.progress(i + 1)
  time.sleep(0.01)

st.write(round(prediction[0] , 2) , "   **kilocalories**")

st.write("---")
st.header("Similar Results : ")
latest_iteration = st.empty()
bar = st.progress(0)
for i in range(100):
  # Update the progress bar with each iteration
  bar.progress(i + 1)
  time.sleep(0.01)

range = [prediction[0] - 10 , prediction[0] + 10]
ds = exercise_df[(exercise_df["Calories"] >= range[0]) & (exercise_df["Calories"] <= range[-1])]
st.write(ds.sample(5))

st.write("---")
st.header("General Information : ")

boolean_age = (exercise_df["Age"] < Age).tolist()
boolean_duration = (exercise_df["Duration"] < Duration).tolist()
boolean_body_temp = (exercise_df["Body_Temp"] < Body_Temp).tolist()
boolean_heart_rate= (exercise_df["Heart_Rate"] < Heart_Rate).tolist()

st.write("You are older than " , round(sum(boolean_age) / len(boolean_age) , 2) * 100 , "% of other people.")
st.write("Your had higher exercise duration than " , round(sum(boolean_duration) / len(boolean_duration) , 2) * 100 , "% of other people.")
st.write("You had more heart rate than " , round(sum(boolean_heart_rate) / len(boolean_heart_rate) , 2) * 100 , "% of other people during exercise.")
st.write("You had higher body temperature  than " , round(sum(boolean_body_temp) / len(boolean_body_temp) , 2) * 100 , "% of other people during exercise.")
