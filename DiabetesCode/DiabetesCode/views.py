from django.shortcuts import render
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def index(request):
    return render(request, 'index.html')

def predict(request):
    return render(request, 'predict.html')

def result(request):

    data = pd.read_csv("C:/Users/dhruv/Downloads/Diabetes Prediction System (2)/Diabetes Prediction System/Dataset/main_data.csv")

    x = data.drop('Outcome', axis=1)
    y = data['Outcome']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

    model = LogisticRegression()
    model.fit(x_train, y_train)

    val1 = float(request.GET['a1'])
    val2 = float(request.GET['a2'])
    val3 = float(request.GET['a3'])
    val4 = float(request.GET['a4'])
    val5 = float(request.GET['a5'])
    val6 = float(request.GET['a6'])

    prediction = model.predict([[val1, val2, val3, val4, val5, val6]])

    result = ""
    if prediction==[1]:
        result = "You have a Diabetes"
        suggetions = "Suggetions"
    else:
        result = "You does not have a Diabetes"
        suggetions = "Suggetions"

    return render(request, 'predict.html', {"result1": result, "suggetions": suggetions})

def suggetions(request):
    return render(request, 'suggetions.html')
