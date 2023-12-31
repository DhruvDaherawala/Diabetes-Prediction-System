import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = pd.read_csv("D:\Dhruv College Coding + Lab Manual + Projects\Dhruv DE Projects\Diabetes Prediction System\Dataset\main_data.csv")
print(data)

duplicate_rows = data[data.duplicated()]
print(duplicate_rows)

x = data.drop('Outcome', axis=1)
y = data['Outcome']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
x_test

model = LogisticRegression()
model.fit(x_train, y_train)