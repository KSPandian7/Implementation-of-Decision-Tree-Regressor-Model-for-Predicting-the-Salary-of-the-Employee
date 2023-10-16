# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
STEP 1 :
Import the standard libraries. 2.Upload the dataset and check for any null values using .isnull() function.

STEP 2 :
Import LabelEncoder and encode the dataset.

STEP 3 :
Import DecisionTreeRegressor from sklearn and apply the model on the dataset.

STEP 4 :
Predict the values of arrays.

STEP 5 :
Import metrics from sklearn and calculate the MSE and R2 of the model on the dataset 7.Predict the values of array.

STEP 6 :
Apply to new unknown values.

## Program:

```ruby
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

Developed by: KULASEKARAPANDIAN K
RegisterNumber:  212222240052
```
```python
import pandas as pd
df = pd.read_csv("Salary.csv")
df.head()

df.info()

df.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

df["Position"]=le.fit_transform(df["Position"])
df.head()

x=df[["Position","Level"]]
x.head()

y=df[["Salary"]]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])
```

## Output:

### Data Head Values:
![OUTPUT](/head.png)

### Data Info:
![OUTPUT](/info.png)

### Data Isnull():
![OUTPUT](/isnull.png)

### Position Head Values :
![OUTPUT](/positiondata.png)

### Position & Level:
![OUTPUT](/position&level.png)

### MSE Value:
![OUTPUT](/mse.png)

### R² Value:
![OUTPUT](/r2.png)

### Prediction Value:
![OUTPUT](/pred56.png)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
