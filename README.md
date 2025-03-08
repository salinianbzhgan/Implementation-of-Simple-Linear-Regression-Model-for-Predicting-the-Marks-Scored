# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries. 
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn. 
4.Assign the points for representing in the graph. 
5.Predict the regression for marks by using the representation of the graph. 
6.Compare the graphs and hence we obtained the linear regression for the given datas.
## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: SALINI A
RegisterNumber:  212223220091
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print("df.head\n",df.head())
print("df.tail\n",df.tail())
x = df.iloc[:,:-1].values
print("Array value of x:",x)
y = df.iloc[:,1].values
print("Array value of y:",y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print("Values of y predict:\n",y_pred)
print("Array values of y test:\n",y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='purple')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:
## df.head
![Screenshot 2025-03-08 142157](https://github.com/user-attachments/assets/81bbd1db-123c-484f-84e8-fc7109a8fcb3)

## df.tail
![Screenshot 2025-03-08 142208](https://github.com/user-attachments/assets/528f6bbb-a017-4de4-b369-cf62038fb71c)

## Array value of X
![Screenshot 2025-03-08 142245](https://github.com/user-attachments/assets/fe447518-d105-43c6-9d64-1081dd95026d)

## Array value of Y
![Screenshot 2025-03-08 142312](https://github.com/user-attachments/assets/4d51febf-01fd-4150-99aa-974462ddc1d9)

## Values of Y prediction
![Screenshot 2025-03-08 142322](https://github.com/user-attachments/assets/56249ecb-aa54-4289-a339-0279a73b5219)

## Array values of Y test
![Screenshot 2025-03-08 142349](https://github.com/user-attachments/assets/731fd81a-e96c-492b-8fc3-6e2ffaf58a2e)

## Training Set Graph
![Screenshot 2025-03-08 142425](https://github.com/user-attachments/assets/8ee6ba9b-68f9-4886-9820-05c18f8183f8)

## Test set graph
![Screenshot 2025-03-08 142439](https://github.com/user-attachments/assets/3393a7e8-509b-4281-a552-537aae02559a)

## Values of MSE, MAE and RMSE
![Screenshot 2025-03-08 142452](https://github.com/user-attachments/assets/e49c8001-7df2-4cc4-8d11-8cba52924910)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
