# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
#### 1.Import the standard Libraries. 
#### 2.Set variables for assigning dataset values. 
#### 3.Import linear regression from sklearn. 
#### 4.Assign the points for representing in the graph. 
#### 5.Predict the regression for marks by using the representation of the graph. 
#### 6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Santhosh V
RegisterNumber:  212224230251
*/
```
```
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt

dataset=pd.read_csv('student_scores.csv')
print(dataset.head())
dataset=pd.read_csv('student_scores.csv')
print(dataset.tail())
x=dataset.iloc[:,:-1].values
print(x)
y=dataset.iloc[:,1].values
print(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)
print(y_pred)
print(y_test)

plt.scatter(x_train,y_train,color='purple')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_absolute_error(y_test,y_pred)
print('Mean Square Error = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('Mean Absolute Error = ',mae)
rmse=np.sqrt(mse)
print("Root Mean Square Error = ",rmse)
```

## Output:
Head value
<img width="1090" height="115" alt="319854949-4ddf5d62-c261-42be-8b67-6f6df35f3d36" src="https://github.com/user-attachments/assets/b9778721-51ed-4e80-9f5b-5c75867d85bf" />
Tail value
<img width="1090" height="115" alt="319855044-dfa9fd82-e723-4ed4-aaec-ad66ffa348e1" src="https://github.com/user-attachments/assets/6c2bf6f5-c89e-46bb-9190-de8de5dd66fb" />
Compare 
<img width="1090" height="461" alt="319855079-9118849a-2323-4b9c-a362-3dc8d060587a" src="https://github.com/user-attachments/assets/1a0fbaf7-0922-4395-8390-db019e549f94" />
Predication value

<img width="1090" height="55" alt="319855219-5ed7921d-08e1-408e-8383-6899933b01ee" src="https://github.com/user-attachments/assets/01e31fbc-0ed6-40ae-a0d0-a605e3afc584" />


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
