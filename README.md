# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the needed packages.
2. Assigning hours To X and Scores to Y.
3. Plot the scatter plot.
4. Use mse,rmse,mae formula to find.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: SWETHA.S
RegisterNumber: 212222230155

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('/content/student_scores.csv')
#displaying the content in datafile
df.head()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('/content/student_scores.csv')
#displaying the content in datafile
df.tail()

# Segregating data to variables
X=df.iloc[:,:-1].values
X

Y=df.iloc[:,1].values
Y

#splitting train and test data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)

#displaying predicted values
Y_pred

#displaying actual values
Y_test

#graph plot for training data
plt.scatter(X_train,Y_train,color='black')
plt.plot(X_train,regressor.predict(X_train),color='violet')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#graph plot for test data
plt.scatter(X_test,Y_test,color='violet')
plt.plot(X_train,regressor.predict(X_train),color='black')
plt.title("Hours vs Scores(Test set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print("RMSE= ",rmse)
*/
```

## Output:

df.head():

![Screenshot 2023-08-24 091419](https://github.com/swethaselvarajm/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119525603/0d3910f7-776c-4096-bf29-baf36dbea7e6)

df.tail():

![Screenshot 2023-08-24 091426](https://github.com/swethaselvarajm/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119525603/1a6110fc-a265-42f6-a95f-0330f3eb9d2f)

Array value of X:

![Screenshot 2023-08-24 091436](https://github.com/swethaselvarajm/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119525603/92f36a09-ea7d-4414-bedc-c19e1ee5835d)

Array value of Y:

![Screenshot 2023-08-24 091448](https://github.com/swethaselvarajm/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119525603/300ea970-2e67-4116-ba1b-205cacce499c)

Values of Y prediction:

![Screenshot 2023-08-24 091459](https://github.com/swethaselvarajm/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119525603/446c4c65-eeb7-455e-907f-fe7a3eeb43a6)

Array values of Y test:

![Screenshot 2023-08-24 091506](https://github.com/swethaselvarajm/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119525603/e1c3a155-2bf3-4856-9ec7-e5f6e1ee870c)

Training set graph:

![Screenshot 2023-08-24 091525](https://github.com/swethaselvarajm/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119525603/81a6936f-ec9c-434a-acda-5ce7f3f2ceca)

Test set graph:

![Screenshot 2023-08-24 091818](https://github.com/swethaselvarajm/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119525603/00b2c7f0-6769-4d6b-96fe-7d9322ae8be4)

Values of MSE,MAE and RMSE:

![Screenshot 2023-08-24 091830](https://github.com/swethaselvarajm/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119525603/a0dfe318-6969-4529-b6e9-ccaeba764f93)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
