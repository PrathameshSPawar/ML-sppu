import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import matplotlib.pyplot as plt

df=pd.read_csv('./temperatures.csv')
df

df.isna().sum()
df.info()

x=df[['YEAR']]
y=df[['ANNUAL']]

x_train,x_test,y_train,y_test= train_test_split(x,y,train_size=0.75,random_state=42)

model=LinearRegression()
model.fit(x_train,y_train)
y_predict=model.predict(x_test)
y_predict=model.predict(x_test)
r2_score(y_test,y_predict)
plt.scatter(x,y,label='Actual Temperature')
plt.plot(x_test,y_predict,color='red',label='Predicted Temperature')
plt.title('Month-wise Teemperature Prediction')
plt.xlabel('Month')
plt.ylabel('Temperature(celsius)')
plt.show()
sns.regplot(data=df,x=x_train,y=y_train,)

from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
print(f"MSE:  {mean_squared_error(y_test,y_predict)}")
print(f"MAE:  {mean_absolute_error(y_test,y_predict)}")
print(f"R-Sqaure :  {r2_score(y_test,y_predict)}")
