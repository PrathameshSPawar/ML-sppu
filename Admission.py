import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import precision_recall_curve,recall_score,accuracy_score,confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import confusion_matrix

df=pd.read_csv('./Admission_Predict.csv')
df
p=df.isna().sum()
df['Chance of Admit ']=[1 if each > 0.75 else 0 for each in df['Chance of Admit ']]
df
df.drop('Serial No.',axis=1)
x=df[['GRE Score','TOEFL Score','University Rating','SOP','LOR ','CGPA','Research']]
y=df['Chance of Admit ']
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.80,random_state=42)
model= DecisionTreeRegressor()
model.fit(x_train,y_train)
y_predict=model.predict(x_test)
accuracy_score(y_test,y_predict)
confusion_matrix(y_test,y_predict)
model2 = LogisticRegression()
model2.fit(x_train,y_train)
y_pred2 = model2.predict(x_test)
accuracy_score(y_test,y_pred2)
confusion_matrix(y_test,y_predict)
cm2 = confusion_matrix(y_test, y_pred2)
plt.figure(figsize=(4, 4))
sns.heatmap(cm2, annot=True, fmt='d', cmap='Greens', cbar=False)
plt.title('Confusion Matrix - Logistic Regression')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
