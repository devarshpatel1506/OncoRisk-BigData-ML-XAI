import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import joblib

from xgboost import XGBClassifier

df=pd.read_csv('final_risk_factor.csv')

X=df.drop(columns=['breast_cancer_history'])
y=df['breast_cancer_history']

print(df.head())

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#Working with Logistic Regression
lr=LogisticRegression()
lr.fit(X_train,y_train)
lr_prediction=lr.predict(X_test)

lr_acc=accuracy_score(lr_prediction,y_test)

print("Logistic Regression:",lr_acc)


#Working with RandomForestClassifier

rf=RandomForestClassifier()
rf.fit(X_train,y_train)
rf_prediction=rf.predict(X_test)
rf_acc=accuracy_score(rf_prediction,y_test)

print("Random Forest:",rf_acc)

#Working with XGBoost Classifier

y_train=y_train.replace(9,2)
y_test=y_test.replace(9,2)
xgboost=XGBClassifier(use_label_encoder=False)
xgboost.fit(X_train,y_train)
xgboost_prediction=xgboost.predict(X_test)
xgboost_acc=accuracy_score(xgboost_prediction,y_test)

print("XGBoost:",xgboost_acc)

joblib.dump(xgboost,'model.pkl')