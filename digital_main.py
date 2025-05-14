# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 17:07:34 2025

@author: DELL
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

data=pd.read_csv('digital_marketing_campaign_dataset real.csv')

df1=data.drop(columns=['CustomerID','AdvertisingPlatform','AdvertisingTool'],axis=1)

df=pd.get_dummies(df1,drop_first=True)

X=df.drop(columns=['Conversion'])
y=df['Conversion'].values

from sklearn.preprocessing import MinMaxScaler

from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)

# Fit and apply the transformation to oversample the minority class
X_resampled, y_resampled = smote.fit_resample(X, y)
pd.Series(y_resampled).value_counts()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

selected_features = ['EmailClicks', 'TimeOnSite', 'AdSpend', 'ClickThroughRate', 'EmailOpens']  

X_train_selected = X_train[selected_features]  

# Retrain the scaler on the selected features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_selected)
X_test_selected=X_test[selected_features]
X_test_scaled=scaler.transform(X_test_selected)

from xgboost import XGBClassifier
xgb=XGBClassifier()
model=xgb.fit(X_train_scaled,y_train)
y_pred_xgb=xgb.predict(X_test_scaled)


import joblib
joblib.dump(model, 'xgboost_model9.joblib')
joblib.dump(scaler, 'minmax_scaler9.joblib')

