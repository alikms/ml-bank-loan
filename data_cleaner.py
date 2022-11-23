import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,GridSearchCV
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
pd.set_option('display.max_columns',None)
df=pd.read_csv('train.csv')
df=df[['Loan Amount','Funded Amount','Home Ownership',
           'Delinquency - two years','Interest Rate','Revolving Balance',
           'Funded Amount Investor','Loan Status']]
x_train=df[['Loan Amount','Funded Amount','Home Ownership',
          'Delinquency - two years','Interest Rate','Revolving Balance',
         'Funded Amount Investor']][:54000]
y_train=df['Loan Status'][:54000]
x_test=df[['Loan Amount','Funded Amount','Home Ownership',
           'Delinquency - two years','Interest Rate','Revolving Balance',
           'Funded Amount Investor']][54000:]
y_test=df["Loan Status"][54000:]
#scale data
scaler=MinMaxScaler()
df=pd.DataFrame(scaler.fit_transform(df),columns=df.columns)
#fill missing values by neighbors
imputer=KNNImputer()
df=pd.DataFrame(imputer.fit_transform(df),columns=df.columns)
#print(df.isna().any())


#L1model=LinearRegression().fit(x_train,y_train)
L2model=LogisticRegression()
c_space=np.linspace(0.000001,1,20)
param_grid={'C':c_space,
            'penalty':['none','l2','l1']}
grid=GridSearchCV(L2model,param_grid)
grid.fit(x_train,y_train)
print(grid.best_estimator_.score(x_test,y_test),'\n',grid.best_estimator_,sep='')



#KNNmodel.score(x_train,y_train)
#y_predict=KNNmodel.predict(x_test)

#df.plot()
#df.dtypes
#df.astype('float')
'''
Funded Amount Investor
Interest Rate

Delinquency - two years
'''
