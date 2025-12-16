import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
import pickle

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('gas_turbine_CO.csv')
df.drop_duplicates(inplace=True)  
df.drop(['Unnamed: 0'], axis=1,inplace=True) 

outlier_cols = ['NOX', 'TIT', 'CO']

def remove_outliers_iqr(data, column):
  q1,q2,q3 = np.percentile(data[column],[25,50,75])
  #print("q1,q2,q3 is :",q1,q2,q3)
  IQR = q3-q1
  #print("IQR is :" ,IQR)
  lower_limit = q1-(1.5*IQR)
  upper_limit = q3+(1.5*IQR)
  data[column]=np.where(data[column]>upper_limit,upper_limit,data[column]) # Capping the upper limit
  data[column]=np.where(data[column]<lower_limit,lower_limit,data[column]) # Flooring the lower limit

for column in outlier_cols:
  remove_outliers_iqr(df,column)


#splitting data into dependent and independent columns
x=df.drop('NOX',axis=1)
y=df['NOX']

#Scaling
normalisation = StandardScaler()
x_scaled = normalisation.fit_transform(x)
# Coverting to Dataframe
x=pd.DataFrame(x_scaled)
with open('scaling.pkl', 'wb') as f:
    pickle.dump(normalisation, f)

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state =42,test_size=0.33)

# Train a Linear Regression model
#model = LinearRegression()
model = RandomForestRegressor(n_estimators=600,max_depth=40, min_samples_split=2,min_samples_leaf=1,max_features='sqrt',random_state=42,n_jobs=-1)
model.fit(x_train, y_train)

# save the model 
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Model saved as model.pkl")

