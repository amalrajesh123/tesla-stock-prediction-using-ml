import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from matplotlib import style
style.use("fivethirtyeight")
import os
for dirname, _, filenames in os.walk(''):
    for filename in filenames:
        print(os.path.join(dirname, filename))
 df=pd.read_csv("/content/ferrari.csv")
df.head() 
df.info()  
df.describe() 
df["HL_Perc"]=(df["High"]-df["Low"])/df["Low"]*100
df["CO_Perc"]=(df["Close"]-df["Open"])/df["Open"]*100 
dates=np.array(df["Date"])
dates_check=dates[-30:]
dates=dates[:-30]  
df=df[["HL_Perc","CO_Perc","Adj Close","Volume"]]
df["Price next month"]=df["Adj Close"].shift(-30)
df.tail()
df.tail()
X= np.array(df.drop(["Price next month"],axis=1))
X=preprocessing.scale(X)
X_check=X[-30:]
X=X[:-30]
df.dropna(inplace=True)
y=np.array(df["Price next month"])
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
model=RandomForestRegressor()
model.fit(X_train,y_train)
conf=model.score(X_test,y_test)
print(conf)
model.fit(X,y)
predictions=model.predict(X_check)
actual=pd.DataFrame(dates,columns=["Date"])
actual["Close Price"]=df["Adj Close"]
actual["Forecast"]=np.nan
actual.set_index("Date", inplace=True)
forecast=pd.DataFrame(dates_check,columns=["Date"])
forecast["Forecast"]=predictions
forecast["Close Price"]=np.nan
forecast.set_index("Date",inplace=True)
var=[actual,forecast]
result=pd.concat(var)
result.plot(figsize=(20,10),linewidth=1.5)
plt.legend(loc=2,prop={'size':20})
plt.xlabel('Date')

plt.ylabel('Price')
