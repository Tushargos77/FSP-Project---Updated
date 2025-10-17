import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

df=pd.read_csv("/content/city_day.csv")
df.head()

df.info()

df.describe()

df.isna().sum()

"""## **EDA Handling Missing **"""

pmean=df["PM2.5"].mean()
df["PM2.5"].fillna(pmean,inplace=True)

pmmean=df["PM10"].mean()
df["PM10"].fillna(pmmean,inplace=True)

nmean=df["NO"].mean()
df["NO"].fillna(nmean,inplace=True)
nomean=df["NO2"].mean()
df["NO2"].fillna(nomean,inplace=True)
noxmean=df["NOx"].mean()
df["NOx"].fillna(noxmean,inplace=True)
nhmean=df["NH3"].mean()
df["NH3"].fillna(nhmean,inplace=True)
cmean=df["CO"].mean()
df["CO"].fillna(cmean,inplace=True)
smean=df["SO2"].mean()
df["SO2"].fillna(smean,inplace=True)
omean=df["O3"].mean()
df["O3"].fillna(omean, inplace=True)
bmean=df["Benzene"].mean()
df["Benzene"].fillna(bmean,inplace=True)
tmean=df["Toluene"].mean()
df["Toluene"].fillna(tmean,inplace=True)
xmean=df["Xylene"].mean()
df["Xylene"].fillna(xmean,inplace=True)
amean=df["AQI"].mean()
df["AQI"].fillna(amean,inplace=True)

df=df.drop('AQI_Bucket',axis=1)

df.isna().sum()

"""**Dividing the Data into X and Y**"""

x=df.iloc[:,2:13].values
y=df.iloc[:,-1].values

x

y

from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest= train_test_split(x,y,test_size=0.4,random_state=0)

from sklearn.linear_model import LinearRegression

linreg=LinearRegression()

linreg.fit(xtrain,ytrain)

ypred=linreg.predict(xtest)

"""Model Evaluation"""

x=df.iloc[:,2:13]
y=df.iloc[:,-1]

linreg.intercept_

linreg.coef_

coef_df=pd.DataFrame(linreg.coef_,x.columns,columns=["Coefficient"])
coef_df

plt.scatter(ytest,ypred)

sns.distplot((ytest-ypred),bins=50)

from sklearn.metrics import mean_absolute_error as mae, mean_squared_error as mse,r2_score

print(f"MAE:-{mae(ytest,ypred)}")

print(f"MSE:-{mse(ytest,ypred)}")

print(f"RMSE:-{np.sqrt(mse(ytest,ypred))}")

print(f"R-squared:-{r2_score(ytest,ypred)}")
