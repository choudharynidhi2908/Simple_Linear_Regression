import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns


df = pd.read_csv('USA_Housing.csv')
#print(df.head())

df.info()
df.columns

sns.pairplot(df)
##Distribution Plot for Target

sns.distplot(df['Price'])

#print(df.corr())
sns.heatmap(df.corr(),annot = True)

x = df.iloc[:,0:5]
y = df['Price']

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size= 0.4,random_state = 101)

from sklearn.linear_model import LinearRegression
Linear_Variable = LinearRegression()
Linear_Variable.fit(x_train,y_train)

##Find Intercept
print(Linear_Variable.intercept_)

##Finding Coeff
print(Linear_Variable.coef_)

##Creating DataFrame for above Coeff

coeff = pd.DataFrame(Linear_Variable.coef_,x.columns,columns = ['coeff'])
print(coeff)

##Predicting The test model

predicted_price = Linear_Variable.predict(x_test)

##Compare Actual vs Predictied price
##1st Method pyplot

plt.scatter(y_test,predicted_price)
plt.show()
##Plotting Histogram for predictions

sns.distplot(y_test-predicted_price)
##Evaluation of Regression metric
from sklearn import metrics
error = y_test-predicted_price

MAE = metrics.mean_absolute_error(y_test,predicted_price)
print(MAE)

MSE = metrics.mean_squared_error(y_test,predicted_price)
print(MSE)

RMSE = np.sqrt(MSE)
print(RMSE)