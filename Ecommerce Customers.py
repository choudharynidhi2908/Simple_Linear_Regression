##Question Company should focus more on website or mobile app for business
##Project from Udemy Sction 15(85)

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline

df = pd.read_csv('C:\\Users\\nidhchoudhary\\OneDrive - Deloitte (O365D)\\Assignment\\Task\\Ecommerce Customers.csv')
df.head()
df.columns
print(df.isnull().values.any())
sns.jointplot(data = df,x= 'Time on Website',y = 'Time on App')
sns.pairplot(data = df)
y =df ['Yearly Amount Spent']
y
#x = df.iloc[:,3:7]
x= df[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
x.columns
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 101)
from sklearn.linear_model import LinearRegression
Linear_Variable = LinearRegression()
Linear_Variable.fit(x_train,y_train)
#To findintercept
print('Intercept of model',Linear_Variable.intercept_)
##to find Coeff
coeff = print('Codeff of model',Linear_Variable.coef_)
coeff_tab = pd.DataFrame(data = coeff ,index = x.columns,columns = ['Coeff_Value'])

##From Above Data sete analyse freesing all other attributes how much 1 attribute changes effect the Y
##Predicting the Test Model
predicted_value = Linear_Variable.predict(x_test)
##lineplot between predicted and actual values

plt.scatter(y_test,predicted_value)
plt.plot(y_test,predicted_value)
plt.xlabel('Actual_Money_Spent')
plt.ylabel('Predicted_Money_Spent')
##Plotting Distribution Plot
sns.distplot(y_test-predicted_value)
from sklearn import metrics
##MEan Absolute Error

MAE = print('Mean absolute Error',metrics.mean_absolute_error(y_test,predicted_value))
MSE = metrics.mean_squared_error(y_test,predicted_value)
print('Mean squared Error',MSE)
RMSE = print('Root Mean squared Error',np.sqrt(MSE))


print('Answer: Since just considering the coeff values change in 1 unit of website will effect 0.19$ in yearly spent which is a good sign for business but this can be further blend to find more accurate result')