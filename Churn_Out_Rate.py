import pandas as pd 
import matplotlib.pyplot as py
import numpy as np 
import statsmodels.formula.api as smf 

churn_rate = pd.read_csv ('C:/Users/nidhchoudhary/Desktop/Assignment/SIMPLE_LINEAR_Regression/emp_data.csv')

print(churn_rate.head())


print('Training For SLR Started..')

model = smf.ols("Churn_out_rate~Salary_hike",data= churn_rate).fit()

print(model.summary())

pred = model.predict(churn_rate.Salary_hike)##R2 0.831

print(pred)


model.conf_int(0.05) # 95% confidence interval

model2 = smf.ols("Churn_out_rate~np.log(Salary_hike)",data= churn_rate).fit()

print(model2.summary())

pred2 = model2.predict(churn_rate.Salary_hike)##r2 0.849

print(pred2)

model3 = smf.ols("np.log(Churn_out_rate)~Salary_hike",data= churn_rate).fit()##0.874
print(model3.summary())

pred3 = model3.predict(churn_rate.Salary_hike)
print(pred3)

churn_rate['Quad4'] =  (churn_rate.Salary_hike*churn_rate.Salary_hike)



print(churn_rate.head())
model4 = smf.ols('Churn_out_rate~Quad4+Salary_hike',data = churn_rate).fit()

print(model4.summary())####0.974

pred4 = model4.predict(churn_rate)
print(pred4)


####Since 4th Model has highest R2 value so thisQuadratic Model is the best model
