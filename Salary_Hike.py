import pandas as pd 
import matplotlib.pyplot as py
import numpy as np 
import statsmodels.formula.api as smf 
import warnings
warnings.filterwarnings('ignore')


Sal_Hike = pd.read_csv ('C:/Users/nidhchoudhary/Desktop/Assignment/SIMPLE_LINEAR_Regression/salary_data.csv')


print(Sal_Hike.head())


print('Training For SLR Started..')

first_model = smf.ols('Salary~YearsExperience',data= Sal_Hike).fit()

print(first_model.summary())##0.957

pred = first_model.predict(Sal_Hike.iloc[:,0])
print(pred)

py.scatter(x=Sal_Hike['YearsExperience'],y=Sal_Hike['Salary'],color= 'black')
py.plot(Sal_Hike['YearsExperience'],pred,color= 'red')
py.xlabel('YearsExperience')
py.ylabel('Salary')
py.show()

print(pred.corr(Sal_Hike.Salary))


model2 = smf.ols('Salary~np.log(YearsExperience)',data= Sal_Hike).fit()
print(model2.summary())##0.854

pred2 = model2.predict(Sal_Hike.iloc[:,0])
print(pred2)

py.scatter(x=Sal_Hike['YearsExperience'],y=Sal_Hike['Salary'],color = 'green')
py.plot(Sal_Hike['YearsExperience'],pred2,color = 'orange')
py.xlabel('YearsExperience')
py.ylabel('Salary')
py.title('2nd Model')
py.show()



model3 = smf.ols('np.log(Salary)~YearsExperience',data= Sal_Hike).fit()
print(model3.summary())##0.932
pred3 = model3.predict(Sal_Hike)

print(pred3)

py.scatter(x=Sal_Hike['YearsExperience'],y=Sal_Hike['Salary'],color = 'green')
py.plot(Sal_Hike['YearsExperience'],pred3,color = 'orange' )
py.xlabel('YearsExperience')
py.ylabel('Salary')
py.title('3rd Model')
py.show()

Sal_Hike['YEXP'] = Sal_Hike.YearsExperience*Sal_Hike.YearsExperience
model4 = smf.ols('Salary~YEXP+YearsExperience',data=Sal_Hike).fit()
print(model4.summary())###0.957
pred4 = model4.predict(Sal_Hike)

py.scatter(x=Sal_Hike['YearsExperience'],y=Sal_Hike['Salary'],color = 'green')
py.plot(Sal_Hike['YearsExperience'],pred4,color = 'orange' )
py.xlabel('YearsExperience')
py.ylabel('Salary')
py.title('4th Model')


##Since Quadratic Model has highest r2 value so this is the best model
py.show()

