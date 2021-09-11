import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
##%matplotlib inline
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score

Data = pd.read_csv('C://Users//nidhchoudhary//Downloads//sonuslaptop//Salary_Data.csv')

##Understand the dataset
# #Identifying the number of features or columns
print(len(Data.columns))
#Identifying the features or columns
print(Data.columns)
#Identify the size of the dataset
print(Data.shape)
#Checking if the dataset has empty cells
print(Data.isnull().values.any())
#Identifying the number of empty cells by features or columns
print(Data.isnull().values.sum())
#Identifying the number of empty cells by features or columns
Data.fillna(4825,axis = 1,inplace= True)
##Check Again if Value is filled or not
print(Data.isnull().values.any())

###Performing EDA on DataSet
##Histogram EDA Years of Experience
plt.figure(figsize=(20,10))
plt.subplot(2,4,2)
plt.hist(Data['YearsExperience'])
plt.title("Experience")
#plt.show()
##Histogram EDA Salary
plt.subplot(2,4,5)
plt.hist(Data['Salary'])
plt.title("Salary")
#plt.show()
##Density Plot
plt.subplot(2,4,2)
sns.distplot(Data['Salary'],kde=True)
#plt.show()

##violin plot
plt.subplot(2,4,5)
sns.violinplot(Data['Salary'])
#plt.show()

plt.subplot(2,4,5)
sns.violinplot(Data['YearsExperience'])
#plt.show()
##Boxplot
plt.boxplot(Data['Salary'])
#plt.show()
####Qplot

sm.qqplot(Data['Salary'],line='45')
plt.show()


###All Graph Shows there are no outliers and Salary Data is not Normal
##This can be checked via Saphiro Test

alpha = 0.5
def normality_Check(df):
	for columnName,j in df.iteritems():
		print("Shapiro test for {columnName}".format(columnName=columnName))
		pvalue1 = stats.shapiro(j)
		pvalue = round(pvalue1[1],2)
		if pvalue > alpha:
			print("Data is not Normal")
		else:
			print("Data is  Normal")


normality_Check(Data)


##This Indicates Experience Data is Normal and Salary Data is not Normal
##Next step to Normalise the Salary Data

##Normalizing the data creating new columns for Normalize Data

Data['Norm_Salary'] = preprocessing.normalize(Data[['Salary']],axis= 0)
Data['Norm_YearsExperience'] = preprocessing.normalize(Data[['YearsExperience']],axis= 0)
print(Data.head())

##Declare x,y
##Train the model 
##Fit the model
##R2_score
##Coeff
##Intercept
##Rootmeansquare

##Linear Regression using scikit-learn 


print(Data)

print(Data.iloc[0:1,0:1])
print(Data.index[['YearsExperience']].tolist())

def regression(Data):
    regression = LinearRegression()
    x = Data.iloc[:,1:2]###Need to check
    y = Data.iloc[:,0:1]##Need to check
    #x = Data['YearsExperience']
    #y = Data['Salary']
    #print(x)
    #print(y)
    regression.fit(x,y)
    y_pred = regression.predict(x)
    ##Coefficient of predictor
    print("Coefficient of Predictor :",regression.coef_)
    # ##intercept  of predictor
    print("Coefficient of Intercept :",regression.intercept_)
    # ##Root mean square
    print("Root Mean Square:",mean_squared_error(y,y_pred))
    # ##R2
#     print("R2:",r2_score(y,y_pred))


# ##Drive the code

# regression(Data[['Salary','YearsExperience']]) 


#SLR model using Stats model.formula.api

def smf_ols(Data):
	x = Data.iloc[:,1:2]
	y = Data.iloc[:,0:1]
     ##Train the modelm
	model = smf.ols('y~x',data=Data).fit()
     ##Summary
	print(model.summary())
	y_predict = model.predict(x)
	error = y -y_predict
	#print("actual Error",error)

 ##Drive the code 

smf_ols(Data[['Salary','YearsExperience']])



  


