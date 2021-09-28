import pandas as pd 
import matplotlib.pyplot as py
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression 



score = pd.read_csv('C:\\Users\\nidhchoudhary\\Desktop\\Assignment\\Task\\Simple_Linear_Regression\\student_scores.csv')
print(score.shape)
print(score.head())
x=score['Hours']
y=score['Score']
py.scatter(x,y,color = 'red')
py.xlabel('Hours Studied Daily')
py.ylabel('Score Expected')
py.title('Simple_Linear_Regression')
py.show()


model = LinearRegression.fit(x,score['Score'])
# r_sq = model.score(x,y)
# print('Coefficient of ',r_sq)


