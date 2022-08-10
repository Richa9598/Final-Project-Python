#keam clustering
import matplotlib
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

df=pd.read_csv('Dry_Bean_Dataset.csv')
print(df.head())

plt.scatter(df['Area'],df['Perimeter'])
plt.show()

km = KMeans(n_clusters=7)
print(km)

y_predict=km.fit_predict(df[['Area','Perimeter']])
print(y_predict)

df['Cluster']=y_predict
print(df.head())
df1=df[df.Cluster==0];
df2=df[df.Cluster==1];
df3=df[df.Cluster==2];
df4=df[df.Cluster==3];
df5=df[df.Cluster==4];
df6=df[df.Cluster==5];
df7=df[df.Cluster==6];

plt.scatter(df1.Area,df1['Perimeter'], color='green')
plt.scatter(df2.Area,df2['Perimeter'], color='red')
plt.scatter(df3.Area,df3['Perimeter'], color='yellow')
plt.scatter(df4.Area,df4['Perimeter'], color='blue')
plt.scatter(df5.Area,df5['Perimeter'], color='black')
plt.scatter(df6.Area,df6['Perimeter'], color='pink')
plt.scatter(df7.Area,df7['Perimeter'], color='orange')

plt.xlabel('Area')
plt.ylabel('Preimeter')
plt.show()


##########################################################################################
#importing the environments
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

#adding data file
data = pd.read_csv('Dry_Bean_Dataset.csv',header=0)
data = data.dropna();
#printing list of datas
print(data.shape)
print(list(data.columns))

#data viewing based on class column
data.head()
data['Class'].unique()
data['Class'].value_counts()
sns.countplot(x='Class',data=data,palette='hls')
plt.show()
plt.savefig('count_flass')


#counting the percentage split of classes (Seker, Barbunya, Bombay, Cali, Dermosan, Horoz and Sira)
length=len(data);
print(length);
count_no_Seker=0;

count_no_Seker = len(data[data['Class']=='SEKER'])
count_sub_Barbunya = len(data[data['Class']=='BARBUNYA'])
count_no_Bombay = len(data[data['Class']=='BOMBAY'])
count_sub_Cali = len(data[data['Class']=='CALI'])
count_no_Dermosan = len(data[data['Class']=='DERMASON'])
count_sub_Horoz = len(data[data['Class']=='HOROZ'])
count_no_Sira = len(data[data['Class']=='SIRA'])

pct_of_no_Seker = count_no_Seker/(count_no_Seker+count_sub_Barbunya+count_no_Bombay+count_sub_Cali+count_no_Dermosan+count_sub_Horoz+count_no_Sira)
pct_of_no_Barbunya = count_sub_Barbunya/(count_no_Seker+count_sub_Barbunya+count_no_Bombay+count_sub_Cali+count_no_Dermosan+count_sub_Horoz+count_no_Sira)
pct_of_no_Bombay = count_no_Bombay/(count_no_Seker+count_sub_Barbunya+count_no_Bombay+count_sub_Cali+count_no_Dermosan+count_sub_Horoz+count_no_Sira)
pct_of_no_Cali = count_sub_Cali/(count_no_Seker+count_sub_Barbunya+count_no_Bombay+count_sub_Cali+count_no_Dermosan+count_sub_Horoz+count_no_Sira)
pct_of_no_Dermosan = count_no_Dermosan/(count_no_Seker+count_sub_Barbunya+count_no_Bombay+count_sub_Cali+count_no_Dermosan+count_sub_Horoz+count_no_Sira)
pct_of_no_Horoz = count_sub_Horoz/(count_no_Seker+count_sub_Barbunya+count_no_Bombay+count_sub_Cali+count_no_Dermosan+count_sub_Horoz+count_no_Sira)
pct_of_no_Sira = count_no_Sira/(count_no_Seker+count_sub_Barbunya+count_no_Bombay+count_sub_Cali+count_no_Dermosan+count_sub_Horoz+count_no_Sira)

print("percentage of Seker is : ", pct_of_no_Seker*100)
print(" ")
print("percentage of Barbunya is : ", pct_of_no_Barbunya*100)
print(" ")

print("percentage of Bombay is : ", pct_of_no_Bombay*100)
print(" ")

print("percentage of Cali is : ", pct_of_no_Cali*100)
print(" ")

print("percentage of Dermonsa is : ", pct_of_no_Dermosan*100)
print(" ")

print("percentage of Haroz is : ", pct_of_no_Horoz*100)
print(" ")

print("percentage of Sira is : ", pct_of_no_Sira*100)
print(" ")

print(data.groupby('Class').mean())





cat_vars=['Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength', 'AspectRation', 'Eccentricity', 'ConvexArea', 'EquivDiameter', 'Extent', 'Solidity', 'roundness', 'Compactness', 'ShapeFactor1', 'ShapeFactor2', 'ShapeFactor3', 'ShapeFactor4', 'Class']
for var in cat_vars:
    cat_list='var'+'_'+var
    cat_list = pd.get_dummies(data[var], prefix=var)
    data1=data.join(cat_list)
    data=data1

cat_vars=['Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength', 'AspectRation', 'Eccentricity', 'ConvexArea', 'EquivDiameter', 'Extent', 'Solidity', 'roundness', 'Compactness', 'ShapeFactor1', 'ShapeFactor2', 'ShapeFactor3', 'ShapeFactor4', 'Class']
data_vars=data.columns.values.tolist()
to_keep=[i for i in data_vars if i not in cat_vars]

#final data
data_final=data[to_keep]
print(data_final.columns.values)

X = data_final.loc[:, data_final.columns != 'Class']
y = data_final.loc[:, data_final.columns == 'Class']
from imblearn.over_sampling import SMOTE
os = SMOTE(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
columns = X_train.columns
os_data_X,os_data_y=os.fit_sample(X_train, y_train)
os_data_X = pd.DataFrame(data=os_data_X,columns=columns )
os_data_y= pd.DataFrame(data=os_data_y,columns=['y'])
# we can Check the numbers of our data
print("length of oversampled data is ",len(os_data_X))
print("Number of no Seker oversampled data",len(os_data_y[os_data_y['Class']=='SEKER']))
print("Number of Barbuniya oversamped",len(os_data_y[os_data_y['Class']=='BARBUNYA']))
print("Proportion of no Seker data in oversampled data is ",len(os_data_y[os_data_y['Class']=='SEKER'])/len(os_data_X))
print("Proportion of Barbuniya data in oversampled data is ",len(os_data_y[os_data_y['Class']=='BARBUNYA'])/len(os_data_X))
print("Hello there")

data_final_vars=data_final.columns.values.tolist()
y=['Class']
X=[i for i in data_final_vars if i not in y]
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
rfe = RFE(logreg, 20)
rfe = rfe.fit(os_data_X, os_data_y.values.ravel())
print(rfe.support_)
print(rfe.ranking_)

cols=['Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength', 'AspectRation', 'Eccentricity', 'ConvexArea', 'EquivDiameter', 'Extent', 'Solidity', 'roundness', 'Compactness', 'ShapeFactor1', 'ShapeFactor2', 'ShapeFactor3', 'ShapeFactor4', 'Class']
X=os_data_X[cols]
y=os_data_y['Class']
import statsmodels.api as sm
logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary2())

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)
print('the percentage accuracy of is'.format(logreg.score(X_test, y_test)))

