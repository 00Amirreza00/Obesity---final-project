import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt 
import seaborn as sns

# Read the data from the csv file
df = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')

# Print the first 5 rows of the dataframe.
df.head()

# Print the last 5 rows of the dataframe.
df.tail()

# Print the shape of the dataframe.
df.shape


#Adding some missing values to the data
random_state = np.random.RandomState(42)
for i in range(3):
    random_column = random_state.choice(df.columns, size=1)[0]
    random_indexes = random_state.randint(0,df[random_column].shape[0], int(len(df.index)*0.02+i/100*len(df.index)))
    df[random_column][random_indexes] = np.nan

print("Some information about the data:",df.info())
print("Null valaues: ",df.isnull().sum())

# Fill the missing values with the mean of the column   
sorted_nan_values=df.isnull().sum().sort_values(ascending=False)
i=0
while sorted_nan_values.values[i]!=0:
    df[sorted_nan_values.index[i]].fillna(df[sorted_nan_values.index[i]].mean(), inplace=True)
    i=i+1

# Fixing the data types
df.FCVC=round(df.FCVC)
df.NCP=round(df.NCP)
df.CH2O=round(df.CH2O)
df.FAF=round(df.FAF)
df.TUE=round(df.TUE)


#data preprocessing ------------------------------------------
    
df_copy = df.copy()
df_copy_getdummy = pd.get_dummies(df_copy[['Gender', 'CAEC', 'MTRANS','CALC']],drop_first=True)
df_copy_getdummy.head()
df_copy= pd.concat([df_copy, df_copy_getdummy], axis=1)
df_copy.drop(['Gender', 'CAEC', 'MTRANS','CALC'],axis=1, inplace=True)

df_copy['family_history_with_overweight'].replace({'yes':1,'no':0},inplace=True)
df_copy['SMOKE'].replace({'yes':1,'no':0},inplace=True)
df_copy['FAVC'].replace({'yes':1,'no':0},inplace=True)
df_copy['SCC'].replace({'yes':1,'no':0},inplace=True)
df_copy['NObeyesdad'].replace(dict(zip(df_copy['NObeyesdad'].unique(),np.arange(len(df_copy['NObeyesdad'].unique())))),inplace=True)

# Corrolation
df_copy_corr=df_copy.corr()
print('Corrolation between y and the other features: ',df_copy_corr['NObeyesdad'].sort_values(ascending=False))

# EDA Part------------------------------------------

plt.figure(figsize=(20,20))
sns.heatmap(df_copy_corr,annot=True,cmap='coolwarm')



# Grouping the data by the target variable
df_groupby=df.groupby('NObeyesdad')


plt =df_groupby['Weight'].mean().plot(kind='bar', figsize=(8,6),title='Weight vs Obesity')


plt =df_groupby['Age'].mean().plot(kind='bar', figsize=(8,6),title='Age vs Obesity')



plt =df_groupby['FCVC'].mean().plot(kind='bar', figsize=(8,6),title='FCVC vs Obesity')

plt =df_groupby['NCP'].mean().plot(kind='bar', figsize=(8,6),title='NCP vs Obesity')


   
plt =df_groupby['FAF'].mean().plot(kind='bar', figsize=(8,6),title='FAF vs Obesity')



plt =df_groupby['TUE'].mean().plot(kind='bar', figsize=(8,6),title='TUE vs Obesity')


plt =df_groupby['CALC'].value_counts().plot(kind='bar', figsize=(8,6),title='CALC vs Obesity')


plt = df_groupby['MTRANS'].value_counts().plot(kind='bar', figsize=(8,6),title= 'MTRANS vs Obesity')

plt = df_groupby['CAEC'].value_counts().plot(kind='bar', figsize=(8,6),title= 'CAEC vs Obesity')



obesity_df_new = df.copy()
obesity_df_new['NObeyesdad'].replace(dict(zip(list(df.groupby('NObeyesdad').groups.keys()),[0,0,1,1,1,1,1])),inplace=True)


plt.figure(figsize=(8,6))
ax =obesity_df_new.groupby('NObeyesdad')['Weight'].mean().plot(kind='bar')
ax.set_ylabel('Average Weight')
ax.set_xlabel('Obesity')
ax.set_title('Weight vs obesity')
ax.xaxis.set_ticklabels(['no','yes'],rotation=0)
plt.show()

plt.figure(figsize=(8,6))
ax =obesity_df_new.groupby('NObeyesdad')['SMOKE'].value_counts().plot(kind='bar',stacked=True,rot=0,color=['red','red','blue','blue'])
ax.set_ylabel('# of people')
ax.set_xlabel('Obesity')
ax.set_title('Smoking vs obesity')
ax.xaxis.set_ticklabels(labels=['No obesity - No Smoking','No obesity - Smoking', 'Obesity - No Smoking','Obesity - Smoking'],rotation=30)
plt.show()

plt.figure(figsize=(8,6))
ax =obesity_df_new.groupby('NObeyesdad')['family_history_with_overweight'].value_counts().plot(kind='bar',stacked=True,rot=0,color=['red','red','blue','blue'])
ax.set_ylabel('# of people')
ax.set_xlabel('Obesity')
ax.set_title('Family history with overweight vs obesity')
ax.xaxis.set_ticklabels(labels=['No obesity - No family history','No obesity - family history', 'Obesity - No family history','Obesity - family history'],rotation=30)
plt.show()

plt.figure(figsize=(8,6))
plt.bar([0,1],obesity_df_new.groupby('NObeyesdad')['FAF'].mean())
plt.ylabel('Average FAF')
plt.xlabel('Obesity')
plt.title('FAF vs obesity')
plt.xticks([0,1],['No','Yes'],rotation=0)
plt.show()



# Modelling Part ------------------------------------------

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,  mean_squared_error
from sklearn.metrics import classification_report


# train test split
x=df_copy.drop('NObeyesdad',axis=1).values
y=df_copy['NObeyesdad'].values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

knn=KNeighborsClassifier(n_neighbors=6)
knn.fit(x_train,y_train)
y_pred=knn.predict(x_test)
print('KNN Modelling Score: ',knn.score(x_test,y_test))
print('Confusion Matrix: ',confusion_matrix(y_test,y_pred))
print('Classification Reprot: ',classification_report(y_test,y_pred))

# Standardization
scaler= StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)

# KMeans Clustering
model=KMeans(n_clusters=7)
model.fit(x_train_scaled)
labels=model.predict(x_train_scaled)
new_labels=model.predict(x_test_scaled)

print('accuracy: ',model.score(x_test_scaled,y_test))
plt.scatter(x_train_scaled[:,0],x_train_scaled[:,1],c=labels)
plt.get_title('KMeans Clustering')  
plt.show()

centroids=model.cluster_centers_
centroids_x=centroids[:,0]
centroids_y=centroids[:,1]
plt.scatter(centroids_x,centroids_y,s=50)
plt.title('Centroids')
plt.show()

print('Model Inertia: ',model.inertia_)

# Cross Tabulation
dddf=pd.DataFrame({'labels':labels,'NObeyesdad':y_train})
ct=pd.crosstab(dddf['labels'],dddf['NObeyesdad'])
print('Cross Tabualtion:',ct)


# Ridge Regression
scores=[]
for alpha in np.linspace(0,100,1000):
    ridge=Ridge(alpha=alpha)
    ridge.fit(x_train_scaled,y_train)
    scores.append(ridge.score(x_test_scaled,y_test))


plt.plot(np.linspace(0,100,1000),scores)
plt.title('Alpha in Ridge Regression')
plt.show()

Alpha= np.linspace(0,100,1000)[scores.index(max(scores))]
ridge=Ridge(alpha=Alpha)
ridge.fit(x_train_scaled,y_train)
print('Ridge Model Score: ',ridge.score(x_test_scaled,y_test))

# Linear Regression
linear_Reg=LinearRegression()
linear_Reg.fit(x_train,y_train)
y_pred=linear_Reg.predict(x_test)
print('R^2: ',linear_Reg.score(x_test,y_test))
print('MSE: ',mean_squared_error(y_test,y_pred))
print('RMSE: ',np.sqrt(mean_squared_error(y_test,y_pred)))
 