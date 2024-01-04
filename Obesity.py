import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Read the data from the csv file
df = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')

# Print the first 5 rows of the dataframe.
df.head()

# Print the last 5 rows of the dataframe.
df.tail()

# Print the shape of the dataframe.
df.shape



random_state = np.random.RandomState(42)
for i in range(3):
    random_column = random_state.choice(df.columns, size=1)[0]
    random_indexes = random_state.randint(0,df[random_column].shape[0], int(len(df.index)*0.02+i/100*len(df.index)))
    df[random_column][random_indexes] = np.nan

# np.random.seed(42)
# df[df.columns[np.random.randint(0, len(df.columns))]][np.random.randint(0,len(df.index),] = np.nan
# df[df.columns[np.random.randint(0, len(df.columns))]][np.random.randint(0,len(df.index),int(len(df.index)*0.02))] = np.nan
# df[df.columns[np.random.randint(0, len(df.columns))]][np.random.randint(0,len(df.index),int(len(df.index)*0.01))] = np.nan
    
print(df.infio())

print(df.isnull().sum())

sorted_nan_values=df.isnull().sum().sort_values(ascending=False)
i=0
while sorted_nan_values.values[i]!=0:
    df[sorted_nan_values.index[i]].fillna(df[sorted_nan_values.index[i]].mean(), inplace=True)
    i=i+1


#data preprocessing
    
df_copy = df.copy()
df_copy_getdummy = pd.get_dummies(df_copy[['Gender', 'CAEC', 'MTRANS','CALC']])
df_copy_getdummy.head()
df_copy= pd.concat([df_copy, df_copy_getdummy], axis=1)
df_copy.drop(['Gender', 'CAEC', 'MTRANS','CALC'],axis=1, inplace=True)



df_copy['family_history_with_overweight'].replace({'yes':1,'no':0},inplace=True)
df_copy['SMOKE'].replace({'yes':1,'no':0},inplace=True)
df_copy['FAVC'].replace({'yes':1,'no':0},inplace=True)
df_copy['SCC'].replace({'yes':1,'no':0},inplace=True)
df_copy['NObeyesdad'].replace(dict(zip(df_copy['NObeyesdad'].unique(),np.arange(len(df_copy['NObeyesdad'].unique())))),inplace=True)

df_corr=df_copy.corr()
print('Corrolation between y and the other features: ',df_corr['NObeyesdad'].sort_values(ascending=False))



