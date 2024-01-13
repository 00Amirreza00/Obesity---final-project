import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt 
import seaborn as sns
import streamlit as st

st.image('pict.jpg')

st.title('Obesity Project')
st.write('The dataset includes data for the estimation of obesity levels in individuals from the countries of Mexico, Peru, and Colombia, based on their eating habits and physical condition. The dataset consists of 17 attributes and 2111 records, each labeled with the class variable "NObesity" (Obesity Level), allowing for the classification of data using feature values. ')
st.write('The dataset was downloaded from Kaggle: https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition')

# Read the data from the csv file
df = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')

if st.button('Variables table'):
    st.write(pd.read_csv('variables_table.csv'))

if st.button('Raw data'):
    st.write(df)

st.sidebar.title('Variables table')
st.sidebar.write(pd.read_csv('variables_table.csv').iloc[5:,[0,3]].reset_index(drop=True))

st.subheader('Data Cleaning and Preprocessing')
with st.expander("See explanation"):
    st.write('Data is generally clean, but some missing values are randomly added to the data as well as the data types are fixed in the data.')
    st.write('The data is then preprocessed by converting categorical variables into dummy variables and the target variable is encoded into numerical values.')

#Adding some missing values to the data
random_state = np.random.RandomState(42)
for i in range(3):
    random_column = random_state.choice(df.columns, size=1)[0]
    random_indexes = random_state.randint(0,df[random_column].shape[0], int(len(df.index)*0.02+i/100*len(df.index)))
    df[random_column][random_indexes] = np.nan

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

df_copy= pd.concat([df_copy, df_copy_getdummy], axis=1)
df_copy.drop(['Gender', 'CAEC', 'MTRANS','CALC'],axis=1, inplace=True)

df_copy['family_history_with_overweight'].replace({'yes':1,'no':0},inplace=True)
df_copy['SMOKE'].replace({'yes':1,'no':0},inplace=True)
df_copy['FAVC'].replace({'yes':1,'no':0},inplace=True)
df_copy['SCC'].replace({'yes':1,'no':0},inplace=True)
df_copy['NObeyesdad'].replace(dict(zip(df_copy['NObeyesdad'].unique(),np.arange(len(df_copy['NObeyesdad'].unique())))),inplace=True)

# EDA Part------------------------------------------
st.subheader('Data Exploration')
with st.expander("See explanation"):
    st.write('The data is explored by plotting the distribution of the target variable and the features.')

    # Grouping the data by the target variable
    df_groupby=df.groupby('NObeyesdad')

    obesity_df_new = df.copy()
    obesity_df_new['NObeyesdad'].replace(dict(zip(list(df.groupby('NObeyesdad').groups.keys()),[0,0,1,1,1,1,1])),inplace=True)

    # Distribution of the target variable
    selected_variable = st.radio('Show the distribution of:',options=['Target Variable','Age and Weight'])

    fig1, ax = plt.subplots(1,2)
    obesity_df_new['NObeyesdad'].value_counts().plot(kind='pie',autopct='%1.1f%%',figsize=(10,10),ax=ax[0],title='Obesity Status',ylabel='',labels=['Obesity','No Obesity'])
    df['NObeyesdad'].value_counts().plot(kind='pie',autopct='%1.1f%%',figsize=(10,10),ax=ax[1],startangle=90,title='Distribution of obesity levels in individuals',ylabel='')
    

    # Distribution of the features
    fig2, ax = plt.subplots(1,2,figsize=(10,4))
    ax[0].hist(df['Age'],bins=20)
    ax[0].set_ylabel('Number of people')
    ax[0].set_xlabel('Age')
    ax[0].set_title('Age Distribution')
    
    ax[1].hist(df['Weight'],bins=20)
    ax[1].set_ylabel('Number of people')
    ax[1].set_xlabel('Weight')
    ax[1].set_title('Weight Distribution')


    if selected_variable=='Target Variable':
        st.write(fig1)
    elif selected_variable=='Age and Weight' :
        st.write(fig2)


    # Corrolation
    # if st.button('Corrolation'):
    corr_opt = st.radio('Show the correlation between:',options=['Features and target variable','All features - Heatmap'])
        
    df_copy_corr=df_copy.corr()
    if corr_opt=='Features and target variable':
        st.bar_chart(df_copy_corr['NObeyesdad'].sort_values(ascending=False)[1:],width=0,height=0,use_container_width=True)
    elif corr_opt=='All features - Heatmap':
        pl=plt.figure(figsize=(40,30))
        sns.heatmap(df_copy_corr,annot=True,cmap='coolwarm')
        st.write(pl)

    st.write('Select a feature to plot vs Obesity types:')
    selected_feature=st.selectbox('',['Select one option','Weight','Age','FCVC','NCP','FAF','TUE'])
    if selected_feature!='Select one option':
        fig,ax=plt.figure(figsize=(6,4)),plt.axes()
        ax=df_groupby[selected_feature].mean().plot(kind='bar',title=selected_feature+' vs Obesity Levels',xlabel='Obesity Level',ylabel='Average '+selected_feature)
        st.write(fig)

    st.write('All types of obesity are cosidered as one class and the Normal and Insufficient weight are considered as another class.')
    st.write('The data is then plotted again based on the new target variable.')
    
    selected_fn= st.selectbox('Show the distribution of the new target variable:',options=['Select one option','Having food between meals - CAEC','Transportation type - MTRANS','Consuming alcohol - CALC','Weight','SMOKE','Family history with overweight','Physical activity frequency - FAF'])
    if selected_fn!='Select one option':
        if selected_fn=='Having food between meals - CAEC':
            fig,ax = plt.figure(figsize=(8,6)),plt.axes()
            ax =obesity_df_new.groupby('NObeyesdad')['CAEC'].value_counts().plot(kind='bar',color=['red','red','red','red','blue','blue','blue','blue'])
            ax.set_ylabel('# of people')
            ax.set_xlabel('Obesity')
            ax.set_title('CAEC vs obesity')
            st.write(fig)

        if selected_fn=='Transportation type - MTRANS':
            fig,ax = plt.figure(figsize=(8,6)),plt.axes()
            ax =obesity_df_new.groupby('NObeyesdad')['MTRANS'].value_counts().plot(kind='bar',color=['red','red','red','red','red','blue','blue','blue','blue','blue'])
            ax.set_ylabel('# of people')
            ax.set_xlabel('Obesity')
            ax.set_title('MTRANS vs obesity')
            st.write(fig)

        if selected_fn=='Consuming alcohol - CALC':
            fig,ax = plt.figure(figsize=(8,6)),plt.axes()
            ax =obesity_df_new.groupby('NObeyesdad')['CALC'].value_counts().plot(kind='bar',color=['red','red','red','red','blue','blue','blue'])
            ax.set_ylabel('# of people')
            ax.set_xlabel('Obesity')
            ax.set_title('CALC vs obesity')
            st.write(fig)

        if selected_fn=='Weight':
            fig,ax = plt.figure(figsize=(8,6)),plt.axes()
            ax =obesity_df_new.groupby('NObeyesdad')['Weight'].mean().plot(kind='bar',color=['red','blue'])
            ax.set_ylabel('Average Weight')
            ax.set_xlabel('Obesity')
            ax.set_title('Weight vs obesity')
            ax.xaxis.set_ticklabels(['no','yes'],rotation=0)
            st.write(fig)

        if selected_fn=='SMOKE':
            fig,ax = plt.figure(figsize=(8,6)),plt.axes()
            ax =obesity_df_new.groupby('NObeyesdad')['SMOKE'].value_counts().plot(kind='bar',stacked=True,rot=0,color=['red','red','blue','blue'])
            ax.set_ylabel('# of people')
            ax.set_xlabel('Obesity')
            ax.set_title('Smoking vs obesity')
            ax.xaxis.set_ticklabels(labels=['No obesity - No Smoking','No obesity - Smoking', 'Obesity - No Smoking','Obesity - Smoking'],rotation=30)
            st.write(fig)

        if selected_fn=='Family history with overweight':
            fig,ax = plt.figure(figsize=(8,6)),plt.axes()
            ax =obesity_df_new.groupby('NObeyesdad')['family_history_with_overweight'].value_counts().plot(kind='bar',stacked=True,rot=0,color=['red','red','blue','blue'])
            ax.set_ylabel('# of people')
            ax.set_xlabel('Obesity')
            ax.set_title('Family history with overweight vs obesity')
            ax.xaxis.set_ticklabels(labels=['No obesity - No family history','No obesity - family history', 'Obesity - No family history','Obesity - family history'],rotation=30)
            st.write(fig)

        if selected_fn=='Physical activity frequency - FAF':
            fig,ax = plt.figure(figsize=(8,6)),plt.axes()
            plt.bar([0,1],obesity_df_new.groupby('NObeyesdad')['FAF'].mean(),color=['red','blue'])
            plt.ylabel('Average FAF')
            plt.xlabel('Obesity')
            plt.title('FAF vs obesity')
            plt.xticks([0,1],['No','Yes'],rotation=0)
            st.write(fig)

# Modelling Part ---------------------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,  mean_squared_error
from sklearn.metrics import classification_report

st.subheader('Data Modelling')
with st.expander("See explanation"):
    # train test split
    st.write('The data is split into train and test sets and then the following models are trained and evaluated:')
    st.write('1. Linear Regression')
    st.write('2. KNN')
    st.write('3. Ridge Regression')
    st.write('4. KMeans Clustering')


    model_selected= st.selectbox('Select a model:',['Select one option','Linear Regression','KNN','Ridge Regression','KMeans Clustering'])

    st.write('Please set the test data size')
    test_s = st.slider("Slide me", min_value=0.0, max_value=1.0, step=0.01, value=0.2)
    x=df_copy.drop('NObeyesdad',axis=1).values
    y=df_copy['NObeyesdad'].values
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=test_s,random_state=42)

    # Standardization
    if st.checkbox('Standardization'):
        scaler= StandardScaler()
        x_train=scaler.fit_transform(x_train)
        x_test=scaler.transform(x_test)
    if model_selected!='Select one option':
        # KNN Model
        if model_selected=='KNN':
            n_neighbors = st.number_input('Please set the number of neighbors: ',min_value=1,max_value=15)
            knn=KNeighborsClassifier(n_neighbors)
            knn.fit(x_train,y_train)
            y_pred=knn.predict(x_test)
            st.write('KNN Modelling Score: ',knn.score(x_test,y_test))
            st.write('Confusion Matrix: ',confusion_matrix(y_test,y_pred))
            st.write('Classification Reprot: ')
            st.code(classification_report(y_test,y_pred))

        # KMeans Clustering
        if model_selected=='KMeans Clustering':
            n_clusters = st.number_input('Please set the number of clusters: ',min_value=1,max_value=15)
            model=KMeans(n_clusters)
            model.fit(x_train)
            labels=model.predict(x_train)
            new_labels=model.predict(x_test)

            st.write('accuracy: ',model.score(x_test,y_test))
            st.write('Model Inertia: ',model.inertia_)
            # Cross Tabulation
            dddf=pd.DataFrame({'labels':labels,'NObeyesdad':y_train})
            ct=pd.crosstab(dddf['labels'],dddf['NObeyesdad'])
            st.write('Cross Tabualtion:',ct)

            fig = plt.figure(figsize=(8,6))
            plt.scatter(x_train[:,0],x_train[:,1],c=labels)
            plt.title('KMeans Clustering')  
            st.pyplot(fig)

            centroids=model.cluster_centers_
            centroids_x=centroids[:,0]
            centroids_y=centroids[:,1]
            fig = plt.figure(figsize=(8,6))
            plt.scatter(centroids_x,centroids_y,s=50)
            plt.title('Centroids')
            st.pyplot(fig)

        # Ridge Regression
        if model_selected=='Ridge Regression':
            scores=[]
            alpha_min = st.number_input('Please set the minimum value for the alpha : ',min_value=0,max_value=100)
            alpha_max = st.number_input('Please set the maximum value for the alpha: ',min_value=0,max_value=100)
            alpha_step = st.number_input('Please set the step: ',min_value=0.0,max_value=100.0)
            for alpha in np.arange(alpha_min,alpha_max,alpha_step):
                ridge=Ridge(alpha=alpha)
                ridge.fit(x_train,y_train)
                scores.append(ridge.score(x_test,y_test))

            fig=plt.figure(figsize=(8,6))
            plt.plot(np.arange(alpha_min,alpha_max,alpha_step),scores)
            plt.title('Alpha in Ridge Regression')
            st.pyplot(fig)

            Alpha= np.arange(alpha_min,alpha_max,alpha_step)[scores.index(max(scores))]
            ridge=Ridge(alpha=Alpha)
            ridge.fit(x_train,y_train)
            st.write('Ridge Model: The Best Score: ',ridge.score(x_test,y_test))

        # Linear Regression
        if model_selected=='Linear Regression':
            linear_Reg=LinearRegression()
            linear_Reg.fit(x_train,y_train)
            y_pred=linear_Reg.predict(x_test)
            st.write('R^2: ',linear_Reg.score(x_test,y_test))
            st.write('MSE: ',mean_squared_error(y_test,y_pred))
            st.write('RMSE: ',np.sqrt(mean_squared_error(y_test,y_pred)))
    





