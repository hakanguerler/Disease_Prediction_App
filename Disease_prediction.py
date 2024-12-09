#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,KFold,cross_val_score,GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix,classification_report,confusion_matrix,precision_score,roc_curve
import seaborn as sns
from sklearn.utils import shuffle
# from pandas_profiling import ProfileReport
from sklearn.linear_model import LogisticRegression, Perceptron, RidgeClassifier, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier 
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, VotingClassifier 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics


# In[8]:


df = pd.read_csv('dataset.csv')
#df = pd.read_csv('/Users/hakangurler/Desktop/final_DSR /diease_prediction/archive/dataset.csv')
df = shuffle(df,random_state=42)
df.head()


# In[9]:


df.isnull().sum()


# In[10]:


df = df.fillna("")
df['Symptom']=""
for i in range(1,18):
    df['s']=df["Symptom_{}".format(i)]
    df['Symptom']=df['Symptom']+df['s']


# In[11]:


df.head()


# In[12]:


for i in range(1,18):
    df=df.drop("Symptom_{}".format(i),axis=1)
df=df.drop("s",axis=1)


# In[13]:


df.head()


# In[14]:


df[0:1].values


# In[15]:


df['Disease'].value_counts()


# In[16]:


X=df['Symptom']
y=df['Disease']


# In[17]:


X.head(10)


# In[18]:


y.head(10)


# In[19]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.25,shuffle=True,random_state=44)


# In[20]:


y_train.head(10)


# In[21]:


print('Training Data Shape:', X_train.shape)
print('Testing Data Shape: ', X_test.shape)


# In[22]:


y_train.value_counts()


# In[23]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(X_train)
X_train_tfidf.shape


# In[24]:


pd.DataFrame(X_train_tfidf.toarray())


# In[25]:


from sklearn.svm import LinearSVC
clf = LinearSVC()
clf.fit(X_train_tfidf,y_train)


# In[26]:


from sklearn.pipeline import Pipeline
text_clf = Pipeline([('tfidf', TfidfVectorizer()),
                     ('clf', LinearSVC()),])


text_clf.fit(X_train, y_train)  


predictions = text_clf.predict(X_test)


# In[27]:


from sklearn import metrics
print(metrics.confusion_matrix(y_test,predictions))


# In[28]:


print(metrics.classification_report(y_test,predictions)) 


# In[29]:


print(metrics.accuracy_score(y_test,predictions))


# In[30]:


df1 = pd.read_csv('symptom_Description.csv')
#df1 = pd.read_csv('/Users/hakangurler/Desktop/final_DSR /diease_prediction/archive/symptom_Description.csv')
df1.head()


# In[31]:


df1.isnull().sum()


# In[32]:


df1.index=df1['Disease']
df1=df1.drop('Disease',axis=1)
df1.head(10)


# In[33]:


df2 = pd.read_csv('symptom_precaution.csv')
#df2 = pd.read_csv('/Users/hakangurler/Desktop/final_DSR /diease_prediction/archive/symptom_precaution.csv')
df2.head()


# In[34]:


df2.isnull().sum()


# In[35]:


df2 = df2.fillna("")
df2.head(10)


# In[36]:


df2['precautions']=""
df2['punc']=', '
for i in range(1,5):
    df2['s']=df2["Precaution_{}".format(i)]+df2['punc']
    df2['precautions']=df2['precautions']+df2['s']
df2.head(10)


# In[37]:


for i in range(1,5):
    df2=df2.drop("Precaution_{}".format(i),axis=1)
df2.head(10) 


# In[38]:


df2=df2.drop(['s','punc'],axis=1)
df2.head(10)


# In[39]:


df2.index=df2['Disease']
df2=df2.drop('Disease',axis=1)
df2.head()


# In[41]:


import streamlit as st
import pandas as pd

# Load the processed data (from the notebook)
def load_data():
    data = {
        "Disease": ["Malaria", "Drug Reaction", "Allergy"],
        "Precautions": [
            "Consult nearest hospital, avoid oily food, avoid non-veg food, keep mosquitos out",
            "Stop irritation, consult nearest hospital, stop taking the drug, follow up",
            "Apply calamine, cover area with bandage, use ice to compress itching",
        ]
    }
    return pd.DataFrame(data)

# Title
st.title("Disease Precaution Finder")

# Load data
df = load_data()

# Sidebar for selecting a disease
disease = st.sidebar.selectbox("Select a Disease", df["Disease"])

# Display Precautions
if disease:
    st.write(f"### Precautions for {disease}")
    precautions = df[df["Disease"] == disease]["Precautions"].values[0]
    st.write(precautions)


# In[42]:


import streamlit as st
import pandas as pd

# Title
st.title("Disease Precaution Finder")

# Description
st.write("""
This application allows users to search for diseases and view the recommended precautions.
It is built to help raise awareness about various diseases and promote health and safety.
""")

# Sidebar Upload
st.sidebar.header("Upload Processed Data")
uploaded_file = st.sidebar.file_uploader("Upload your processed disease-precaution CSV file", type=["csv"])

if uploaded_file:
    # Load the data
    df = pd.read_csv(uploaded_file, index_col=0)

    # Sidebar for selecting a disease
    disease = st.sidebar.selectbox("Select a Disease", df.index)

    # Display Precautions
    if disease:
        st.subheader(f"Precautions for {disease}")
        precautions = df.loc[disease, "precautions"]
        for i, precaution in enumerate(precautions.split(','), 1):
            st.write(f"{i}. {precaution.strip()}")

    # Display all diseases in the main page
    st.subheader("All Diseases and Precautions")
    st.dataframe(df)

else:
    st.info("Please upload a processed data file to proceed.")
    st.write("Hint: The file should be a CSV where one column is 'Disease' and another is 'precautions'.")

# Footer
st.write("---")
st.write("Made with ❤️ using Streamlit")


# In[ ]:




