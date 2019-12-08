#!/usr/bin/env python
# coding: utf-8

# In[1]:


import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


datasets = pd.read_csv("C:/Users/sourav/Desktop/Dataset-of-Thesis CSV.csv")


# In[5]:


datasets.head


# In[6]:


X = datasets.iloc[:, [2,3]].values
Y = datasets.iloc[:, 4].values


# In[7]:


from sklearn.model_selection import train_test_split
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 0.25, random_state = 0)


# In[8]:


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_Train = sc_X.fit_transform(X_Train)
X_Test = sc_X.transform(X_Test)


# In[18]:


from sklearn import svm
clf = svm.SVC(kernel='rbf')
clf.fit(X_Train, Y_Train) 


# In[20]:


yhat = clf.predict(X_Test)
yhat [0:5]


# In[21]:


from sklearn.metrics import classification_report, confusion_matrix
import itertools


# In[22]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[30]:


from sklearn.metrics import f1_score
f1_score(Y_Test, yhat, average='weighted') 


# In[31]:


from sklearn.metrics import jaccard_similarity_score
jaccard_similarity_score(Y_Test, yhat)

