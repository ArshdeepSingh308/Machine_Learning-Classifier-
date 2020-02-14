#!/usr/bin/env python
# coding: utf-8

# In[1]:


from IPython.display import display, HTML, Image

from TAS_Python_Utilities import data_viz
from TAS_Python_Utilities import data_viz_target
from TAS_Python_Utilities import visualize_tree

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
import random

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn import metrics
from scipy.spatial import distance
from sklearn.neighbors import DistanceMetric



get_ipython().run_line_magic('matplotlib', 'inline')
#%qtconsole


# # Template Match

# In[2]:


# Create a new classifier which is based on the sckit-learn BaseEstimator and ClassifierMixin classes
class TemplateMatch(BaseEstimator, ClassifierMixin):
    
    # Constructor for the classifier object
    def __init__(self, dist_param= 'euclidean'):
        self.dist_param = dist_param
        
        #print(self.dist_param)
    
    # The fit function to train a classifier
    def fit(self, X, y):
            
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        # Count the number of occurrences of each class in the target vector (uses mupy unique function that returns a list of unique values and their counts)
        unique, counts = np.unique(y, return_counts=True)
        
        self.classes_ = unique
        #print(unique, counts)
        
        got_index = []
        for i in unique:
            indexes = np.where(y == i)
            got_index.append(indexes[0])

        dist = []
        for i in got_index:
            newArr = []
            for j in i:
                select_label = X[j]
                newArr.append(select_label)
            dist.append(np.mean(newArr,axis=0))

        # Create a new dictionary of classes and their distances
        self.distances_ = dict(zip(unique, dist))
        
        # Return the classifier
        return self
    
    # The predict function to make a set of predictions for a set of query instances
    def predict(self, X):
        
        # Check is fit had been called by confirming that the distances_ dictionary has been set up
        check_is_fitted(self, ['distances_'])

        unique = []
        dist = []
        for i in self.distances_.keys():
            unique.append(i)
            dist.append(self.distances_.get(i))
            
        # Check that the input features match the type and shape of the training features
        X = check_array(X)

        # Initialise an empty list to store the predictions made
        predictions = list()
        
        
        # Iterate through the query instances in the query dataset 
        predictions =[]
        my_dict = dict(zip(unique, dist))
        
        # Iterate through the query instances in the query dataset 
        distance_metric_model = DistanceMetric.get_metric(self.dist_param)
        print("Distance Metric : ",self.dist_param)
        
        for instance in X:
            var_model = []
            for item in my_dict:
                current_label = my_dict[item]
                array = np.vstack((current_label, instance))
                dist = distance_metric_model.pairwise(array)

                var = np.amin(np.array(dist)[dist != np.amin(dist)])
                var_model.append(var)

            min_value = min(var_model)
            dict_model_dist = dict(zip(unique, var_model))

            for key in dict_model_dist.keys():
                if (dict_model_dist[key] == min_value):
                    label_pred = key
            predictions.append(label_pred) 

        return np.array(predictions)
    
    
    # The predict function to make a set of predictions for a set of query instances
    def predict_proba(self, X):

        # Check is fit had been called by confirming that the distances_ dictionary has been set up
        check_is_fitted(self, ['distances_'])

        # Check that the input features match the type and shape of the training features
        X = check_array(X)

        # Initialise an array to store the prediction scores generated
        predictions = np.zeros((len(X), len(self.classes_)))
        
        distance_metric_model = DistanceMetric.get_metric(self.dist_param)
        
        unique = []
        dist = []
        for i in self.distances_.keys():
            unique.append(i)
            dist.append(self.distances_.get(i))

        # Iterate through the query instances in the query dataset
        predictions_prob = []
        my_dict = dict(zip(unique, dist))
        
        for instance in X:
            prob_dist = []
            for item in my_dict:
                current_label = my_dict[item]
                array = np.vstack((current_label, instance))
                dist = distance_metric_model.pairwise(array)

                var = np.amin(np.array(dist)[dist != np.amin(dist)])
                prob_dist.append(1/var)

            sum_value = sum(prob_dist)
            dict_model_prob = dict(zip(unique, (prob_dist/sum_value)))

            predictions_prob.append(dict_model_prob)

        return predictions_prob



# # MNIST Dataset

# In[5]:


dataset = pd.read_csv('fashion-mnist_train.csv')

Y = dataset.pop('label').values
X = dataset.values


# In[6]:


X_train_plus_valid, X_test, y_train_plus_valid, y_test     = train_test_split(X, Y, random_state=0,                                     train_size = 0.7)

X_train, X_valid, y_train, y_valid     = train_test_split(X_train_plus_valid,                                         y_train_plus_valid,                                         random_state=0,                                         train_size = 0.5/0.7)


# In[7]:


for metric in distance_metrics: 
    my_model = TemplateMatch(dist_param = metric)
    my_model.fit(X_train, y_train)
    #print (my_model.distances_)
    y_pred = my_model.predict(X_valid)
    
    # Print performance details
    accuracy = metrics.accuracy_score(y_valid, y_pred) # , normalize=True, sample_weight=None
    print("Accuracy: " +  str(accuracy))
    
    # Print nicer homemade confusion matrix
    print("Confusion Matrix")
    display(pd.crosstab(y_valid, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))
    print("\n------------------------\n")


# In[8]:


my_model = TemplateMatch(dist_param = 'euclidean')
my_model.fit(X_train_plus_valid, y_train_plus_valid)
#print (my_model.distances_)
y_pred = my_model.predict(X_test)

# Print performance details
accuracy = metrics.accuracy_score(y_test, y_pred) # , normalize=True, sample_weight=None
print("Accuracy: " +  str(accuracy))

# Print nicer homemade confusion matrix
print("Confusion Matrix")
display(pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))
print("\n------------------------\n")


# In[9]:


y_pred = my_model.predict_proba(X_test)

# Unable to print the probabilities

_ = pd.DataFrame(y_pred).hist(figsize = (10,10))


# ## $Cross$ $Validation$ (Different distance metric)

# In[10]:


for metric in distance_metrics: 
    my_model = TemplateMatch(dist_param = metric)
    print ("Distance Metric : ", metric)
    scores = cross_val_score(my_model, X_train_plus_valid, y_train_plus_valid, cv=2,  n_jobs=-1) #verbose= 2
    print (scores.mean())
    print ("\n------------------------\n")


# # Grid Search

# In[11]:


param_grid = [
 {'dist_param': distance_metrics}
]

# Perform the search
my_tuned_model = GridSearchCV(TemplateMatch(), param_grid, cv=2, verbose = 2, n_jobs=-1)
my_tuned_model.fit(X_train_plus_valid, y_train_plus_valid)

# Print details
print("\nBest parameters set found on development set:")
print("Distance Metric : ", my_tuned_model.best_params_)
print("Accuracy Score : ", my_tuned_model.best_score_)

