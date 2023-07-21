# ClusterWise: Wine Data Exploration

## Wine Dataset Clustering
 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1MnnLm4nhtdIgZUqm3-5efOiyyB14pv5e/view?usp=sharing) and [Dataset Source](https://archive.ics.uci.edu/dataset/109/wine)

## Problem:
 More than 178 wine samples are present in the dataset. Manually analyzing each wine component makes it tedious and difficult to separate each type of wine. Hence, finding clusters of the same type of wine requires the application of Machine Learning models.

## Goal:
 Goal is to predict the class of wine from the 13 measured parameters, and find out the major differences among the three different classes.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Approach

### Step 1: Loading packages

 ```python
 import pandas as pd
 from sklearn import cluster
 from pandas._libs.tslibs import dtypes
 from sklearn.preprocessing import StandardScaler
 from sklearn.decomposition import PCA
 import plotly.express as px
 from sklearn.metrics import silhouette_score
 from sklearn.cluster import AgglomerativeClustering
 from sklearn.cluster import KMeans
 from DataProcess import ClusterProcess
 ```

### Step 2: Preprocessing the data

### Step 3: Performing Principal Component Analysis

### Step 4: Visualizing 3D Scatter Plot

### Step 5: Training the dataset in clustering models
 Training in different clustering models to find the best fitting clustering for this dataset,

 * [Agglomerative Clustering](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html)
 * [K-Means Clustering](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)

### Step 6: Predicting the result

### Citation:
 * [Learning to Classify Text](https://www.nltk.org/book/ch06.html)
 * [K-Means Clustering](https://towardsdatascience.com/k-means-clustering-algorithm-applications-evaluation-methods-and-drawbacks-aa03e644b48a)
 * [Explanation of Principal Component Analysis (PCA)](https://builtin.com/data-science/step-step-explanation-principal-component-analysis)
 * Tan, Pang-Ning, et al. Introduction to Data Mining. Pearson Education, 2020.
