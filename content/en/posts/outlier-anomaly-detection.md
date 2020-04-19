---
author: "Mohit Sharma"
title:  "How to identify Anomalies/Outliers in Python"
date: "2019-12-20"
description: "In this article, we discuss both univariate and multivariate outlier detection techniques with examples."
tags:
- Python
- Outlier
- Anomaly
- Multivariate
- Univariate
- DBSCAN
- IsolationForest
- Cooks Distance

categories:
- Statistics
- Machine Learning
- Unsupervised Learning

libraries:
- katex
- mathjax

featured_image: "notesfeatureimg/outlier-detection-anomaly.png"
---
## Introduction
Data objects or points which exhibit very different behavior than the expectations are called as **outliers or anomalies**. They can indicate variability in the measurement, an error in the collection, a new point(due to some changes), or it could be true, which happens to be away from most of the observations.

{{< alert theme="warning" >}}
 Detecting and treating outliers is an important part of data exploration.
{{< /alert >}}

<!--more-->
## Different Methods For Detecting Outlier
Before we dig into different methods which you can use to identify outliers, it is important to understand the different types of an outlier.

## Types of outliers
The outliers can be of two types:
- **Univariate** - An observation is termed as **univariate outlier** if we consider the distribution of only one feature.
- **Multivariate** - On the other hand, **Multivariate outliers** are found by considering n-dimensional space.

## Example #1 - Univariate outlier detection using boxplots
```
import seaborn as sns
sns.boxplot(x=boston['ZN'])
```
![Anomany Outlier Detection - Boxplot](/images/outlier/boxplot.png)
In the above plot, the back dots represent outliers. These outliers are calculated based on the below-mentioned formula. Remember, the outliers can be on either side.

\(Higher side Outliers = Q3 \+ 1.5 \* IQR \)  
\(Lower side Outliers = Q1 \- 1.5 \* IQR \)  

The IQR stands for Inter Quartile Range.

IQR = Q3 - Q1

Here - Q3 is the 75<sup>th</sup> and Q1 is 25<sup>th</sup> percentile.

## Example #2 - Univariate outlier detection using Z-Score
Z-score is a measure that helps us know how many standard deviations below or above the population mean a raw score is.

Z<sub>score</sub> \= \frac{x - \mu\over \sigma}

```
from scipy.stats import zscore
import numpy as np
z = np.abs(zscore(boston))
print(z)
```

{{< boxmd >}}
**Output**
[[0.41978194 0.28482986 1.2879095  ... 1.45900038 0.44105193 1.0755623 ]
 [0.41733926 0.48772236 0.59338101 ... 0.30309415 0.44105193 0.49243937]
 [0.41734159 0.48772236 0.59338101 ... 0.30309415 0.39642699 1.2087274 ]
 ...
 [0.41344658 0.48772236 0.11573841 ... 1.17646583 0.44105193 0.98304761]
 [0.40776407 0.48772236 0.11573841 ... 1.17646583 0.4032249  0.86530163]
 [0.41500016 0.48772236 0.11573841 ... 1.17646583 0.44105193 0.66905833]]
{{< /boxmd >}}

From the above generated Z-scores, the observations which have got a score higher than 3 are the outliers.

```
threshold = 3
print(np.where(z > 3))
```

{{< boxmd >}}
**Output**
(array([ 55,  56,  57, 102, 141, 142, 152, 154, 155, 160, 162, 163, 199,
       200, 201, 202, 203, 204, 208, 209, 210, 211, 212, 216, 218, 219,
       220, 221, 222, 225, 234, 236, 256, 257, 262, 269, 273, 274, 276,
       277, 282, 283, 283, 284, 347, 351, 352, 353, 353, 354, 355, 356,
       357, 358, 363, 364, 364, 365, 367, 369, 370, 372, 373, 374, 374,
       380, 398, 404, 405, 406, 410, 410, 411, 412, 412, 414, 414, 415,
       416, 418, 418, 419, 423, 424, 425, 426, 427, 427, 429, 431, 436,
       437, 438, 445, 450, 454, 455, 456, 457, 466], dtype=int64),
       array([ 1,  1,  1, 11, 12,  3,  3,  3,  3,  3,  3,  3,  1,  1,  1,  1,  1,
        1,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  5,  3,  3,  1,  5,
        5,  3,  3,  3,  3,  3,  3,  1,  3,  1,  1,  7,  7,  1,  7,  7,  7,
        3,  3,  3,  3,  3,  5,  5,  5,  3,  3,  3, 12,  5, 12,  0,  0,  0,
        0,  5,  0, 11, 11, 11, 12,  0, 12, 11, 11,  0, 11, 11, 11, 11, 11,
       11,  0, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11],
      dtype=int64))
{{< /boxmd >}}

The first array contains the row numbers where the values were higher than the threshold( set at 3 in our case). The second array contains the column numbers.

## Example #3 - Multivariate outlier detection using DBSCAN
**DBSCAN(Density-based spatial clustering of applications with noise)** is a density-based clustering algorithm. The algorithm works on the intuition that clusters are nothing but a collection of similar points which are present as dense regions in the data space.

### DBSCAN Parameters Explained
1. **eps**: It is the distance between two data points. If the distance between two points is closer or equal to the `eps` value than these two points are considered normal points or neighbors. If the distance between two points is greater than the specified `eps` value than that point is considered as noise. A smaller `eps` value results in too many outliers. A lager `eps` value than different clusters will get merged, and most of the data points will be in the same clusters.

2. **min_samples**: Represents the number of points to be considered as core data points around a specific point. If the data is dense, then it is advised to select a larger number of `min_samples`. The `min_samples` can also be derived using the following formula:

min_samples = D + 1

Here, D is the number of Dimensions(features) in a dataset.

### How DBSCAN algorithm works?
The algorithm works on the premise of multivariate clustering. It looks for data points that are geometrically closer to each other. To understand the algorithm, you should first understand the following terms.

1. **Core Points** - These are the points that have `min_samples` within `eps` distance.

2. **Border Points** - These are the points that are in the range of `eps` but have less than `min_samples` points.

3. **Noise** - All other points which are neigther **Border Points** nor are **Core Points**.

The algorithm can be abstracted in the following steps:

1. Start with a random point; find neighboring points that are within the `eps` radius. If these nearby points exceed or equal the `min_sample` then they care marked as core points.

2. Check if the core point has been assigned to a cluster, if not, create a new group.

3. Recursively find density connected points and assign them to the same cluster as a core point.
**The above process is called chaining**. A point x and y are said to be close enough if there is a point z, which has enough number of points within `eps` distance. So basically, if y is a neighbor of z, z is a neighbor of x.

4. Like this iterate over the remaining data points. The points that do not get assigned to a cluster are considered as noise or outliers.

### Working Example
Let us load the Python packages, which we will be using for our example.

```
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
```
Loading and preparing `iris` dataset for the example.

{{< alert theme="warning" >}}
If variables are on different scales, then ensure that you bring them on the same scale. If in doubt, it is advised to go ahead with the scaling.
{{< /alert >}}


```
iris = load_iris()
X_train = iris.data
Y_train = iris.target
columns = iris.feature_names
#Loading iris dataset

X_train = pd.DataFrame(X_train)
X_train.columns = columns
# Adding column names

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
# Scaling - Bringing data points on same scale
```
Now that the dataset is all prepared let's use DBSCAN to identify the outliers in the dataset.
```
# Fitting the DBSCAN model
clustering = DBSCAN(eps=0.5, min_samples=5).fit(X_train)

# Abstracting the data points
labels = clustering.labels_

# Adding False for non core points
core_points = np.zeros_like(labels, dtype = bool)
core_points[clustering.core_sample_indices_] = True

# counting the number of outliers
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)
```
**Out of 150 data points, around 34 points have been identified as outlier**

### How to visualize the DBSCAN output
We will be using `matplotlib` to plot the DBSCAN output. The points which are represented by small solid dots are the outliers.

```
import matplotlib.pyplot as plt
plt.figure(figsize=(15,8))
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = X_train[class_member_mask & core_samples]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)

    xy = X_train[class_member_mask & ~core_samples]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()
```
![Visualising DBSCAN outliers](/images/outlier/dbscan.png)

**Cooks distance** captures the change in the regression output by removing individual data points. The results will change drastically if an influential point is removed. Cook's distance measures the effect of an individual data point on all of the fitted values. The cook's distance statistics is calculated as given below:

![cooks distance formula](/images/outlier/cooks-distance.png)

**IF THE** D<sub>i</sub>  $\ge$ 1 **THE VALUE IS CONSIDERED EXTREME AND IS TREATED AS AN OUTLIER**

### Working Example

```
from yellowbrick.regressor import CooksDistance
from yellowbrick.datasets import load_concrete

# Load the regression dataset
X, y = load_concrete()

# Instantiate and fit the model
cd = CooksDistance()
cd.fit(X_train, Y_train)
cd.show()
```
![cooks distance formula](/images/outlier/cooks-distance-fig2.png)  
The example is borrowed from [yellobrick](https://www.scikit-yb.org/en/latest/api/regressor/influence.html) documentation.

{{< alert theme="danger" >}}
Cook's Distance is not effective in detecting a group of outliers. Because if you remove one value from the cluster of an outlier, the effect on the model will not be much.
{{< /alert >}}


### How does the isolation forest works
To isolate an observation, the algorithm uses a random feature and then splits the values between the minimum and the maximum value of the selected feature. The intuition behind the algorithm is very simple: isolating an anomaly should be easier as only a few conditions should be required to separate such cases from the normal observations. On the other hand, more conditions are required to isolate normal cases.

The **isolation forest** first constructs random decision trees or isolation trees and repeats the process several times. Then, the average path length is calculated and normalized.

{{< alert theme="success" >}}
The Isolation Forest algorithm shows strong promise as the other Machine Learning methods tend to work fine only in case the patterns are balanced, meaning the dataset contains the equal amount of normal and bad values in the dataset.
{{< /alert >}}


### Working Example

```
from yellowbrick.datasets import load_concrete
from sklearn.ensemble import IsolationForest

# Loading the concrete dataset
X_train, Y_train = load_concrete()

# Building isolationforest model to identify anomalies
isolationTree = IsolationForest(random_state = 0).fit(X_train)
pred_outliers_scores = isolationTree.decision_function(X_train)

#Plotting and visualising the data points
plt.figure(figsize=(20, 10))
plt.hist(pred_outliers_scores, bins = 50)

```
![Isolation Forest for anomaly detection](/images/outlier/isolation-tree.png)

<!--more-->

So we see that there are clusters under -0.04. Thus, observation with the average score for path lengths shorter than -0.04 will be considered as anomalies or outliers.

Please leave comments, if
1. You find anything incorrect.
2. You want to add more information to the topic.
3. You wish to add another example to the topic.
4. You need more details in regards to a specific section.
5. You are unable to execute an example code.
