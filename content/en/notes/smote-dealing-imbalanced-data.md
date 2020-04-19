---
author: "Mohit Sharma"
title:  "SMOTE for dealing with imbalanced dataset"
date: "2020-01-15"
description: "A brief note on how SMOTE works and how it can be used to train a model when one class dominates the other."
tags:
- Python
- SMOTE
- Imbalanced data
- Data preparation
- Upsampling

categories:
- Statistics


featured_image: "notesfeatureimg/smote.png"
---

It is generally not a good idea to train a Machine Learning algorithm when one of the class dominates the other. It is advisable to upsample the minority class or downsample the majority class. **Synthetic Minority Over-sampling Technique (SMOTE)** is one such algorithm that can be used to upsample the minority class.

![Synthetic Minority Over-sampling Technique (SMOTE) - fig 1](/images/smote/fig1.png)

## When to use SMOTE
Machine Learning algorithms find it challenging to learn the patterns if the examples from one of the classes are limited. The final results of a classification problem can also be misleading. For instance, in the case of strokes dataset, only 2% of the total recorded data points consist of individuals who have had a heart attack in the past. In such a case, a Machine Learning algorithm could classify everything as `No Stroke` and still be correct 98% of the time.

## How does SMOTE work
SMOTE works by manufacturing more minority classes between any two or more real minority instances. Step by step, guide on how the algorithm works.

1. Draw a line between the nearest minority instance( as per the parameter, k)
2. Generate new synthetic minority instances on these lines.

## Understanding SMOTE parameters
The `SMOTE()` function is available in `over_sampling` module inside the `imbalanced-learn ` Python package. Some of the parameters of the `SMOTE` function are deprecated and will not be available in version 0.60. So I'll only talk about parameters which are very important and are not deprecated. However, to understand them, we need to learn a bit more about how `SMOTE` works.

```
!pip install imbalanced-learn
# using this inside jupyter notebook or spyder GUI
```

The algorithm creates new minority instances at some distance from the existing minority classes, towards there neighbors. So, the question arises, which and how many neighbors are considered for every minority instance.

The **two** important parameters which `SMOTE` function takes are: `sampling_strategy` and `k_neighbors`.

At `k_neighbors = 1`, the closest minority class to a data point from the same class is considered. At `k_neighbors = 2`, both first and second closest neighbor of the same class is considered. The new data points are synthesized on the imaginary straight line, which connects the points. The exercise is repeated for all the minority data points.

### Smote at k_neighbors = 1
![Synthetic Minority Over-sampling Technique (SMOTE) - fig 2](/images/smote/fig2.png)

### Smote at k_neighbors = 2
![Synthetic Minority Over-sampling Technique (SMOTE) - fig 3](/images/smote/fig3.png)

`sampling_strategy` helps define how to resample the data set. The parameter can take inputs in the following forms:

1. `float` - it corresponds to the desired ratio between the number of samples from minority class over the majority class. However, this option is only available for the `binary classification` problem. An error is raised when passed for multi-class classification.

2. `str` - specify the class to be resampled. The possible choices are:

- 'minority': only the minority class is resampled

- 'not minority': All classes but the minority class is resampled

- 'not majority': All classes but the majority class is resampled

- 'all': resample all classes

- 'auto': equivalent to 'not majority'

3. `dict` - In a dictionary, key corresponds to the target variable classes.  The values correspond to the number of samples to be created for each target class.

## Example #1 SMOTE function code in Python
We will load a stroke dataset and check the distribution of target variable classes.

```
import pandas as pd
from imblearn.over_sampling import SMOTE
strokes = pd.read_csv("M:/GitHub/data/healthcare-dataset-stroke-data/train_2v.csv")
round(strokes.stroke.value_counts()/strokes.shape[0] * 100, 2)
```

{{< boxmd >}}
**Output**
0    98.2
1     1.8
{{< /boxmd >}}

You can see the dataset is highly imbalanced, with only ~2% of observations comprising of people who have had a heart attack.

Let us fit `SMOTE` and check out the number of observations in each category of the target variable.


```
# Prepared dataset, created dummy variable
Y_train = strokes["stroke"]
X_train = strokes.drop(['stroke', "id"], axis = 1)
X_train = pd.concat([pd.get_dummies(X_train[col]) for col in X_train], axis=1, keys=X_train.columns)

# Fit the SMOTE
sm = SMOTE(random_state=2)
X_train, Y_train = sm.fit_sample(X_train, Y_train)

# Getting the count of target variable class
np.bincount(Y_train)
```

## Example #2 SMOTE function code in Python
In  last example we saw that both the target levels had the same count. Sometimes you may be interested in increaing the minority class to a certain pre-defined numbers. For example, let's say you want to maintain 90 - 10 raito between 0 and 1.

```
# Prepared dataset, created dummy variable
Y_train = strokes["stroke"]
X_train = strokes.drop(['stroke', "id"], axis = 1)
X_train = pd.concat([pd.get_dummies(X_train[col]) for col in X_train], axis=1, keys=X_train.columns)

# Fit the SMOTE using dict as sampling strategy
sm = SMOTE(random_state = 2,  sampling_strategy = {0: 42617, 1: 4261} )
X_train, Y_train = sm.fit_sample(X_train, Y_train)

# Getting the count of target variable class
np.bincount(Y_train)/Y_train.shape[0] * 100

```

{{< boxmd >}}
**Output**
array([90.9104484,  9.0895516])
{{< /boxmd >}}


Please leave comments, if
1. You find anything incorrect.
2. You want to add more information to the topic.
3. You wish to add another example to the topic.
4. You need more details in regards to a specific section.
5. You are unable to execute an example code.
