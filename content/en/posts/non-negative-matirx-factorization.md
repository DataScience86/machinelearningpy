---
author: "Mohit Sharma"
title:  "Non-Negative Matrix Factorization(NMF) - With Examples in Python"
date: "2020-02-15"
description: "A practical guide about Non-Negative Matrix Factorization(NMF). The article provides examples of how to use NMF for tasks like Topics Modeling, Dimensionality Reduction, Face Detection, Building Recommender Systems, and clustering."
tags:
- Python
- Topic Modeling
- NLP
- NMF
- Clustering
- Recommender Systems
- Face Detection
- Feature Extraction
- Non-Negative Matrix Factorization
- Dimensionality Reduction

categories:
- Machine Learning
- Unsupervised Learning
- Supervised Learning
- Natural Language Processing
libraries:
- mathjax

featured_image: "postfeatureimg/nmf-topicmodel.png"
---

**Non-Matrix Factorization**, aka NMF, is a widely used algorithm for the analysis of non-negative high dimensional data. The algorithm is handy in extracting meaningful features from a non-negative matrix. It was first introduced in 1994 by Paatero and Tapper. The algorithm has been long used for extracting information from chemical systems using data-driven approaches under the name **self-modeling curve resolution**. It got its name as Non-Negative Matrix Factorization after a popular article from Lee and Seung in 1999. The paper discussed **properties of the algorithm and published some simple and useful algorithms for two types of factorizations**.[[<sup>1</sup>]](https://en.wikipedia.org/wiki/Non-negative_matrix_factorization)

<!--more-->

## How does NMF work?
Non-Negative Matrix Factorization is a state of the art technique that is used for extracting important features from a large number of weak and ambiguous attributes. Using these attributes in combination, NMF is able to extract meaningful topics/themes and patterns. For example, NMF is able to produce a context for the word that can occur in different documents.

blood + bank = Blood Bank
<!--more-->
bank + savings = Financial Institution

The algorithm decomposes a matrix **V** into two lower rank matrices **W** and **H** such that when you multiply them, you will get back the original matrix values.

&nbsp;&nbsp;&nbsp;&nbsp; \\(V \approx W \* H\\)

Practically, it can be challenging to recover the same matrix, and so, the algorithm tries to get the numbers as close as possible to the original matrix. Let us look at the below example to understand it better.

1. Creating a 2D matrix with values 12, 24, 36, and 12.

```
import numpy as np
V =  np.array([[12, 24],[ 36, 12]])
print(V)
```

{{< boxmd >}}
**Output**
[[12 24]
 [36 12]]
{{< /boxmd >}}

2. Using NMF decomposition to generate W and H Matrices

```
from sklearn.decomposition import NMF
model = NMF(n_components = V.shape[1], init='random', random_state=0)
W = model.fit_transform(X)
H = model.components_
```

{{< boxmd >}}
 **Output - print(W)**
[[0.36187306 3.22309622]
 [6.05892629 0.54819937]]

**Output - print(H)**
[[5.66230231 1.32023976]
 [3.08739425 7.29802579]]
{{< /boxmd >}}

- **n_components** - It is the number of features you want to select as part of the final solution. It is like asking the algorithm to return the most critical `n` features from the total number of features. If n_components is not set, all features are kept.

- **init** - Here, you need to provide the method that will be used to initialize the procedure.

- **random_state** -  Setting the seed to get reproducible results.

3. Taking the dot product of `W` and `H` to get the original `V` matrix.
```
print(np.dot(W , H))
```

{{< boxmd >}}
**Output**
[[12.00000342 23.99999855]
 [35.99997991 12.0000085 ]]
{{< /boxmd >}}

## Understanding W and H Matrix
![Non-negative matrix factorization figure 1](/images/nmf/matrixfactorization.png)

When NMF used as a Machine Learning algorithm, **W** is the **weight matrix**, and **H** is the **feature Matrix**. In the `W` matrix, each row corresponds to observations, and each column corresponds to a feature. On the other hand, the `H` matrix contains the feature weights relative to each observation. Here, each row corresponds to a feature, and each column corresponds to a column for each column in the original matrix.

## Why Non-Negative?
The algorithm is named as Non-Negative because it returns non-negative value for the feature and weight matrix. Therefore, all features should either be positive or should have zero values.

## How Matrix Factorization is Performed
1. **Gradient descent** - It is one of the most common techniques which is used for matrix factorization. The technique has many variants such as `RMSprop`, `Adadelta`, `Adagrad`, `SGD`, and `Momentum method`.  

### The Gradient Descent Algorithm
{{< boxmd >}}
- Intialize W and H with some random(small) numbers
- for i until specified_iteration:
  - for row, col in V:
    - if V[row][col] > 0:
      - compute error for elements
      - compute gradient descent for error
      - update the values of W and H
    - Compute total error
    - if error < threshold:
      - break
- return W, H.Transpose
{{< /boxmd >}}

2. **Singular value decomposition(SVD)** - SVD is another widely used data reduction technique in Machine Learning. Many international companies like Google, Netflix, and YouTube use this method as a core element in their recommender systems.

## Applications of NMF
1. **Dimensionality Reduction Or Feature Extraction** - Just like principal component analysis, you can use Non-Negative Matrix Factorization(NMF) for dimensionality reduction or feature extraction.

2. **Clustering** - NMF is closely related to the unsupervised clustering algorithm and is widely used for document clustering or topic modeling.

3. **Recommender Systems** - It is a collaborative filtering algorithm based on Non-negative Matrix Factorization and can be used for building a recommender system.

4. **Visual Pattern Recognition** - In recent years, non-negative matrix factorization (NMF) methods have found attention in the computer vision community. The algorithm has shown promising results with face and gesture recognition.

### Example #1 - Feature Extraction Using NMF
We will be using the `digits` dataset for this example. Each data point in the digits dataset represents a collection of `8x8` digit images. There are overall **10 classes with ~180 examples per class**. There are **1797** images with **64** attributes/features for each one.

```
from time import time
import logging
import pylab as pl

from sklearn.decomposition import NMF
from sklearn.datasets import load_digits

import matplotlib.pyplot as plt

digits = load_digits()
plt.gray()
plt.matshow(digits.images[7])
plt.show()
```
![Feature Extraction Using NMF - seven - fig 1](/images/nmf/seven.png)

We are preparing data by reshaping the data using n_samples and n_features.

```
n_samples = len(digits.images)
X = digits.images.reshape((n_samples, -1))
n_features = X.shape[1]
```

From the above dataset, which has 64 features, and we will extract 20 non-negative features using the below code.

```
n_components = 20
print(f"Building NMF model for extracting {n_components} non-negative features")
feature_model = NMF(n_components=n_components, init='nndsvd',\
                    sparseness="components",tol=1e-2).fit(X)

extracted_components = feature_model.components_
```
Using the below code, we will plot the results.

```
n_row, n_col = 4, 4
img = pl.figure(figsize=(1.5 * n_col, 1.5 * n_row))
for i in range(n_row * n_col):
    pl.subplot(n_row, n_col, i + 1)
    pl.imshow(extracted_components[i].reshape((12, 12)), interpolation='nearest')
    pl.xticks(())
    pl.yticks(())
pl.show()
```
![Feature Extraction Using NMF - output - fig 2](/images/nmf/nmffeatureExtract.png)

As of now, the results are not very impressive for NMF with 20 components and selected hyperparameters. However, I encourage you to play around with these parameters and see where you get the desired result.

{{< notice info >}} NMF is a complex algorithm as compared to PCA, because all its components are trained at the same time, and they are also dependent on each other. Thus, if you add another component, the first components which were generated originally may change. Also, we cannot match the variance explained by each component.{{< /notice>}}

#### How to choose the number of components in NMF
It is challenging to figure how many components are requires as each component cannot be matched to the variance it explains. However, what you can do is fit a new instance of NMF for a number of components, and compare the total variance explained.

```
from sklearn.metrics import explained_variance_score

def get_optimal_components(components = [1,2,3,4], X_train_nmf = X, X_test_nmf =  X):
    variance_train = []
    variance_train = []
    for k in components:
        nmf = NMF(n_components=k).fit(X_train_nmf)
        variance_train.append(get_score(nmf, X_train_nmf))
        variance_train.append(get_score(nmf, X_test_nmf))
    return(list(zip(components,variance_train)), list(zip(variance_train)))

def get_score(model, data, scorer = explained_variance_score):
    """ Estimate performance of the model on the data """
    prediction = model.inverse_transform(model.transform(data))
    return scorer(data, prediction)
```
**Note:** We did not create the test and train dataset. Thus, we will be passing our actual dataset as part of both train and test arguments.

Calling these functions on our dataset of digits, which we have stored as X will provide the variance explained by the components.

```
train_var, test_var = get_optimal_components(components = [20, 30, 40, 50],\
                                             X_train_nmf = X, \
                                             X_test_nmf =  X)
print(train_var)
```
{{< boxmd >}}
**Output**
[(20, 0.6392660812902782), (30, 0.6392660812902782), (40, 0.7276944542659505), (50, 0.7276944542659505)]
{{< /boxmd >}}
**Looks like with 50 components we can explain approx 73% of the variance.**

### Example #2 - Topic Modeling using NMF
In natural language processing, topic modeling is used to identify and group similar topics under common themes. Intuitively, documents belonging to a particular topic will have similar words. For this example, we will be using the `news summary` dataset, which has 4514 summaries.

We will be loading the following Python packages, these we will be using to do the necessary cleaning and build **NMF topic models**.

```
import re
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
stop = stopwords.words('english')
```

Reading the dataset using `read_csv` function from the `pandas` package.
```
news_summary = pd.read_csv("M:/GitHub/data/news-summary/news_summary.csv", encoding='latin-1')
news_summary.head()
```

Let's look at the top 5 news article.

```
news_summary["ctext"].head()
```
{{< boxmd >}}
**Output**
0    The Daman and Diu administration on Wednesday ...
1    From her special numbers to TV?appearances, Bo...
2    The Indira Gandhi Institute of Medical Science...
3    Lashkar-e-Taiba's Kashmir commander Abu Dujana...
4    Hotels in Mumbai and other Indian cities are t...
{{< /boxmd >}}

Next, we will be doing the text cleaning. As part of text cleaning, we do the following tasks.

- Converting text to lower
- Remove words with character length less than 3
- Remove numbers from the text
- Remove punctuations
- Remove whitespaces

```
# Converting all text to lower
news_summary["ctext"] = news_summary["ctext"].apply(lambda x: str(x).lower())

# removing words with character length less than 3
news_summary["ctext"] = news_summary["ctext"].apply(lambda x: re.sub(r'\b\w{1,3}\b', ' ', x))

# removing numbers
news_summary["ctext"] = news_summary["ctext"].apply(lambda x:re.sub(r'\d+', "", x))

# Removing punctuations
news_summary["ctext"] = news_summary["ctext"].apply(lambda x: re.sub(r'[^\w\s]','',x))

# removing whitespaces
news_summary["ctext"] = news_summary["ctext"].apply(lambda x: re.sub('\s+', ' ', x).strip())
```
Now that we have cleaned the text, we can generate the TF-IDF weighted document-term matrix by using `TfidfVectorizer`. The idea is to give more weights to the more **important** terms. After building the tf-idf matrix, we will map each token to their respective tf-idfs.

```
vectorizer = TfidfVectorizer(min_df=5, analyzer='word', ngram_range=(1, 2))
vz = vectorizer.fit_transform(list(data['Snippet']))

# Mapping tokens to there tf-idf
tf_idf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
tf_idf = pd.DataFrame(columns=['tfidf']).from_dict(dict(tfidf), orient='index')

tf_idf.columns = ['tfidf']

print( f"Printing the top 5 words from TF-IDF weighted document-term matrix \n {tfidf.head()}" )
```
{{<boxmd>}}
**Output**

Created 4514 X 12753 TF-IDF-normalized document-term matrix

Printing the top 5 words from TF-IDF weighted document-term matrix
             tfidf
aadhaar  5.777574
aadhar   6.930254
aadmi    4.696662
aamir    5.543959
aaps     7.623401
{{</boxmd>}}

Finally, we use NMF to identify and generate 8 topics.

```
from sklearn.decomposition import NMF
k = 8

# apply the model and extrcating the two smaller matrices
nmf_8 = NMF(n_components = k, random_state = 10, l1_ratio = .8, init = 'nndsvd', verbose = True, max_iter = 100, tol=0.001)

W = nmf_8.fit_transform(vz)
H = nmf_8.components_
print(f"Actual Number of Iterations: {nmf_8.n_iter_}")
```

Let's print the topics and also see the top 10 words which mostly describe these topics.

```
no_top_words = 20
no_topics_display = 8

for topic_idx, topic in enumerate(H[:no_topics_display]):
    print("Topic %d:"% (topic_idx))
    print(" | ".join([terms[i]
                    for i in topic.argsort()[:-no_top_words - 1:-1]]))
```
{{< boxmd >}}
**Output**
Topic 0:
film | actor | that | khan | with | films | about | kapoor | salman | says
Topic 1:
said | that | will | have | government | from | with | this | they | their
Topic 2:
police | woman | were | said | arrested | accused | station | incident | from | allegedly
Topic 3:
kohli | cricket | india | test | team | australia | captain | series | virat | england
Topic 4:
party | congress | minister | election | pradesh | uttar | assembly | chief | kejriwal | modi
Topic 5:
court | supreme | justice | case | high | bench | that | government | apex | order
Topic 6:
pakistan | army | kashmir | china | india | indian | chinese | security | pakistani | jammu
Topic 7:
nitish | kumar | bihar | lalu | modi | tejashwi | yadav | minister | alliance | prasad
{{< /boxmd >}}

We will now draw a bar plot showcasing the word importance for topic number 4.

```
def plot_top_term_weights( terms, H, topic_index, top ):
    top_indices = np.argsort( H[topic_index,:] )[::-1]
    top_terms = []
    top_weights = []
    for term_index in top_indices[0:top]:
        top_terms.append( terms[term_index] )
        top_weights.append( H[topic_index,term_index] )
    top_terms.reverse()
    top_weights.reverse()
    fig = plt.figure(figsize=(13,8))
    ypos = np.arange(top)
    ax = plt.barh(ypos, top_weights, align="center", color="orange",tick_label=top_terms)
    plt.xlabel("Term Importance - Based on Weights",fontsize=14)
    plt.tight_layout()
    plt.show()
plot_top_term_weights(terms, H, 4, 10)
```
![Topic modeling Using NMF - output - fig 1](/images/nmf/topic_modeling_nmf_Fig1.png)



### Example #3 - Building Recommender Systems using NMF

{{< notice info >}} We can use NMF to generate recommendations for people by reconstructing the matrix using `W` and `H` matrix. During the reconstruction of matrix `V` the algorithm also assigns values to the unknown values which we have filled with zeros in our case. Through the use of latent features, certain weights are assigned to the movies in the column. These values can then be arranged in descending order to determine what all movies should be recommended to the customers.{{< /notice >}}

We will be using [MovieLens](https://grouplens.org/datasets/movielens/) for this example. The dataset consists of 100,000 ratings and 3600 tags, which are applied to around 9000 movies by 600+ users.

For this example, we will be using `sklearn`, `numpy`, and `pandas` Python module. The downloaded zip file comes with four csv files and one readme text file. Although for this example, we only need the Ratings dataset, we include Movies to get the information on movie names and genres as well.

In the below code, we are doing the following:
- Importing the data manipulation packages like `numpy` and `pandas`.
- setting the display setting to fit my requirements.
- Reading movies and rating dataset
- Merging the two datasets to get the information in one file.

```
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1000)

ratings = pd.read_csv("ratings.csv")
movies = pd.read_csv("movies.csv")
movie_ratings = pd.merge(movies, ratings, on = "movieId")
movie_ratings.to_csv("movie_ratings.csv", index = True)
movie_ratings.head(5)
```

{{< boxmd >}}
**Output**
   movieId             title                                       genres  userId  rating   timestamp
0        1  Toy Story (1995)  Adventure|Animation|Children|Comedy|Fantasy       1     4.0   964982703
1        1  Toy Story (1995)  Adventure|Animation|Children|Comedy|Fantasy       5     4.0   847434962
2        1  Toy Story (1995)  Adventure|Animation|Children|Comedy|Fantasy       7     4.5  1106635946
3        1  Toy Story (1995)  Adventure|Animation|Children|Comedy|Fantasy      15     2.5  1510577970
4        1  Toy Story (1995)  Adventure|Animation|Children|Comedy|Fantasy      17     4.5  1305696483

{{< /boxmd >}}

From the above output, you can see that users with ID 1, 5, 7, 15, and 17 all saw the Toy Story movie. Among these users, the user with ID 15 gave this movie less than average rating, whereas others gave higher ratings.

The dataset looks good, but we want one userId per row and one movie per column. We can use `pivot` function to rearrange the dataset in this particular format.

```
movie_ratings_pivot = movie_ratings.pivot(index = 'userId', columns ='movieId', values = 'rating')
movie_ratings_pivot.head()
```

{{< boxmd >}}
**Output**
movieId  1       2       3       4       5       6       7       8       9       10      ...  193565  193567
userId                                                                                   ...                                                                                
1           4.0     NaN     4.0     NaN     NaN     4.0     NaN     NaN     NaN     NaN  ...     NaN     NaN    
2           NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN  ...     NaN     NaN    
3           NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN  ...     NaN     NaN    
4           NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN  ...     NaN     NaN    
5           4.0     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN  ...     NaN     NaN    

[5 rows x 9724 columns]
{{< /boxmd >}}

Last, we need to replace NaN value with Zero and convert the data frame into the numpy array.

```
movie_ratings_array = movie_ratings_pivot.fillna(0).as_matrix()
movie_ratings_array.head()
```

{{< boxmd >}}
**Output**
array([[4. , 0. , 4. , ..., 0. , 0. , 0. ],
       [0. , 0. , 0. , ..., 0. , 0. , 0. ],
       [0. , 0. , 0. , ..., 0. , 0. , 0. ],
       ...,
       [2.5, 2. , 2. , ..., 0. , 0. , 0. ],
       [3. , 0. , 0. , ..., 0. , 0. , 0. ],
       [5. , 0. , 0. , ..., 0. , 0. , 0. ]])

{{< /boxmd >}}

`sklearn.decompositon` has a function `NMF`, which can be used to build to fit the final model. We are going to use 200 latent features with both l1 and l2 regularization to generate the `W` and `H` matrix.

```
from sklearn.decomposition import NMF

# Building model - Basic NMF
feature_model = NMF(n_components = 200, init='random', solver = 'mu',\
                    l1_ratio = 0.5, verbose = True, random_state= 40)
W = feature_model.fit_transform(movie_ratings_array)
H = feature_model.components_
```

Now that we have both a `W` and `H` matrix. We can use `np.dot()` function to reconstruct the `V` matrix. Also, in the dataset, movieId's are not continuous numbers being assigned. So, we need to replace columns names with the actual Id's.

```
reconstructed_V = pd.DataFrame(W.dot(H))
reconstructed_V.columns = movie_ratings.movieId.unique()
```

Now we will filter out movies which users have already watched, and then we need to select the top 10 recommended movies by sorting the values of the `Score` column in descending order. Let us see the top 10 recommended movies for the user with id = 1.

```
i= 0
temp_test = reconstructed_V[reconstructed_V.index == i]
temp_test = temp_test.T
temp_test.reset_index(level=0, inplace=True)
temp_test.columns = ["movieId", "Score"]
movieid_already_watched = ratings[ratings.userId == i+1]["movieId"]
keep_list = set(temp_test["movieId"]).difference(set(movieid_already_watched))
temp_test = temp_test[temp_test['movieId'].isin(keep_list)]
temp_test_recom = temp_test.sort_values(ascending = False, by = "Score")
temp_test_recom[0:10]
```

{{< boxmd >}}
**Output**
      movieId     Score
1066     1387  0.390069
2492     3328  0.322647
1444     1968  0.301258
784      1027  0.297047
1745     2342  0.293182
2274     3019  0.288483
2799     3745  0.287456
946      1248  0.286723
2416     3213  0.280908
1791     2391  0.274578
{{< /boxmd >}}

To get the movie names or tags, you can look into movies or tags dataset. We can also loop through all the userId and save the top 10 recommended movies in a dictionary.

It is a good idea to check if the reconstructed matrix had closer results for the movies which were already rated by the user. Below we are comparing the actual and reconstricted ratings for the user with userId (1). If the model is right, then the expectation is to have the numbers in the reconstructed matrix closer to actual ratings.

```
Actual_rating = movie_ratings.loc[movie_ratings.userId == 1, ["movieId","rating"]].sort_values(by = "rating", ascending = False).head()
Predicted_Rating = temp_test.loc[temp_test.movieId.isin(Actual_rating.movieId)].sort_values(by = "movieId")

final_out = pd.merge(Actual_rating, Predicted_Rating, on = "movieId")
final_out.columns = ["movieId","Actual_Rating", "Pred_Rating"]
final_out
```

{{< boxmd >}}
**Output**
   movieId  Actual_Rating  Pred_Rating
0     5060            5.0     5.051744
1     2872            5.0     4.930985
2     1291            5.0     5.220716
3     1298            5.0     4.958132
4     2948            5.0     5.040785
{{< /boxmd >}}


### Example #4 - Using NMF For Face Detection
For this example, we are using [Olivetti faces dataset from AT&T](https://scikit-learn.org/stable/datasets/index.html#olivetti-faces-dataset) (classification). The dataset is available in `sklearn.datasets` module.

First, we will load the required Python packages and setting some parameters.

```
from numpy.random import RandomState
import matplotlib.pyplot as plt
import pylab as pl

from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import NMF, PCA

n_row, n_col = 2, 3
n_components = n_row * n_col
image_shape = (64, 64)
rs = RandomState(0)
```

We will now load the dataset and perform centering.

```
dataset = fetch_olivetti_faces(shuffle=True, random_state=rs)
faces = dataset.data
n_samples, n_features = faces.shape

# global centering
faces_centered = faces - faces.mean(axis=0)

# local centering
faces_centered -= faces_centered.mean(axis=1).reshape(n_samples, -1)
```

Let us view one of the sample images.

```
def plot_gallery(title, images):
    pl.figure(figsize=(2. * n_col, 2.26 * n_row))
    pl.suptitle(title, size=16)
    for i, comp in enumerate(images):
        pl.subplot(n_row, n_col, i + 1)
        vmax = max(comp.max(), -comp.min())
        pl.imshow(comp.reshape(image_shape), cmap=pl.cm.gray,
                  interpolation='nearest',
                  vmin=-vmax, vmax=vmax)
        pl.xticks(())
        pl.yticks(())
    pl.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)

plot_gallery("First centered Olivetti faces", faces_centered[:n_components])
```

![Non-negative matrix factorization(NMF) face detection example figure 1](/images/nmf/FacedetectionFig1.png)

Next, we build the NMF model to estimate the faces and visualize the final results.

```
estimator.fit(faces)
components_ = estimator.components_
plot_gallery("First centered Olivetti faces", components_[:n_components])
```
![Non-negative matrix factorization(NMF) face detection final result figure 2](/images/nmf/FacedetectionFig2.png)

{{< alert theme="info" >}}
The above example was originally presented in Python Documentation <a href="https://scikit-learn.org/stable/auto_examples/decomposition/plot_faces_decomposition.html">Link</a>
{{< /alert >}}

### Example #5 - Building Clusters Using NMF
NMF is an efficient method that provides a powerful technique for class discovery. The algorithm has advantages over other methods, such as hierarchical clustering. For building clusters, we will be using [bignmf](https://pypi.org/project/bignmf/). The following example that illustrates the typical usage of the algorithm is borrowed from [HERE](https://pypi.org/project/bignmf/).

```
from bignmf.models.jnmf.integrative import IntegrativeJnmf
from bignmf.datasets.datasets import Datasets

Datasets.list_all()
data_dict = {}
data_dict["sim1"] = Datasets.read("SimulatedX1")
data_dict["sim2"] = Datasets.read("SimulatedX2")

k = 3
iter =100
trials = 50
lamb = 0.1

# Runs the model
model = IntegrativeJnmf(data_dict, k, lamb)
model.run(trials, iter, verbose=0)
print(model.error)

# Clusters the data
model.cluster_data()
print(model.h_cluster)

#Calculates the consensus matrices
model.calc_consensus_matrices()
print(model.consensus_matrix_w)
```


Please leave comments, if
1. You find anything incorrect.
2. You want to add more information to the topic.
3. You wish to add another example to the topic.
4. You need more details in regards to a specific section.
5. You are unable to execute an example code.
