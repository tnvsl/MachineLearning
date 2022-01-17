import numpy as np
from scipy.stats import mode
import sys

get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib
import matplotlib.pyplot as plt
from scipy.io import loadmat
import time
from helper import *

print('You\'re running python %s' % sys.version.split(' ')[0])

# <h2>k-Nearest Neighbors Implementation in Python</h2>
#
# <p>The goal of implementing your $k$-NN classifier is to build a classifier for face recognition. We have obtained some data, images of faces, for testing your code. The data resides in the file <code>faces.mat</code>, which holds the dataset for our exercises below.</p>

# <p>We will refer to the training vectors as <b>xTr</b> with labels <b>yTr</b>. Our testing vectors are <b>xTe</b> with labels <b>yTe</b>.
# As a reminder, to predict the label or class of an image in <b>xTe</b>, we will look for the <i>k</i>-nearest neighbors in <b>xTr</b> and predict a label based on their labels in <b>yTr</b>. For evaluation, we will compare these labels against the true labels provided in <b>yTe</b>.</p>

# <h3> Visualizing the Data</h3>
#
# <p>Let us take a look at the data. The following script will take the first ten training images from the face dataset and visualize them. Run the code cell to see the visualized data.</p>

# In[2]:


xTr, yTr, xTe, yTe = loaddata("faces.mat")

plt.figure(figsize=(11, 8))
plotfaces(xTr[:9, :])


#
# <h2>Implement k-NN for Facial Recognition</h2>
# <p>The following four project parts will step you through implementing each function necessary to build your facial recognition system.</p>

# <h3>Part 1: Implement <b><code>findknn</code></b> [Graded]</h3>
#
# Implement the function <b><code>findknn</code></b>, which should find the $k$ nearest neighbors ($k \le n$) of a set of vectors within a given training data set. With `xTr` of size $n \times d$ and `xTe` of size $m \times d$, the call of:
# ```python
# [I, D] = findknn(xTr, xTe, k)
# ```
# should result in two matrices `I` and `D`, both of dimensions $k\times m$, where $m$ is the number of input vectors in <code>xTe</code>. The matrix `I[i, j]` is the index of the $i^{th}$ nearest neighbor of the vector `xTe[j, :]`.
#
# So, for example, if we set <code>i = I(1, 3)</code>, then <code>xTr[i, :]</code> is the first nearest neighbor of vector <code>xTe[3, :]</code>. The second matrix `D` returns the corresponding distances. So `D[i, j]` is the distance of `xTe[j, :]` to its $i^{th}$ nearest neighbor.
#
# `l2distance(X, Z)` from the last exercise is readily available to you with the following specification:
# ```python
# """
# Computes the Euclidean distance matrix.
# Syntax: D = l2distance(X, Z)
# Input:
#     X: nxd data matrix with n vectors (rows) of dimensionality d
#     Z: mxd data matrix with m vectors (rows) of dimensionality d
# Output:
#     Matrix D of size nxm
#         D(i, j) is the Euclidean distance of X(i, :) and Z(j, :)
# call with only one input: l2distance(X) = l2distance(X, X).
# """
# ```
#
# One way to use `l2distance()` is as follows:
# 1. Compute distances `D` between `xTr` and `xTe` using `l2distance`.
# 2. Get indices of `k`-smallest distances for each testing point to create the `I` matrix.
# 3. Use `I` to re-order `D` or create `D` by getting the `k`-smallest distances for each testing point.
#
# You may find <a href="https://docs.scipy.org/doc/numpy/reference/generated/numpy.argsort.html"><code>np.argsort(D, axis=0)</code></a> and <a href="https://docs.scipy.org/doc/numpy/reference/generated/numpy.sort.html"><code>np.sort(D, axis=0)</code></a> useful when implementing <code>findknn</code>.

# In[21]:


def findknn(xTr, xTe, k):
    """
    function [indices,dists]=findknn(xTr,xTe,k);

    Finds the k nearest neighbors of xTe in xTr.

    Input:
    xTr = nxd input matrix with n row-vectors of dimensionality d
    xTe = mxd input matrix with m row-vectors of dimensionality d
    k = number of nearest neighbors to be found

    Output:
    indices = kxm matrix, where indices(i,j) is the i^th nearest neighbor of xTe(j,:)
    dists = Euclidean distances to the respective nearest neighbors
    """
    D = l2distance(xTe, xTr)
    m, n = D.shape

    indices = []
    dists = []
    for i in range(m):
        small_index = np.argsort(D[i], axis=0)
        index = small_index[:k]
        distance = D[i, index]
        indices.append(index)
        dists.append(distance)

    indices = np.transpose(np.array(indices))
    dists = np.transpose(np.array(dists))
    return indices, dists

    raise NotImplementedError()

# <p>The following demo samples random points in 2D. If your <code>findknn</code> function is correctly implemented, you should be able to click anywhere on the plot to add a test point. The function should then draw direct connections from your test point to the k  nearest neighbors. Verify manually if your code is correct.</p>

# In[27]:


get_ipython().run_line_magic('matplotlib', 'notebook')
visualize_knn_2D(findknn)

# <p>We can visualize the k=3 nearest training neighbors of some of the test points (Click on the image to cycle through different test points).</p>

# In[28]:


get_ipython().run_line_magic('matplotlib', 'notebook')
visualize_knn_images(findknn, imageType='faces')


# <h3>Part 2: Implement <b><code>accuracy</code></b> [Graded]</h3>
#
# The function <b><code>accuracy</code></b> should compute the accuracy of a classifier. The call of:
# ```python
# result = accuracy(truth, preds)
# ```
# should output the <b>accuracy</b> in variable <code>result</code>. The input variables <code>truth</code> and <code>preds</code> should contain vectors of true and predicted labels respectively.
#
# For example, the call:
# ```python
# accuracy([1, 2, 1, 2], [1, 2, 1, 1])
# ```
# <p>should return an accuracy of 0.75. Here, the true labels are 1,2,1,2 and the predicted labels are 1,2,1,1. So the first three examples are classified correctly, and the last one is wrong -- 75% accuracy.</p>
# <p>You may find the following functions helpful: <code>flatten()</code>, <code>np.mean()</code> and <code>np.abs()</code>.</p>

# In[29]:


def accuracy(truth, preds):
    """
    function output=accuracy(truth,preds)
    Analyzes the accuracy of a prediction against the ground truth

    Input:
    truth = n-dimensional vector of true class labels
    preds = n-dimensional vector of predictions

    Output:
    accuracy = scalar (percent of predictions that are correct)
    """

    diff = np.abs(truth - preds)
    false = np.count_nonzero(diff)
    corr = diff.shape[0] - false
    accuracy = np.float64(corr / diff.shape[0])
    return accuracy
    raise NotImplementedError()


# <h3>Part 3: Implement <b><code>knnclassifier</code></b> [Graded]</h3>
#
# Implement the function <b><code>knnclassifier</code></b>, which should perform `k` nearest neighbor classification on a given test data set. The call:
# ```python
# preds = knnclassifier(xTr, yTr, xTe, k)
# ```
# should output the predictions for the data in <code>xTe</code>, i.e. <code>preds[i]</code> will contain the prediction for <code>xTe[i, :]</code>.
#
# You may find it helpful to use <code>flatten()</code> in the implementation of this function. It will also be useful to  refer back to the mode function you implemented in <a href="https://lms.ecornell.com/courses/1451693/modules/items/16187695">Additional NumPy Exercises</a>.

# In[78]:


def knnclassifier(xTr, yTr, xTe, k):
    """
    function preds=knnclassifier(xTr,yTr,xTe,k);

    k-nn classifier

    Input:
    xTr = nxd input matrix with n row-vectors of dimensionality d
    yTr = n-dimensional vector of labels
    xTe = mxd input matrix with m row-vectors of dimensionality d
    k = number of nearest neighbors to be found

    Output:

    preds = predicted labels, ie preds(i) is the predicted label of xTe(i,:)
    """
    # fix array shapes
    yTr = yTr.flatten()

    ind, dist = findknn(xTr, xTe, k)
    labels = yTr[ind]
    preds = mode(labels)[0][0]
    return preds
    raise NotImplementedError()

# You can compute the actual classification error on the test set by calling
# ```python
# yPreds = knnclassifier(xTr, yTr, xTe, 3)
# accuracy(yTe, yPreds)
# ```

# <h3><b>Part 4: Calculate Accuracy</b></h3>
#
# <p>The following script runs your $k$-nearest neighbor classifier over the faces and digits data set. The faces data set has $40$ classes and the digits data set has $10$. What classification accuracy would you expect from a random classifier?</p>

# In[80]:


print("Face Recognition: (1-nn)")
xTr, yTr, xTe, yTe = loaddata("faces.mat")  # load the data
t0 = time.time()
preds = knnclassifier(xTr, yTr, xTe, 1)
result = accuracy(yTe, preds)
t1 = time.time()
print("You obtained %.2f%% classification acccuracy in %.4f seconds\n" % (result * 100.0, t1 - t0))

# <h3>k-NN Boundary Visualization</h3>
#
# <p>To help give you a visual understanding of how the k-NN boundary is affected by $k$ and the specific dataset, feel free to play around with the visualization below.</p>
# <h4>Instructions:</h4>
# <ol>
#     <li>Run the cell below.</li>
#     <li>Click anywhere in the graph to add a negative class point.</li>
#     <li>Hold down <b>'p'</b> key and click anywhere in the graph to add a positive class point.</li>
#     <li>To increase $k$, hold down <b>'h'</b> key and click anywhere in the graph.</li>
# </ol>

# In[81]:

get_ipython().run_line_magic('matplotlib', 'notebook')
visualize_knn_boundary(knnclassifier)










