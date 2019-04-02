"""
A Complete Guide to Matrix Notation and Linear Regression
=========================================================

"""


######################################################################
# Let’s really understand matrix notation in context of linear regression,
# from the ground up.
# 
# Linear Regression finds the best line, or *hyperplane* :math:`\hat{y}`
# in higher dimension, or generally a function :math:`f`:
# 
# .. math::  \hat{y} = f(x) = wx 
# 
# that fits the whole data. This is just a dot product between vector
# :math:`w` and a data point :math:`x` in :math:`d` dimension:
# 
# .. math::  \hat{y} = w_0 + w_1x_1 + w_2x_2 + ... + w_dx_d 
# 
# Notice that we use :math:`w_0` as an intercept term, and thus we need to
# add a dummy dimension with value of “1” (:math:`x_0`) for all data
# points :math:`x`. Thus, :math:`x` here is on :math:`d+1` dimension.
# Think of it as the y-intercept term :math:`c` in 2-dimension
# (:math:`y = mx + c`).
# 
# Another way to look at this is that :math:`f(x)` transforms a data point
# :math:`x` on :math:`d+1` dimension into a predicted scalar value
# :math:`\hat{y}` that is close to target :math:`y`:
# 
# .. math::
# 
# 
#    \begin{bmatrix}
#    x_0 \\
#    x_1 \\
#    \vdots \\
#    x_d 
#    \end{bmatrix}
#    \xrightarrow{f}
#    \hat{y}
#    \approx y
# 
# 


######################################################################
# The Sum of Squared Error Loss
# -----------------------------
# 
# The best way to solve this is to find :math:`w` that minimizes the **sum
# of squared errors (SSE)**\ :math:`^\dagger`, or the “error” between all
# of predicted value :math:`\hat{y}^i` and the target :math:`y^i` of
# :math:`i^{th}` data point for :math:`i = 1` to :math:`n`, writing this
# as a loss function :math:`L(w)`:
# 
# .. math::
# 
#     
#    L(w) = \sum_{i=1}^{n} \left( y^i - \hat{y}^i \right)^2 = \sum_{i=1}^{n} \left( y^i - wx^i \right)^2
# 
# From now on we refer to a data point (d+1 vector) as :math:`x^i` and its
# corresponding target (scalar) as :math:`y^i`.
# 
# Surprisingly, the SSE loss is not from someone’s intuition, but it’s
# from the assumption that there is **Gaussian noise in our observation**
# of the underlying linear relationship. We will show how this leads to
# SSE loss later, but first let’s visualize what we’re trying to do.
# 
# .. raw:: html
# 
#    <center>
# 
# |image0|
# 
# .. raw:: html
# 
#    </center>
# 
# 1. There is a true line, the true linear relationship that we want to
#    discover (blue line).
# 2. The data points are then observed through noise **deviating from that
#    line, with Gaussian distribution**.
# 3. Suppose we predict a random line (red), not the best one yet.
# 4. We calculate the distance or difference between the predicted points
#    (along the line) and the actual data points. This is then **squared
#    and sum up to get sum of squared error**.
# 
# ..
# 
#    Linear regression is the method to get the line that fits the given
#    data with the minimum sum of squared error.
# 
# *Note: I know it’s confusing for the first time, but you’ll get used to
# using superscript for indexing data points…*
# 
# .. |image0| image:: imgs/img_lr_objective.png
# 
# 


######################################################################
# How to Find the Optimal Solution
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# An optimal solution (:math:`w`) for this equation can be found either
# using *closed-form solution* or via iterative methods like gradient
# descent.
# 
# **A closed-form solution** means we figure out the formula for
# :math:`w = ...`. Implementing that formula in a program directly solves
# the problem. The thing is you have to come up with the correct one
# yourself, by hand.
# 
# Do you remember how to find a minimum (or maximum) value for a function?
# We take the derivative of the function above with respect to :math:`w`,
# set it to zero, and solve for the :math:`w` in terms of other
# parameters. This is like taking a single jump to the optimal value. We
# do all the hard work for computers.
# 
# Luckily we can do this for linear regression, but not all loss functions
# be solved this way, actually, only a few. In those cases, we use
# **iterative methods like gradient descent** to search for the solution.
# In contrast to closed-form solution, we do not jump directly to the
# optimal answer, instead, we take many steps that lead us near to where
# the optimal answer lives.
# 
# Next let’s derive the closed-form solution for linear regression. In
# order to do that efficiently, we need some matrix notations.
# 


######################################################################
# Going into Matrix Notation
# --------------------------
# 
# Writing things down in matrix notation makes things much faster in
# NumPy. **But it’s not easy to read matrix notation, especially if you
# study machine learning on your own.** There’re things like dot product,
# matrix multiplication, transpose and stuff that you need to keep track
# of in your head. If you’re starting out, then please write them on
# papers, drawing figures as needed to make you understand. It really pays
# off.
# 
# On top of that, these few **key standards** will make our lives with
# linear algebra easier:
# 
# 1. Always a *column* vector
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# When you see standalone vectors in a matrix notation formula, assumes
# it’s a column vector. E.g.,
# 
# .. math::
# 
#    x = 
#    \begin{bmatrix}
#    1 \\
#    2 \\
#    3 \\
#    \end{bmatrix}
# 
# and so its transpose is a row vector,
# 
# .. math::
# 
#    x^T = 
#    \begin{bmatrix}
#    1 & 2 & 3
#    \end{bmatrix}
# 
# Likewise, you should try to make the final result of matrix operation to
# be a column vector.
# 
# Note that the NumPy vector created by ``np.zeros``, ``np.arange``, etc.,
# is not really a column vector. It has only one dimension ``(N,)``. So,
# you cannot transpose it directly (``x.T`` still gives you ``x``.) To
# convert it to a column vector, we use ``x.reshape(N,1)`` or
# ``x[:, None]``.
# 


######################################################################
# 2. Feature matrix :math:`X` is rows of data points
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# Our data points :math:`x^i` are on :math:`d+1` dimension, and there is
# :math:`n` of them, we store them all in a 2-d matrix :math:`X`:
# 
# .. math::
# 
# 
#    X = \begin{align}
#    \underset{n\times d}
#    {\begin{bmatrix}
#    \longleftarrow & x^1 & \longrightarrow \\
#    \longleftarrow & x^2 & \longrightarrow \\
#    & \vdots & \\
#    \longleftarrow & x^n & \longrightarrow \\
#    \end{bmatrix}}
#    \end{align}
# 
# Each row in :math:`X` is a row vector for each data point. Also note
# that we use uppercase letter for matrix.
# 


######################################################################
# 3. Again, :math:`w` is a column vector
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# 
# Like the first point, our :math:`w` will be :math:`d+1` dimension column
# vector with w_0 as an intercept term:
# 
# .. math::
# 
# 
#    w = 
#    \begin{bmatrix}
#    w_0 \\
#    w_1 \\
#    \vdots \\
#    w_d 
#    \end{bmatrix}
# 
# 


######################################################################
# 4. Dot products of rows in matrix :math:`X` with vector :math:`w` is :math:`Xw`
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# Sometimes we want the dot product of each row in matrix with a vector:
# 
# .. math::
# 
# 
#    Xw = \underset{n\times (d+1)}
#    {
#        \begin{bmatrix}
#        \longleftarrow & x^1 & \longrightarrow \\
#        \longleftarrow & x^2 & \longrightarrow \\
#        & \vdots & \\
#        \longleftarrow & x^n & \longrightarrow \\
#        \end{bmatrix}
#    }
#    \underset{(d+1) \times 1}
#    {
#        \begin{bmatrix}
#        \uparrow \\
#        w \\
#        \downarrow
#        \end{bmatrix}
#    }
#    =
#    \begin{bmatrix}
#    x^1w \\
#    x^2w \\
#    \vdots \\
#    x^nw
#    \end{bmatrix} 
# 
# given that :math:`X` contains rows of vectors we want to dot product
# with.
# 
# Interestingly, this gives us a column vector of our predictions
# :math:`\hat{y}`:
# 
# .. math::
# 
# 
#    \begin{bmatrix}
#    x^1w \\
#    x^2w \\
#    \vdots \\
#    x^nw
#    \end{bmatrix} =
#    \begin{bmatrix}
#    \hat{y}^1 \\
#    \hat{y}^2 \\
#    \vdots \\
#    \hat{y}^n \\
#    \end{bmatrix} = \hat{y}
# 
# It’s also good to remind yourself that it sums along dimension of
# :math:`x^i` and :math:`w`:
# 
# .. math::
# 
# 
#    Xw =
#    \begin{bmatrix}
#    \sum_{j=0}^{d} x_j^1w_j \\
#    \sum_{j=0}^{d} x_j^2w_j \\
#    \vdots \\
#    \sum_{j=0}^{d} x_j^nw_j \\
#    \end{bmatrix} 
# 
# 


######################################################################
# 5. Sum of Squared is :math:`x^Tx`
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# This is a useful pattern to memorize. Sometimes we want the sum of
# squared of each element in arbitrary d-dimension vector :math:`x`:
# 
# .. math::
# 
# 
#    \sum_{j=1}^{d} x_i^2
# 
# which is simply :math:`x^Tx`:
# 
# .. math::
# 
# 
#    x^Tx = 
#    \begin{bmatrix}
#    x_1 & ... & x_d
#    \end{bmatrix}
#    \begin{bmatrix}
#    x_1 \\
#    \vdots \\
#    x_d 
#    \end{bmatrix}
#    = \sum_{j=1}^{d} x_i^2
# 
# Notice that the result of :math:`x^Tx` is scalar, e.g., a number.
# 
# In fancy term,
# :math:`{\left\lVert x \right\rVert} = \sqrt{\sum_{j=1}^{d} x_i^2}` is
# L2-norm (or Euclidean norm) of :math:`x`. So we can write sum of squared
# as :math:`{\left\lVert x \right\rVert}^2 = \sum_{j=1}^{d} x_i^2`. For
# now, let’s not care what norm actually means.
# 


######################################################################
# Writing SSE Loss in Matrix Notation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# Now you’re ready, let’s write the above SSE loss function in matrix
# notation. If you look at :math:`L(w)` closely, it’s a sum of squared of
# vector :math:`y - \hat{y}`. This means we can kick-off by applying our
# fifth trick:
# 
# .. math::
# 
# 
#    L(w) = {\left\lVert y - \hat{y} \right\rVert}^2
# 
# 


######################################################################
# Next we just have to find :math:`y - \hat{y}`. First we encode target
# :math:`y` in a long column vector of shape ``[n, 1]``:
# 
# .. math::
# 
# 
#    y = 
#    \begin{bmatrix}
#    y^1 \\
#    y^2 \\
#    \vdots \\
#    y^n 
#    \end{bmatrix}
# 
# (Remember that we use superscript for indexing the :math:`i^{th}` data
# point.)
# 
# Next we encode each of our predicted values (:math:`\hat{y}^i`) in a
# column vector :math:`\hat{y}`. Since :math:`\hat{y}^i` is a dot product
# between :math:`w` and each of :math:`x^i`, we can apply
# `4 <#4.-Dot-products-of-rows-in-matrix-$X$-with-vector-$w$-is-$Xw$>`__:
# 


######################################################################
# .. math::
# 
# 
#    \begin{align}
#    y - \hat{y} &= 
#    \begin{bmatrix}
#    y^1 \\
#    y^2 \\
#    \vdots \\
#    y^n 
#    \end{bmatrix}
#    - \begin{bmatrix}
#    \hat{y}^1 \\
#    \hat{y}^2 \\
#    \vdots \\
#    \hat{y}^n 
#    \end{bmatrix} && \text{(Error between target and predicted)} \\ &=
#    \begin{bmatrix}
#    y^1 \\
#    y^2 \\
#    \vdots \\
#    y^n 
#    \end{bmatrix}
#    - \begin{bmatrix}
#    x^1w \\
#    x^2w \\
#    \vdots \\
#    x^nw
#    \end{bmatrix} 
#    && \text{(Predicted is a dot product of $w$ and each of data point $x^i$)} \\ &=
#    \underset{n\times 1}
#    {
#        \begin{bmatrix}
#        y^1 \\
#        y^2 \\
#        \vdots \\
#        y^n 
#        \end{bmatrix}
#    }
#    - \underset{n\times (d+1)}
#    {
#        \begin{bmatrix}
#        \longleftarrow & x^1 & \longrightarrow \\
#        \longleftarrow & x^2 & \longrightarrow \\
#        & \vdots & \\
#        \longleftarrow & x^n & \longrightarrow \\
#        \end{bmatrix}
#    }
#    \underset{(d+1)\times 1}
#    {
#        \begin{bmatrix}
#        \uparrow \\
#        w \\
#        \downarrow
#        \end{bmatrix}
#    } && \text{(Separate them out)} \\ &=
#    y - Xw && \text{(Encode in matrix/vector form)}
#    \end{align}
# 
# 


######################################################################
# Putting all together we get our loss function for linear regression:
# 
# .. math::
# 
# 
#    L(w) = {\left\lVert y - Xw \right\rVert}^2
# 
# In NumPy code, we can compute :math:`L(w) = (y - Xw)^T(y - Xw)`.
# 
# There’s no intuitive way to come up with this nice formula the first
# time you saw it. You have to work it out and put things together
# yourself. Then you’ll start to memorize the pattern and it’ll become
# easier.
# 


######################################################################
# Deriving a Closed-form Solution
# -------------------------------
# 


######################################################################
# To do that, we’ll take derivative of :math:`L(w)` with respect to
# :math:`w`, set to zero and solve for :math:`w`.
# 
# Writing matrix notation is already hard, taking derivative of it is even
# harder. I recommend writing out partial derivatives to see what happens.
# For :math:`L(w) = L_w`, we have to take derivative with respect to each
# dimension of :math:`w`:
# 
# .. math::
# 
# 
#    \nabla L_w = 
#    \begin{bmatrix}
#    \frac{\partial L}{\partial w_0} \\
#    \frac{\partial L}{\partial w_1} \\
#    \vdots \\
#    \frac{\partial L}{\partial w_d} \\
#    \end{bmatrix} 
#    =
#    \begin{bmatrix}
#    \frac{\partial \sum_{i=1}^{n} \left( y^i - wx^i \right)^2}{\partial w_0} \\
#    \frac{\partial \sum_{i=1}^{n} \left( y^i - wx^i \right)^2}{\partial w_1} \\
#    \vdots \\
#    \frac{\partial \sum_{i=1}^{n} \left( y^i - wx^i \right)^2}{\partial w_d}
#    \end{bmatrix} 
#    =
#    \begin{bmatrix}
#    \frac{\partial \sum_{i=1}^{n} \left( y^i - wx^i \right)^2}{\partial w_0} \\
#    \frac{\partial \sum_{i=1}^{n} \left( y^i - wx^i \right)^2}{\partial w_1} \\
#    \vdots \\
#    \frac{\partial \sum_{i=1}^{n} \left( y^i - wx^i \right)^2}{\partial w_d}
#    \end{bmatrix} 
#    =
#    \underset{(d+1) \times 1}
#    {
#    \begin{bmatrix}
#    -2\sum_{i=1}^{n} x^i_0 \left( y^i - wx^i \right) \\
#    -2\sum_{i=1}^{n} x^i_1 \left( y^i - wx^i \right) \\
#    \vdots \\
#    -2\sum_{i=1}^{n} x^i_d \left( y^i - wx^i \right) \\
#    \end{bmatrix}
#    }
# 
# 


######################################################################
# Looks like we might be able to apply our fourth point (:math:`Xw`, but
# in this case :math:`w` is :math:`(y - Xw)`. But unlike our fourth point,
# we now sum along data points (:math:`n`) instead of dimensions
# (:math:`d`). For this, we want each row of :math:`X` to be one given
# dimension along all data points instead of one data point with all
# dimensions, and thus we use :math:`X^T` instead of :math:`X`. Finally,
# here’s the full derivative in matrix notation:
# 
# .. math::
# 
# 
#    \nabla L_w = -2X^T(y-Xw)
# 
# 


######################################################################
# Setting to zero and solve:
# 
# .. math::
# 
# 
#    \begin{align}
#    0 &= -2X^T(y-Xw) \\
#    &= X^T(y-Xw)     \\ 
#    &= X^Ty - X^TXw 
#    \end{align}
# 
# Move :math:`X^TX` to other side and we get a closed-form solution:
# 
# .. math::
# 
# 
#    \begin{align}
#    X^TXw &= X^Ty    \\
#    w &= (X^TX)^{-1}X^Ty
#    \end{align}
# 
# 


######################################################################
# In NumPy, this is:
# 
# .. code:: python
# 
#    w = np.linalg.inv(X.T @ X) @ X @ y
# 


######################################################################
# A NumPy Example
# ---------------
# 

import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1)


######################################################################
# We will create a fake dataset from the underlying equation
# :math:`y = 2x + 7`:
# 

def true_target(x):
  return 2*x + 7


######################################################################
# In practical settings, there is no way we know this exact equation. We
# only get **observed** targets, and there’s some **noise** on it. The
# reason is that it’s impossible to measure any data out there in the
# world perfectly:
# 

def observed_target(x):
  """Underlying data with Gaussian noise added"""
  normal_noise = np.random.normal() * 3
  return true_target(x) + normal_noise


######################################################################
# Creating data points
# ~~~~~~~~~~~~~~~~~~~~
# 


######################################################################
# Next, make 50 data points, observations and targets:
# 

N = 50

# Features, X is [1,50]
X = np.random.rand(N).reshape(N, 1) * 10

# Observed targets
y = np.array([observed_target(x) for x in X]).reshape(N, 1)


######################################################################
# Adding dummy dimension term to each :math:`x^i`:
# 

# Append 1 for intercept term later
X = np.hstack([np.ones((N, 1)), X])


######################################################################
# Note that it **doesn’t matter** here whether we add it to the front or
# back, it will simply reflect correspondingly in our solution :math:`w`.
# 


######################################################################
# Visualize our data points with respect to the true line
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 

# For plotting
features = X[:,1:] # exclude the intercept for plotting
target = y
true_targets = true_target(X[:,1:])

plt.scatter(features, target, s=10, label='Observed data points')
plt.plot(features, true_targets, c='blue', label='True target line y = 2x + 7', alpha=0.3)

plt.xlabel('Feature')
plt.ylabel('Target')
plt.legend(loc='best')
plt.title('True and observed data points')
plt.show()


######################################################################
# Compute a closed-form solution
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 


######################################################################
# Our goal is to get the line that is closest to that true target (blue)
# line as possible, without the knowledge of its existence. For this we
# use linear regression to fit observed data points by following the
# formula from the previous section:
# 

w = np.linalg.inv(X.T @ X) @ X.T @ y


######################################################################
# To predict, we compute :math:`\hat{y} = xw` for each data point
# :math:`x^i`. Here we predict the training set (``X``) itself:
# 

predicted = X @ w # y_hat


######################################################################
# To predict a set of new points, you just make it the same format as
# ``X``, e.g., rows of data points.
# 


######################################################################
# Visualize best fit line vs. true target line
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 

plt.scatter(features, target, s=10, label='Data points')
plt.plot(features, true_targets, c='blue', label='True target line', alpha=0.3)
plt.plot(features, predicted, c='red', label='Best fit line')

plt.xlabel('Feature')
plt.ylabel('Target')
plt.legend(loc='best')
plt.show()


######################################################################
# That’s pretty close.
# 


######################################################################
# Understanding the result
# ~~~~~~~~~~~~~~~~~~~~~~~~
# 
# And our :math:`w` is:
# 

print(w)


######################################################################
# Since we append ones in front of each data point :math:`x`, ``w[0]``
# will be the intercept term and ``w[1]`` will be the slope. So our
# predicted line will be in the format of ``y = w[1]*x + w[0]``. Recall
# the *true* equation :math:`y = 2x + 7`, you can see that we almost got
# the true slope (2):
# 

print("Our slope is", w[1][0])


######################################################################
# The intercept seems a little off, but that’s okay because our data is in
# a big range (:math:`x \in [0, 50], y \in [7, 107]`). If we normalize the
# data into :math:`[0, 1]` range, expect it to be much closer.
# 


######################################################################
# Below is our sum of squared error for the best fit line. Note that the
# number doesn’t mean anything much, apart from that this is the least
# possible loss we would get from any lines that try to fit the data:
# 

diff = (y - X @ w)
loss = diff.T @ diff
print(loss)


######################################################################
# If you don’t want intermediate variable, you can use ``np.linalg.norm``,
# but to get the sum of squared loss, you have to square that after:
# 

loss = np.linalg.norm(y - X @ w, ord=2) ** 2
print(loss)


######################################################################
# Visualize the loss surface
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# Let’s confirm that our solution is really the one with lowest loss by
# seeing the loss surface.
# 
# Our loss function :math:`L(w)` depends on two dimensions of :math:`w`,
# e.g., ``w[0]`` and ``w[1]``. If we plot :math:`L(w)` over possible
# values of ``w[0]`` and ``w[1]``, the minimum of :math:`L(w)` should be
# near ``w = [5.17,2.066]``, which is our solution.
# 
# To plot that out, first we have to create all possible values of w[0]
# and w[1] in a grid.
# 

from mpl_toolkits.mplot3d import Axes3D

# Ranges of w0 and w1 to see, centering at the true line
spanning_radius = 10
w0range = np.arange(7-spanning_radius, 7+spanning_radius, 0.05)
w1range = np.arange(2-spanning_radius, 2+spanning_radius, 0.05)
w0grid, w1grid = np.meshgrid(w0range, w1range)

range_len = len(w0range)
print("Number of values in each axis:", range_len)


######################################################################
# This means we’ll look into a total of 400*400 = 160,000 values of ``w``.
# We have to calculate loss for each pair of ``w0, w1``:
# 

# Make [w0, w1] in (2, 14400) shape
all_w0w1_values = np.hstack([w0grid.flatten()[:,None], w1grid.flatten()[:,None]]).T

# Compute all losses, reshape back to grid format
all_losses = (np.linalg.norm(y - (X @ all_w0w1_values), axis=0, ord=2) ** 2).reshape((range_len, range_len))


######################################################################
# Then, we can plot the loss surface (with minimum at the red point):
# 

fig = plt.figure(figsize=(10,6))
ax = fig.gca(projection='3d')

ax.plot_surface(w0grid, w1grid, all_losses, alpha=0.5, cmap='RdBu')
ax.contour(w0grid, w1grid, all_losses, offset=0, alpha=1, cmap='RdBu')
ax.scatter(w[0], w[1], loss, lw=3, c='red', s=100, label="Minimum point (5.9,2.2)")

ax.legend(loc='best')
ax.set_xlabel('w[0]')
ax.set_ylabel('w[1]')
ax.set_zlabel('L(w)')
ax.set_xticks(np.arange(7-spanning_radius, 7+spanning_radius, 2))
ax.set_yticks(np.arange(2-spanning_radius, 2+spanning_radius, 2))
ax.set_zticks([loss])
plt.show()


######################################################################
# You can notice the bowl **centers** at the solution.
# 


######################################################################
# Using sklearn
# -------------
# 
# Using ``sklearn`` for linear regression is very simple (if you already
# understand all the concepts above).
# 

from sklearn.linear_model import LinearRegression


######################################################################
# First we create the classifier ``clf``. If ``fit_intercept`` is ``True``
# (default), then it adds the dummy ‘1’ to the ``X``. But we already did
# that manually, so set it to ``False`` here.
# 

clf = LinearRegression(fit_intercept=False)


######################################################################
# Then fit the data:
# 

clf.fit(X,y)


######################################################################
# Check the :math:`w` learned, it’s the same as ours:
# 

print(clf.coef_)


######################################################################
# Previously we use ``X @ w`` to predict data. For ``sklearn`` we can use
# ``clf.predict``:
# 

predicted = clf.predict(X)


######################################################################
# And the result is the same:
# 

plt.figure()
plt.scatter(X[:,1:], y, s=10, label='Data points')
plt.plot(X[:,1:], true_targets, c='blue', label='True line', alpha=0.3)
plt.plot(X[:,1:], predicted, c='red', label='Best fit line')
plt.legend(loc='best')
plt.show()