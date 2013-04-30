Description
===========

dpcluster is a package for grouping together (clustering) vectors. It automatically chooses the number of clusters that fits the data best. Specifically, it models the data as a Dirichlet Process mixture in the exponential family. For a tutorial see `"Dirichlet Process" by Y.W. Teh (2010) <http://www.gatsby.ucl.ac.uk/~ywteh/research/npbayes/dp.pdf>`_. Currently the only distribution implemented is the multivariate Gaussian with a Normal-Inverse-Wishart conjugate prior but extensions to other distributions are possible. 

Two inference algorithms are implemented:

* Variational inference as described in `"Variational Inference for Dirichlet Process Mixtures" by Blei et al. (2006) <http://ba.stat.cmu.edu/journal/2006/vol01/issue01/blei.pdf>`_. This is a batch algorithm that requires storing all data in memory.
* An experimental on-line inference algorithm that requires only O(log(n)) memory where n is the total number of observations.

To install locally run::

    python setup.py install --user

Usage
=====

Here is a simple example to demonstrate clustering a number of random points in the plane::

    >>> from dpcluster import *
    >>> n = 10
    >>> data = np.random.normal(size=2*n).reshape(-1,2)
    >>> vdp = VDP(GaussianNIW(2))
    >>> vdp.batch_learn(vdp.distr.sufficient_stats(data))
    >>> plt.scatter(data[:,0],data[:,1])
    >>> vdp.plot_clusters(slc=np.array([0,1]))
    >>> plt.show()

Running this might produce 2-3 clusters depending on the randomly generated data. The adaptive nature of the Dirichlet Process mixture model becomes apparent when we increase the number of data point from ``n = 10`` to ``n = 500``. In this case the clustering algorithm will likely explain the data using only one cluster.

ToDo
====

* Implement more clustering algorithms e.g. based on Gibbs sampling, expectation propagation, stochastic gradient descent.
* Implement more clustering distributions.
* Re-implement algorithms to take advantage of multi-core or GPU computing.


