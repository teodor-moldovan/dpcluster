import math
import numpy as np
import scipy.linalg
import scipy.special
from caching import cached

class VDP:
    """ Variational Dirichlet Process clustering algorithm following `"Variational Inference for Dirichlet Process Mixtures" by Blei et al. (2006) <http://ba.stat.cmu.edu/journal/2006/vol01/issue01/blei.pdf>`_.
 
    :param distr: likelihood-prior distribution pair governing clusters. For now the only option is using a instance of :class:`dpcluster.distributions.GaussianNIW`.
    :param w: non-negative prior weight. The prior has as much influence as w data points.
    :param k: maximum number of clusters.
    :param tol: convergence tolerance.
    """
    def __init__(self,mixture, w = .1, k=50,
                tol = 1e-5,
                max_iters = 10000):
       
        self.max_iters = max_iters
        self.tol = tol
        self.distr = mixture.distr
        self.mixture = mixture
        self.w = w
        self.k = k
        d = self.distr.sufficient_stats_dim()

        self.prior = self.distr.prior_param
        self.s = np.array([0.0,0])
        
    def batch_learn(self,x,verbose=False, sort = True):
        """ Learn cluster from data. This is a batch algorithm that required all data be loaded in memory. 
       
        :arg x: sufficient statistics of the data to be clustered. Can be obtained from raw data by calling :func:`dpcluster.distributions.ConjugatePair.sufficient_stats()`
        :keyword verbose: print progress report 
        :keyword sort: algorithm optimization. Sort clusters at every step.

        Basic usage example::

            >>> distr = GaussianNIW(data.shape[2])
            >>> x = distr.sufficient_stats(data)
            >>> vdp = VDP(distr)
            >>> vdp.batch_learn(x)
            >>> print vdp.cluster_parameters()
        """
        n = x.shape[0] 
        k = self.k
        
        wx = x[:,-1]
        wt = wx.sum()

        lbd = self.prior + x.sum(0) / wx.sum() * self.w
        ex_alpha = 1.0
        
        phi = np.random.random(size=n*k).reshape((n,k))

        for t in range(self.max_iters):

            phi /= phi.sum(1)[:,np.newaxis]
            # m step
            tau = (lbd[np.newaxis,:] + np.einsum('ni,nj->ij', phi, x))
            psz = np.einsum('ni,n->i',phi,wx)

            # stick breaking process
            if sort:
                ind = np.argsort(-psz) 
                tau = tau[ind,:]
                psz = psz[ind]
            
            if t > 0:
                old = al

            al = 1.0 + psz

            if t > 0:
                diff = np.sum(np.abs(al - old))

            bt = ex_alpha + np.concatenate([
                    (np.cumsum(psz[:0:-1])[::-1])
                    ,[0]
                ])

            tmp = scipy.special.psi(al + bt)
            exlv  = (scipy.special.psi(al) - tmp)
            exlvc = (scipy.special.psi(bt) - tmp)

            elt = (exlv + np.concatenate([[0],np.cumsum(exlvc)[:-1]]))

            w = self.s + np.array([-1 + k, -np.sum(exlvc[:-1])])
            ex_alpha = w[0]/w[1]
            
            # end stick breaking process
            # end m step


            # e_step
            grad = self.distr.prior.log_partition(tau,(False,True,False))[1]
            np.einsum('ki,ni->nk',grad,x,out=phi)

            phi /= wx[:,np.newaxis]
            phi += elt
            phi -= phi.max(1)[:,np.newaxis]
            np.exp(phi,phi)
            
            
            if t>0:
                if verbose:
                    print str(diff)
                if diff < wt*self.tol:
                    break

        self.mixture.al = al
        self.mixture.bt = bt
        self.mixture.tau = tau
        self.mixture.lbd = lbd

        return self.mixture

class OnlineVDP:
    """Experimental online clustering algorithm.

    :param distr: likelihood-prior distribution pair governing clusters. For now the only option is using a instance of :class:`dpcluster.distributions.GaussianNIW`.
    :param w: non-negative prior weight. The prior has as much influence as w data points.
    :param k: maximum number of clusters.
    :param tol: convergence tolerance.
    :param max_items: maximum queue length.
    
    """
    def __init__(self, mixture, w=.1, k = 25, tol=1e-3, max_items = 100):
        self.mixture = mixture
        self.w = w
        self.wm = w
        self.k = k
        self.tol = tol
        self.max_n = max_items

        self.xs = []
        self.vdps = []
        self.distr = self.mixture.distr
        self.dim = self.mixture.distr.sufficient_stats_dim()
        
    def put(self,r,s=0):
        """
        Append data.

        :arg r: sufficient statistics of data to be appended. 

        Basic usage example::

            >>> distr = GaussianMixture(data.shape[2])
            >>> x = distr.sufficient_stats(data)
            >>> vdp = OnlineVDP(distr)
            >>> vdp.put(x)
            >>> print vdp.get_model().cluster_parameters()
        """

        if s<len(self.xs):
            ar = np.vstack((self.xs[s],r))
        else:
            ar = r
            self.xs.append(None)
        
        mn = self.max_n*(s+1)

        pcs = ar.shape[0] / mn
        self.xs[s] = ar[pcs*mn:,:]

        if s<len(self.vdps):
            proc = self.vdps[s]
        else:
            if s>0:
                w = self.wm
            else:
                w = self.w
            
            proc = VDP(self.mixture, w, self.k*(s+1), self.tol)
            self.vdps.append(proc)

        if pcs==0:
            return        

        for x in np.split(ar[:pcs*self.max_n,:],pcs):
            model = proc.batch_learn(x,verbose=False)
            xc = model.tau - model.lbd[np.newaxis,:]
            xc = xc[xc[:,-1]>1e-5]
            self.put(xc,s+1)

    def get_model(self):
        """
        Get current model.
        
        :return: instance of :class:`dpcluster.algorithms.VDP`
        """

        np.random.seed(1)
        proc = VDP(self.mixture, self.wm, self.k*(len(self.xs)), self.tol)
        return proc.batch_learn(np.vstack(self.xs[::-1]))
        
        

