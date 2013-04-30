import math
import numpy as np
import scipy.linalg
import scipy.special

class VDP:
    """ Variational Dirichlet Process clustering algorithm following `"Variational Inference for Dirichlet Process Mixtures" by Blei et al. (2006) <http://ba.stat.cmu.edu/journal/2006/vol01/issue01/blei.pdf>`_.
 
    :param distr: likelihood-prior distribution pair governing clusters. For now the only option is using a instance of :class:`dpcluster.distributions.GaussianNIW`.
    :param w: non-negative prior weight. The prior has as much influence as w data points.
    :param k: maximum number of clusters.
    :param tol: convergence tolerance.
    """
    def __init__(self,distr, w = .1, k=50,
                tol = 1e-5,
                max_iters = 10000):
       
        self.max_iters = max_iters
        self.tol = tol
        self.distr = distr
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
            grad = self.distr.prior.log_partition(tau,[False,True,False])[1]
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

        self.al = al
        self.bt = bt
        self.tau = tau
        self.lbd = lbd
        self.glp = grad
        self.elt = elt

        return

    def cluster_sizes(self):
        """:return: Data weight assigned to each cluster.
        """
        return (self.al -1)
        
        
    def cluster_parameters(self):
        """:return: Cluster parameters.
        """
        return self.tau
    def ll(self,x, ret_ll_gr_hs = [True,False,False], **kwargs):
        """
        Compute the log likelihoods (ll) of data with respect to the trained model.

        :arg x: sufficient statistics of the data.
        :arg ret_ll_gr_hs: what to return: likelihood, gradient, hessian. Derivatives taken with respect to data, not sufficient statistics. 
        """

        rt = ret_ll_gr_hs
        llk,grk,hsk = self.distr.posterior_ll(x,self.tau,
                [True,rt[1],rt[2]], **kwargs)

        ll = None
        gr = None
        hs = None

        al = self.al
        bt = self.bt

        tmp = np.log(al + bt)
        exlv  = np.log(al) - tmp
        exlvc = np.log(bt) - tmp
        let = exlv + np.concatenate([[0],np.cumsum(exlvc)[:-1]])

        llk +=let 
        np.exp(llk,llk)

        se = llk.sum(1)
        
        if rt[0]:
            ll = np.log(se)

        if rt[1] or rt[2]:
            p = llk/se[:,np.newaxis]
            gr = np.einsum('nk,nki->ni',p,grk)
        
        if rt[2]:
            hs1  = - gr[:,:,np.newaxis] * gr[:,np.newaxis,:]
            hs2 = np.einsum('nk,nkij->nij',p, hsk)
            # TODO: einsum wrong
            hs3 = np.einsum('nk,nki,nkj->nij',p, grk, grk)

            hs = hs1 + hs2 + hs3
        
        return (ll,gr,hs)


    def resp(self,x, **kwargs):
        """
        Cluster responsabilities.

        :arg x: sufficient statistics of data. 
        """
        llk,grk,hsk = self.distr.posterior_ll(x,self.tau,
                [True,False,False], **kwargs)

        al = self.al
        bt = self.bt

        tmp = np.log(al + bt)
        exlv  = np.log(al) - tmp
        exlvc = np.log(bt) - tmp
        let = exlv + np.concatenate([[0],np.cumsum(exlvc)[:-1]])

        llk +=let
        llk -= llk.max(1)[:,np.newaxis] 
        np.exp(llk,llk)
        se = llk.sum(1)
        p = llk/se[:,np.newaxis]

        return p


    def conditional_ll(self,x,cond):
        """
        Conditional log likelihood.
        
        :arg x: sufficient statistics of data.
        :arg cond: slice representing variables to condition on
        """

        ll , gr, hs = self.ll(x,[True,True,True], usual_x=True)
        ll_ , gr_, hs_ = self.ll(x,[True,True,True], 
                slc=cond, usual_x=True)
        
        ll -= ll_
        gr -= gr_
        hs -= hs_

        return (ll,gr,hs)

    def plot_clusters(self,**kwargs):
        """
        Asks each cluster to plot itself. For Gaussian multidimensional clusters pass ``slc=np.array([i,j])`` as an argument to project clusters on the plane defined by the i'th and j'th coordinate.
        """
        sz = self.cluster_sizes()
        self.distr.plot(self.tau, sz, **kwargs)

class OnlineVDP:
    """Experimental online clustering algorithm.

    :param distr: likelihood-prior distribution pair governing clusters. For now the only option is using a instance of :class:`dpcluster.distributions.GaussianNIW`.
    :param w: non-negative prior weight. The prior has as much influence as w data points.
    :param k: maximum number of clusters.
    :param tol: convergence tolerance.
    :param max_items: maximum queue length.
    
    """
    def __init__(self, distr, w=.1, k = 25, tol=1e-3, max_items = 100):
        self.distr = distr
        self.w = w
        self.wm = w
        self.k = k
        self.tol = tol
        self.max_n = max_items

        self.xs = []
        self.vdps = []
        self.dim = self.distr.sufficient_stats_dim()
        
    def put(self,r,s=0):
        """
        Append data.

        :arg r: sufficient statistics of data to be appended. 

        Basic usage example::

            >>> distr = GaussianNIW(data.shape[2])
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
            
            proc = VDP(self.distr, w, self.k*(s+1), self.tol)
            self.vdps.append(proc)

        if pcs==0:
            return        

        for x in np.split(ar[:pcs*self.max_n,:],pcs):
            proc.batch_learn(x,verbose=False)
            xc = proc.tau - proc.lbd[np.newaxis,:]
            xc = xc[xc[:,-1]>1e-5]
            self.put(xc,s+1)

    def get_model(self):
        """
        Get current model.
        
        :return: instance of :class:`dpcluster.algorithms.VDP`
        """

        np.random.seed(1)
        proc = VDP(self.distr, self.wm, self.k*(len(self.xs)), self.tol)
        proc.batch_learn(np.vstack(self.xs[::-1]))
        #print proc.al.sum()
        return proc
        
        

