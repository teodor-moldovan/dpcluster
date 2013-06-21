import math
import numpy as np
import scipy.linalg
import scipy.special
import matplotlib.pyplot as plt
import time
from caching import cached

#TODO: gradient and hessian information not currently used except for grad log likelihood for the NIW distribution. Consider removing extra info.
class ExponentialFamilyDistribution:
    r""" Models a distribution in the exponential family of the form:
    
        :math:`f(x | \nu) = h(x) \exp( \nu \cdot T(x) - A(\nu) )`
    
        Parameters to be defined in subclasses:

        * h is the base measure
        * nu (:math:`\nu`) are the parameters
        * T(x) are the sufficient statistics of the data
        * A is the log partition function
    """
    #TODO: base measure assumed to be scalar. Needs to be fixed for generality.
    def log_base_measure(self,x, ret_ll_gr_hs = (True,False,False) ):
        """
        Log of the base measure. To be implemented by subclasses.

        :arg x: sufficient statistics of the data.
        """
        pass
    def log_partition(self,nu, ret_ll_gr_hs = (True,False,False) ):
        """
        Log of the partition function and derivatives with respect to sufficient statistics. To be implemented by subclasses.

        :arg nu: parameters of the distribution
        :arg ret_ll_gr_hs: what to return: log likelihood, gradient, hessian
        """
        pass

    #TODO: derivatives not implemented. Consider removing
    def ll(self,xs,nus,ret_ll_gr_hs = (True,False,False)  ):
        """
        Log likelihood (and derivatives, optionally) of data under distribution.

        :arg xs: sufficient statistics of data
        :arg nus: parameters of distribution
        """
        return ((np.einsum('ci,di->dc',nus,xs) 
            + self.log_base_measure(xs)[0]
            - self.log_partition(nus)[0][np.newaxis,:]  ),)
        
        
class Gaussian(ExponentialFamilyDistribution):
    r"""Multivariate Gaussian distribution with density:

    :math:`f(x | \mu, \Sigma) = |2 \pi \Sigma|^{-1/2} \exp(-(x-\mu)^T \Sigma^{-1} (x - \mu)/2)`
    
    Natural parameters:     

    :math:`\nu = [\Sigma^{-1} \mu, -\Sigma^{-1}/2]`
        
    Sufficient statistics of data:
    
    :math:`T(x) = [x, x \cdot x^T]` 
     
    :arg d: dimension.

    """
    def __init__(self,d):
        self.dim = d
        self.lbm = math.log(2*np.pi)* (-d/2.0)
        self.inv = np.linalg.inv
        self.slogdet = np.linalg.slogdet

    def sufficient_stats(self,x):
        r""" Sufficient statistics of data.
        :arg x: data
        """
        tmp = (x[:,np.newaxis,:]*x[:,:,np.newaxis]).reshape(x.shape[0],-1)
        return np.hstack((x,tmp))
    def sufficient_stats_dim(self):
        """
        Dimension of sufficient statistics.
        """
        d = self.dim
        return d + d*d
    def log_base_measure(self,x,ret_ll_gr_hs = (True,True,True)):
        r""" Log base measure.
        """
        return (self.lbm, 0.0,0.0)
    def usual2nat(self,mus, Sgs):
        r"""Convert usual parameters to natural parameters.
        """
        nu2 = np.array(map(self.inv,Sgs))
        nu1 = np.einsum('nij,nj->ni',nu2,mus)
        nu = np.hstack((nu1,-.5*nu2.reshape(nu2.shape[0],-1)))
        return nu
        
    def nat2usual(self,nus):
        """Convert natural parameters to usual parameters"""
        d = self.dim
        nu1 = nus[:,:d]
        nu2 = nus[:,d:].reshape((-1,d,d))        
        Sgs = np.array(map(self.inv,-2.0*nu2))
        mus = np.einsum('nij,nj->ni',Sgs,nu1)
        return mus,Sgs
    def log_partition(self,nus):

        # todo: implement gradient and hessian
        d = self.dim 
        nu1 = nus[:,:d]
        nu2 = nus[:,d:].reshape((-1,d,d))        
        inv = np.array(map(self.inv,nu2))
        # TODO: einsum wrong:
        t1 = -.25* np.einsum('ti,tij,tj->t',nu1,inv,nu1)
        t2 = -.5*np.array(map(self.slogdet,-2*nu2))[:,1]
        return (t1+t2,)



class NIW(ExponentialFamilyDistribution):
    r""" Normal Inverse Wishart distribution defined by:
    
        :math:`f(\mu,\Sigma|\mu_0,\Psi,k) = \text{Gaussian}(\mu|\mu_0,\Sigma/k) \cdot \text{Inverse-Wishart}(\Sigma|\Psi,\nu-d-2)`

        where :math:`\mu, \mu_0 \in R^d, \Sigma, \Psi \in R^{d \times d}, k \in R, \nu > 2d+1 \in R`
        
        This is an exponential family conjugate prior for the Gaussian.
        
        :arg d: dimension
    """
    def __init__(self,d):
        self.dim = d
        self.lbm = math.log(2*np.pi)* (-d/2.0)
    def sufficient_stats_dim(self):
        d = self.dim
        return d+d*d +2

    def log_base_measure(self,x,ret_ll_gr_hs = (True,True,True)):
        return (self.lbm, 0.0,0.0)
    def log_partition(self,nu, ret_ll_gr_hs= (True,False,False),
                no_k_grad=False ):
        
        # todo: implement hessian
        rt = ret_ll_gr_hs
        ll = None
        gr = None
        hs = None

        d = self.dim
        l1 = nu[:,:d]
        l2 = nu[:,d:-2].reshape(-1,d,d)
        l3 = nu[:,-2]
        l4 = nu[:,-1]
        
        nu = (l4-d-2).reshape(-1)
        psi = (l2 - 
            l1[:,:,np.newaxis]*l1[:,np.newaxis,:]/l3[:,np.newaxis,np.newaxis])

        if not no_k_grad:
            ld = np.array(map(np.linalg.slogdet,psi))[:,1]

        if rt[0]:
            if not nu.size==1:
                lmg = scipy.special.multigammaln(.5*nu,d)
            else:
                lmg = scipy.special.multigammaln(.5*nu[0],d)

            al = -.5*d*np.log(l3) + .5*nu*(d * np.log(2) - ld ) + lmg
            ll = al

        if rt[1]:
            inv = np.array(map(np.linalg.inv,psi))
            g1 = (nu/l3)[:,np.newaxis]* np.einsum('nij,nj->ni',inv,l1)
            g2 = -.5*nu[:,np.newaxis] *inv.reshape(l2.shape[0],-1)

            g3 = ( -.5 * d/l3
                - .5/l3 * (g1*l1).sum(1)  )[:,np.newaxis]

            if not no_k_grad:
                g4 = ( + .5 *d*np.log(2) - .5*ld + .5*self.multipsi(.5*nu,d)
                     )[:,np.newaxis]
            else:
                g4 = np.zeros((nu.shape[0],1))
                

            gr = np.hstack((g1,g2,g3,g4))

        if rt[2]:   
            # not implemented
            pass

        return (ll,gr,hs)

    def multipsi(self,a,d):
        res = np.zeros(a.shape)
        for i in range(d):
            res += scipy.special.psi(a - .5*i)
        return res    


    def sufficient_stats(self,mus,Sgs):

        Sgi = np.array(map(np.linalg.inv,Sgs))
        nu1 = np.einsum('nij,nj->ni',Sgi,mus)
        nu = np.hstack((nu1,-.5*Sgi.reshape(Sgi.shape[0],-1)))

        t1 = -.5* (Sgi*mus[:,:,np.newaxis]*mus[:,np.newaxis,:]).sum(1).sum(1)
        #t1 = -.5* np.einsum('ti,tij,tj->t',mus,Sgi,mus)
        t2 = -.5*np.array(map(np.linalg.slogdet,Sgs))[:,1]
        return np.hstack((nu, t1[:,np.newaxis],t2[:,np.newaxis]))

    def usual2nat(self,mu0,Psi,k,nu):
        l3 = k.reshape(-1,1)
        l4 = (nu+2+self.dim).reshape(-1,1)
        l1 = mu0*l3
        l2 = Psi + l1[:,:,np.newaxis]*l1[:,np.newaxis,:]/k[:,np.newaxis,np.newaxis]

        return np.hstack((l1,l2.reshape(l2.shape[0],-1),l3,l4 ))

    @cached
    def nat2usual(self,tau,lgh=(True,False,False)):

        d = self.dim
        n,e = tau.shape
        l1 = tau[:,:d]
        l2 = tau[:,d:-2].reshape(-1,d,d)
        l3 = tau[:,-2]
        l4 = tau[:,-1]

        k = l3
        nu = l4 - 2 - d
        mu = l1/l3[:,np.newaxis]
        
        Psi = -l1[:,:,np.newaxis]*l1[:,np.newaxis,:]/l3[:,np.newaxis,np.newaxis]
        Psi += l2
        
        vs = (mu,Psi,k,nu)
        gr = (None,None,None,None)
        
        if lgh[1]:  
        
            l1g = np.zeros((n,d,e))
            l1g[:,:,:d]=np.eye(d)

            l2g = np.zeros((n,d*d,e))
            l2g[:,:,d:d*(d+1)]=np.eye(d*d)
            l2g = l2g.reshape(n,d,d,e)
            
            kg = np.zeros((n,e))
            kg[:,-2]=1.0

            nug = np.zeros((n,e))
            nug[:,-1]=1.0
            
            wx = np.newaxis
            mug = l1g/k[:,wx,wx] - kg[:,wx,:]*l1[:,:,wx]/(k*k)[:,wx,wx]

            t2 = (l1[:,:,wx,wx]*l1g[:,wx,:,:] + l1[:,wx,:,wx]*l1g[:,:,wx,:] )/l3[:,wx,wx,wx]
            t3 = kg[:,wx,wx,:]*(l1[:,:,wx]*l1[:,wx,:]/(l3*l3)[:,wx,wx])[:,:,:,wx]
            Psig = (l2g - t2 + t3)

            gr = (mug,Psig,kg,nug)
        
        return vs,gr,None

class ConjugatePair:
    """
    Conjugate prior-evidence pair of distributions in the exponential family. Conjugacy means that the posterior has the same for as the prior with updated parameters.
        
    :arg evidence_distr: Evidence distribution. Must be an instance of :class:`ExponentialFamilyDistribution`
    :arg prior_distr: Prior distribution. Must be an instance of :class:`ExponentialFamilyDistribution`
    :arg prior_param: Prior parameters.
    """
    def __init__(self,evidence_distr,prior_distr, prior_param):
        self.evidence = evidence_distr
        self.prior = prior_distr
        self.prior_param = prior_param
    def sufficient_stats(self,data):
        pass
    def posterior_ll(self,x,nu, ret_ll_gr_hs=(True,False,False),
            usual_x=False):
        """ Log likelihood (and derivatives) of data under posterior predictive distribution.
        
        :arg x: sufficient statistics of data
        :arg nu: prior parameters
        """
        
        if usual_x:
            x = self.sufficient_stats(x)
        ll,gr,hs = (None,None,None)

        n = x.shape[0]
        k,d = nu.shape

        nu_p = (nu[np.newaxis,:,:] + x[:,np.newaxis,:])
        
        llk, grk, hsk =  self.prior.log_partition(nu_p.reshape((-1,d)),
                        ret_ll_gr_hs)

        if ret_ll_gr_hs[0]:
            t1 = self.evidence.log_base_measure(x)[0]
            t2 = self.prior.log_partition(nu)[0]
            t3 = llk.reshape((n,k))
            
            ll = t1 - t2[np.newaxis,:] + t3 

        if ret_ll_gr_hs[1]:

            gr = grk.reshape((n,k,d))
            gr += self.evidence.log_base_measure(x)[1]

        if ret_ll_gr_hs[2]:
            hs = hsk.reshape((n,k,d,d))
            hs += self.evidence.log_base_measure(x)[2]
        
        return (ll,gr,hs)


    def sufficient_stats_dim(self):
        return self.prior.sufficient_stats_dim()

class GaussianNIW(ConjugatePair):
    """Gaussian, Normal-Inverse-Wishart conjugate pair.
        
    The predictive posterior is a multivariate t-distribution.

    :arg d: dimension
    """
    def __init__(self,d):
        # old version used 2*d+1+2, this seems to work well
        # 2*d + 1 was also a good choice but can't remember why
        ConjugatePair.__init__(self,
            Gaussian(d),
            NIW(d),
            np.concatenate([np.zeros(d*d + d), np.array([0, 2*d+1])])
            )
    def sufficient_stats(self,data):
        x = self.evidence.sufficient_stats(data)
        x1 = np.insert(x,x.shape[1],1,axis=1)
        x1 = np.insert(x1,x1.shape[1],1,axis=1)
        return x1

    @cached
    def posterior_ll_cache(self,nu,rt):
        
        p = self.prior.dim 
        (mu, psi,k,nu0) = self.prior.nat2usual(nu)[0]
        
        nu = nu0-p+1
        nu0 = nu0+ 1
        psi = psi*((k+1)/k/nu)[:,np.newaxis,np.newaxis]
        sgi = np.array(map(np.linalg.inv,psi))

        if rt:   
            ll0 = (  scipy.special.gammaln( .5*(nu0))
                   - scipy.special.gammaln( .5*(nu)) 
                   - .5*np.array(map(np.linalg.slogdet,psi))[:,1]
                   - .5*p*np.log(nu)
                   ) -  .5*p*np.log(np.pi)
        else:
            ll0 = None
        
        return ll0,nu,nu0,mu,sgi
        
    @cached
    def posterior_ll(self,x,nu,ret_ll_gr_hs=(True,False,False),usual_x=False):

        #cases not optimized for here
        if not usual_x or ret_ll_gr_hs[2]:
            return ConjugatePair.posterior_ll(self,x,nu,
                    ret_ll_gr_hs=ret_ll_gr_hs,usual_x=usual_x)
                
        rt = ret_ll_gr_hs
        ll0,nu,nu0,mu,sgi = self.posterior_ll_cache(nu,rt[0])
        dx = x[:,np.newaxis,:] - mu[np.newaxis,:,:]

        gr = np.einsum('kij,nkj->nki', sgi,dx)
        al = 1 + np.einsum('nki,nki->nk', gr,dx)/nu

        if rt[0]:   
            ll = ll0 - .5*(nu0)*np.log(al)
        else:
            ll = None

        if rt[1]:   
            gr = -((nu0)/al/nu)[:,:,np.newaxis]*gr
        else:
            gr = None
        
        hs = None

        
        return (ll,gr,hs)
        
        

    def marginal(self, nu,slc):
        #TODO: write a test for this

        d = self.prior.dim

        ds = len(slc)
        slice_distr = GaussianNIW(ds)

        l1 = nu[:,:d]
        l2 = nu[:,d:-2].reshape(-1,d,d)
        l3 = nu[:,-2:-1]
        l4 = nu[:,-1:]  # should sub from this one

        l1 = l1[:,slc]
        l2 = l2[:,slc,:][:,:,slc]
        l4 = l4 - (d-ds) #TODO: either *2 or *1. not sure which
        
        nus = np.hstack([l1,l2.reshape(l2.shape[0],-1), l3, l4])

        return slice_distr,nus
        
    def plot(self, nu, szs, slc,n = 100,):

        nuE = self.prior.nat2usual(nu[szs>0,:])[0]
        d = self.prior.dim
        mus, Sgs, k, nu = nuE

        # plot the mode of the distribution
        Sgs=Sgs/(k + slc.size  + 1)[:,np.newaxis,np.newaxis]
        
        szs /= szs.sum()
         
        for mu, Sg,sz in zip(mus[:,slc],Sgs[:,slc,:][:,:,slc],szs):

            w,V = np.linalg.eig(Sg)
            V =  np.array(np.matrix(V)*np.matrix(np.diag(np.sqrt(w))))

            sn = np.sin(np.linspace(0,2*np.pi,n))
            cs = np.cos(np.linspace(0,2*np.pi,n))
            
            x = V[:,1]*cs[:,np.newaxis] + V[:,0]*sn[:,np.newaxis]
            x += mu
            plt.plot(x[:,1],x[:,0],linewidth=sz*10)

    @cached
    def conditionals_cache(self,nus,i1,i2):
         
        mu,Psi,n,nu = self.prior.nat2usual(nus)[0]
        my,mx = mu[:,i1],mu[:,i2]

        A,B,D = Psi[:,i1,:][:,:,i1], Psi[:,i1,:][:,:,i2], Psi[:,i2,:][:,:,i2]

        Di = np.array(map(np.linalg.inv,D))
        P = np.einsum('njk,nkl->njl',B,Di)
        mygx = my - np.einsum('nij,nj->ni',P,mx)

        Li = A-np.einsum('nik,nlk->nil',P,B)
        V1 = Li/(nu - len(i1) -1)[:,np.newaxis,np.newaxis]
        
        V2 = Di
        

        return mygx,mx,P,V1,V2,n


    @cached
    def conditionals_cache_bare(self,nus,i1,i2):
         
        mu,Psi,n,nu = self.prior.nat2usual(nus)[0]
        my,mx = mu[:,i1],mu[:,i2]

        A,B,D = Psi[:,i1,:][:,:,i1], Psi[:,i1,:][:,:,i2], Psi[:,i2,:][:,:,i2]

        Di = np.array(map(np.linalg.inv,D))
        P = np.einsum('njk,nkl->njl',B,Di)
        #mygx = my - np.einsum('nij,nj->ni',P,mx)

        Li = A-np.einsum('nik,nlk->nil',P,B)
        

        return my,mx,P,Li,Di,n,nu


    @cached
    def conditional_expectation(self,x,nu,iy,ix, 
                    ret_ll_gr_hs=(True,True,False) ):

        mygx,mx,P,V1,V2,n = self.conditionals_cache(nu,iy,ix)
        
        m = mygx + np.einsum('kij,nj->nki',P,x)
        return (m,P[np.newaxis,:,:,:],None)

    def conditional_variance(self,x,nu,iy,ix,
                    ret_ll_gr_hs=(True,True,False),full_var=True ):

        mygx,mx,P,V1,V2,n = self.conditionals_cache(nu,iy,ix)        

        df = x[:,np.newaxis,:]-mx[np.newaxis,:,:]
        cf = np.einsum('nkj,nkj->nk',np.einsum('nki,kij->nkj',df, V2),df )

        if full_var:
            V =V1[np.newaxis,:,:,:]*(1.0+ 1.0/n + cf)[:,:,np.newaxis,np.newaxis]
        else:
            V = V1[np.newaxis,:,:,:]*(1.0/n + cf)[:,:,np.newaxis,np.newaxis]
        
        cfg = 2*np.einsum('nki,kij->nkj',df,V2)
        
        gr = V1[np.newaxis,:,:,:,np.newaxis] * cfg[:,:,np.newaxis,np.newaxis,:]

        return V,gr,None


    @cached
    def conditional(self,x,nu,iy,ix,lgh = (True,True,False)):

        my,mx,P,Psi,Lxx,n,nu = self.conditionals_cache_bare(nu,iy,ix)
        
        df = x[:,np.newaxis,:]-mx[np.newaxis,:,:]
        nb = 1.0/(np.einsum('nki,kij,nkj->nk',df,Lxx,df) + 1.0/n)

        nub = np.tile(nu[np.newaxis,:],[nb.shape[0],1])
        
        mub = my + np.einsum('kij,nkj->nki',P,df)
        d = len(iy)
        
        
        wx = np.newaxis
        l3 = nb
        l4 = (nub+2+d)
        l1 = mub*nb[:,:,wx]
        l2 = Psi[wx,:,:,:] + mub[:,:,:,wx]*mub[:,:,wx,:]*nb[:,:,wx,wx]

        tau = np.dstack((l1,l2.reshape(l1.shape[0],l1.shape[1],-1),l3,l4)) 
        
        # gradients
        if lgh[1]:
            l3g = - 2*(nb*nb)[:,:,np.newaxis]* np.einsum('nki,kij->nkj',df,Lxx)
            l4g = np.zeros(l3g.shape)
            
            l1g = nb[:,:,wx,wx]*P[wx,:,:,:] + l3g[:,:,wx,:]*mub[:,:,:,wx]
            
            l2g = (mub[:,:,:,wx,wx]*mub[:,:,wx,:,wx]*l3g[:,:,wx,wx,:]
                + P[wx,:,:,wx,:]*mub[:,:,wx,:,wx]*l3[:,:,wx,wx,wx]
                + mub[:,:,:,wx,wx]*P[wx,:,wx,:,:]*l3[:,:,wx,wx,wx]
                    )
            taug =  np.concatenate((l1g,
                    l2g.reshape(l1g.shape[0],l1g.shape[1],-1,l1g.shape[3]), 
                    l3g[:,:,wx,:],l4g[:,:,wx,:]),axis=2)
        else:
            taug = None
            
        distr = GaussianNIW(d)
        return distr,tau,taug,None
        
        
        
