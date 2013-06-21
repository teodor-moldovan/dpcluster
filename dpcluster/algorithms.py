import math
import numpy as np
import scipy.linalg
import scipy.special
from caching import cached

class VDP(object):
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
    def ll(self,x, ret_ll_gr_hs = (True,False,False)):
        """
        Compute the log likelihoods (ll) of data with respect to the trained model.

        :arg x: sufficient statistics of the data.
        :arg ret_ll_gr_hs: what to return: likelihood, gradient, hessian. Derivatives taken with respect to data, not sufficient statistics. 
        """

        rt = ret_ll_gr_hs
        llk,grk,hsk = self.distr.posterior_ll(x,self.tau,
             (True,rt[1],rt[2]), True)

        ll = None
        gr = None
        hs = None

        let = self.resp_cache(self.al,self.bt)

        llk = llk+let 
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

    @cached
    def resp_cache(self,al,bt):
        tmp = np.log(al + bt)
        exlv  = np.log(al) - tmp
        exlvc = np.log(bt) - tmp
        let = exlv + np.concatenate([[0],np.cumsum(exlvc)[:-1]])
        return let


    @cached
    def pseudo_resp_cache(self,al,bt):
        tmp = scipy.special.psi(al + bt)
        exlv  = (scipy.special.psi(al) - tmp)
        exlvc = (scipy.special.psi(bt) - tmp)

        elt = (exlv + np.concatenate([[0],np.cumsum(exlvc)[:-1]]))


        return elt


    @cached
    def resp(self,x, ret_ll_gr_hs = (True,False,False)):
        """
        Cluster responsabilities.

        :arg x: sufficient statistics of data. 
        """
        
        cll,cgr,chs = ret_ll_gr_hs
        p = None
        gp = None
        hp = None

        llk,grk,hsk = self.distr.posterior_ll(x,self.tau,(True,cgr,chs),True)

        if cll or cgr: 
            llk = llk + self.resp_cache(self.al,self.bt)   
            llk -= llk.max(1)[:,np.newaxis] 
            np.exp(llk,llk)
            se = llk.sum(1)
            p = llk/se[:,np.newaxis]
        
        if cgr:
            mn = np.einsum('nkj,nk->nj',grk,p)
            gp = (grk - mn[:,np.newaxis,:] )*p[:,:,np.newaxis]

        return (p,gp,hp)


    @cached
    def pseudo_resp(self,x, ret_ll_gr_hs = (True,False,False)):
        
        cll,cgr,chs = ret_ll_gr_hs
        p = None
        gp = None
        hp = None

        grad = self.distr.prior.log_partition(self.tau,(False,True,False))[1]
        llk = np.einsum('ki,ni->nk',grad,self.distr.sufficient_stats(x))

        llk += self.pseudo_resp_cache(self.al,self.bt)   
        llk -= llk.max(1)[:,np.newaxis] 
        np.exp(llk,llk)
        se = llk.sum(1)
        p = llk/se[:,np.newaxis]

        return (p,None,None)


    def conditional_ll(self,x,cond):
        """
        Conditional log likelihood.
        
        :arg x: sufficient statistics of data.
        :arg cond: slice representing variables to condition on
        """

        ll , gr, hs    = self.ll(x,(True,True,True), usual_x=True)
        ll_ , gr_, hs_ = self.marginal(cond).ll(x,(True,True,True),usual_x=True)
        
        ll -= ll_
        gr[:,slc] -= gr_
        #line below will fail
        #hs -= hs_

        return (ll,gr,None)

    def plot_clusters(self,**kwargs):
        """
        Asks each cluster to plot itself. For Gaussian multidimensional clusters pass ``slc=np.array([i,j])`` as an argument to project clusters on the plane defined by the i'th and j'th coordinate.
        """
        sz = self.cluster_sizes()
        self.distr.plot(self.tau, sz, **kwargs)

    @cached
    def marginal(self,slc):
        
        distr, tau = self.distr.marginal(self.tau,slc)
        rv = type(self)(distr)
        rv.tau = tau
        rv.al = self.al
        rv.bt = self.bt
        
        return rv

    @cached
    def conditional_expectation(self,x,iy,ix,ret_ll_gr_hs = (True,False,False)):
        ps, psg, trash = self.marginal(ix).resp(x,ret_ll_gr_hs)

        ex, exg, trash = self.distr.conditional_expectation(x,self.tau,iy,ix,
                        ret_ll_gr_hs)
        
        ef = np.einsum('nki,nk->ni',ex,ps)
        efg = np.einsum('nka,nki->nia',psg,ex)+np.einsum('nk,nkia->nia',ps,exg)
        #efg = np.einsum('nka,nki->nia',psg,ex)+np.einsum('nk,nkia->nia',ps,exg)

        
        return ef,efg,None
        
    def conditional_variance(self,x,iy,ix,ret_ll_gr_hs = (True,False,False)):
        ps, psg, trash = self.marginal(ix).resp(x,ret_ll_gr_hs)

        ex, exg, trash = self.distr.conditional_expectation(x,self.tau,iy,ix,
                        ret_ll_gr_hs)

        vr, vrg, trash = self.distr.conditional_variance(x,self.tau,iy,ix,
                        ret_ll_gr_hs)
        
        ef, efg, trash = self.conditional_expectation(x,iy,ix,ret_ll_gr_hs) 
        
        de = ex - ef[:,np.newaxis,:]
        vt = de[:,:,:,np.newaxis]*de[:,:,np.newaxis,:]
        vs = vt+vr
        vf = np.einsum('nk,nkij->nij',ps,vs)
        
        deg = exg - efg[:,np.newaxis,:,:]

        vsg  =  de[:,:,np.newaxis,:,np.newaxis] * deg[:,:,:,np.newaxis,:]
        vsg +=  de[:,:,:,np.newaxis,np.newaxis] * deg[:,:,np.newaxis,:,:]
        vsg += vrg
        
        vfg  = np.einsum('nk,nkija->nija',ps,vsg) 
        vfg += np.einsum('nka,nkij->nija',psg,vs)

        return vf, vfg, None

    def var_cond_exp(self,x,iy,ix,ret_ll_gr_hs = (True,False,False),
            full_var=False):
        ps, psg, trash = self.marginal(ix).resp(x,ret_ll_gr_hs)

        vr, vrg, trash = self.distr.conditional_variance(x,self.tau,iy,ix,
                        ret_ll_gr_hs,full_var)
        ps2 = ps*ps
        
        vf = np.einsum('nk,nkij->nij',ps2,vr)
        
        vfg = None

        if ret_ll_gr_hs[1]:
            vfg  = np.einsum('nk,nkija->nija',ps2,vrg) 
            vfg += 2*np.einsum('nka,nkij->nija',psg*ps[:,:,np.newaxis],vr)

        return vf, vfg, None



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
        return proc
        
        

class Predictor:
    def __init__(self,model,ix,iy):
        self.model = model
        self.ix = ix
        self.iy = iy

    @cached
    def distr_fit(self,x,lgh):
        ix = self.ix
        iy = self.iy

        ps,psg,trash =self.model.marginal(ix).resp(x,lgh)
        tau = np.einsum('nk,ki->ni',ps,self.model.tau) 
        
        if lgh[1]:
            taug =np.einsum('nka,ki->nia',psg,self.model.tau) 
        else:
            taug = None

        return tau,taug

    @cached
    def precomp(self,x,lgh):
        ix = self.ix
        iy = self.iy
        tau,taug = self.distr_fit(x,lgh)
        (mu,Psi,n,nu),(mug,Psig,ng,nug),trs = self.model.distr.prior.nat2usual(tau,lgh)

        A,B,D = Psi[:,iy,:][:,:,iy], Psi[:,iy,:][:,:,ix], Psi[:,ix,:][:,:,ix]

        Di = np.array(map(np.linalg.inv,D))
        P = np.einsum('njk,nkl->njl',B,Di)

        Li = A-np.einsum('nik,nlk->nil',P,B)
        V1 = Li
        
        V2 = Di
        
        ls = mu,P,V1,V2,n,nu
        gr = None

        
        if lgh[1]:
            mug = np.einsum('nia,naj->nij',mug,taug)
            Psig = np.einsum('nija,nab->nijb',Psig,taug)
            Bg = Psig[:,iy,:,:][:,:,ix,:]
            Dg = Psig[:,ix,:,:][:,:,ix,:]
            
            
            Pg = np.einsum('njka,nkl->njla',Bg,Di)
            Pg -= np.einsum('njk,nkl,nlma,nmq->njqa',B,Di,Dg,Di)
            
            gr = (mug,Pg,None,None,None,None)
        
        return ls,gr,None
    @cached
    def predict(self,x,lgh): 

        ix = self.ix
        iy = self.iy

        vl,gr,trs =  self.precomp(x,lgh)
        mu,p,V1,V2,ni,nu =  vl
        mug,pg,t1,t2,t3,t4 =  gr
            
        df = x-mu[:,ix]
        yp = (mu[:,iy] + np.einsum('nij,nj->ni',p,df))
        
        ypg = None
        
        if lgh[1]:
            dfg  = -mug[:,ix,:]
            dfg += np.eye(dfg.shape[1])[np.newaxis,:,:]
            
            ypg = (mug[:,iy,:] 
                + np.einsum('nij,nja->nia',p,dfg) 
                + np.einsum('nija,nj->nia',pg,df)) 

        return yp,ypg,None

        
    def predict_old(self,z,lgh=(True,True,False),full_var=False):
        
        ix = self.ix
        iy = self.iy

        x = z[:,ix]
        y = z[:,iy]
        dx = len(ix)
        dy = len(iy)
        
        mu,exg,V1,V2,ni,nu =  self.precomp(x,lgh)[0]

        yp, ypg, trs = self.predict(x,lgh)
        
        xi = y - yp

        P = np.repeat(np.eye(dy,dy)[np.newaxis,:,:],exg.shape[0],0)
        P = np.dstack((P,-ypg))

        df = x-mu[:,ix]
        cf = np.einsum('nj,nj->n',np.einsum('ni,nij->nj',df, V2),df )

        if full_var:
            V = V1*((1.0+ 1.0/n + cf)/(nu+1.0))[:,np.newaxis,np.newaxis]        
        else:
            V = V1*((1.0/n + cf)/(nu+1.0))[:,np.newaxis,np.newaxis]        

        vi = np.array(map(np.linalg.inv,V))
        
        pr = np.einsum('nij,nj->ni',vi,xi)
        ll = np.einsum('nj,nj->n',pr,xi)

        return ll,xi,P,2*vi

        
class PredictorKL:
    def __init__(self,model,ix,iy):
        self.model = model
        self.ix = ix
        self.iy = iy

    @cached
    def predict(self,x,lgh): 

        ix = self.ix
        iy = self.iy
        
        dstr,tau,taug,tsh = self.model.distr.conditional(x,
                self.model.tau,iy,ix,lgh)

        ps,psg,trash = self.model.marginal(ix).resp(x,lgh)
        
        taun = np.einsum('nk,nki->ni',ps,tau)

        (mu,Psi,n,nu),(mug,Psig,ng,nug),trs = dstr.prior.nat2usual(taun,lgh)

        yp = mu
        if lgh[1]:
            taung = (np.einsum('nkj,nki->nij',psg,tau) 
                + np.einsum('nk,nkij->nij',ps,taug))

            ypg = np.einsum('njk,nij->nik',taung,mug)
        
        v = Psi/(n * (nu - len(iy) + 1.0) )[:,np.newaxis,np.newaxis] 

        return yp,ypg,v

        
    def predict_old(self,z,lgh=(True,True,False),full_var=False):
        
        ix = self.ix
        iy = self.iy

        x = z[:,ix]
        y = z[:,iy]
        dx = len(ix)
        dy = len(iy)
        
        mu,exg,V1,V2,ni,nu =  self.precomp(x,lgh)[0]

        yp, ypg, trs = self.predict(x,lgh)
        
        xi = y - yp

        P = np.repeat(np.eye(dy,dy)[np.newaxis,:,:],exg.shape[0],0)
        P = np.dstack((P,-ypg))

        df = x-mu[:,ix]
        cf = np.einsum('nj,nj->n',np.einsum('ni,nij->nj',df, V2),df )

        if full_var:
            V = V1*((1.0+ 1.0/n + cf)/(nu+1.0))[:,np.newaxis,np.newaxis]        
        else:
            V = V1*((1.0/n + cf)/(nu+1.0))[:,np.newaxis,np.newaxis]        

        vi = np.array(map(np.linalg.inv,V))
        
        pr = np.einsum('nij,nj->ni',vi,xi)
        ll = np.einsum('nj,nj->n',pr,xi)

        return ll,xi,P,2*vi

        
