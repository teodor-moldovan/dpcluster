from unittest import TestCase
from dpcluster import *


class Tests(unittest.TestCase):
    #TODO: hessians not tested
    def test_gaussian(self):
        k = 4
        d = Gaussian(k)

        mus = np.random.sample((10,k))
        Sgs = np.random.sample((10,k,k))
        Sgs = np.einsum('tki,tkj->tij',Sgs,Sgs)

        nu = d.usual2nat(mus, Sgs)
        mus_,Sgs_ = d.nat2usual(nu)
        
        np.testing.assert_array_almost_equal(mus,mus_)
        np.testing.assert_array_almost_equal(Sgs,Sgs_)
        
        data = np.random.sample((100,k))
        xs = d.sufficient_stats(data)

        lls = d.ll(xs,nu)[0]
        
        Sg = Sgs[0,:,:]
        mu = mus[0,:]
        x = data[0,:]
        ll = (-k*.5*math.log(2*np.pi) -.5* np.linalg.slogdet(Sg)[1] 
                -.5* ((mu-x)*scipy.linalg.solve(Sg,(mu-x))).sum()  )
        self.assertAlmostEqual(ll, lls[0,0])
        
    def test_niw(self):
        p = 3
        d = NIW(p)

        mus = np.random.randn(100,p)
        Sgs = np.random.randn(100,p,p)
        Sgs = np.einsum('tki,tkj->tij',Sgs,Sgs)

        x = d.sufficient_stats(mus, Sgs)

        mu0 = np.random.randn(10,p)
        Psi = np.random.randn(10,p,p)
        Psi = np.einsum('tki,tkj->tij',Psi,Psi)
        k = np.random.rand(10)*10
        nu = p - 1 + k
        
        nus = d.usual2nat(mu0,Psi,k,nu)
        mu0_,Psi_,k_,nu_ = d.nat2usual(nus)
        np.testing.assert_array_almost_equal(mu0_,mu0)
        np.testing.assert_array_almost_equal(Psi_,Psi)
        np.testing.assert_array_almost_equal(k_,k)
        np.testing.assert_array_almost_equal(nu_,nu)


        jac = d.log_partition(nus,
                [False,True,False])[1]
        eSgi = -2*jac[:,p:p*(p+1)].reshape(-1,p,p)
        Psii = np.array(map(np.linalg.inv, Psi))
        eSgi_ = Psii*(nu)[:,np.newaxis,np.newaxis]
        
        np.testing.assert_array_almost_equal(eSgi,eSgi_)
            
        lls = d.ll(x,nus)[0]
        
        mu0 = mu0[0,:]
        mu = mus[0,:]
        Sg = Sgs[0,:,:]
        Psi = Psi[0,:,:]
        k = k[0]
        nu = nu[0]
        
        ll1 = (-p*.5*math.log(2*np.pi) -.5* np.linalg.slogdet(Sg/k)[1] 
                -.5* ((mu0-mu)*scipy.linalg.solve(Sg/k,(mu0-mu))).sum()  )
        ll2 = (.5*nu*np.linalg.slogdet(Psi)[1] - .5*nu*p*np.log(2) 
                - scipy.special.multigammaln(.5*nu,p) 
                - .5*(nu+p+1)*np.linalg.slogdet(Sg)[1] 
                - .5 * np.sum(Psi * np.linalg.inv(Sg))  )

        self.assertAlmostEqual(ll1+ll2, lls[0,0] )
        
        al = 1e-10
        nu1 = al*nus[1,:] -al *nus[0,:] + .5 *nus[0,:] + .5*nus[1,:]
        nu2 = al*nus[0,:] -al *nus[1,:] + .5 *nus[0,:] + .5*nus[1,:]
        
        diff = (d.log_partition(nu2[np.newaxis,:])[0]
                - d.log_partition(nu1[np.newaxis,:])[0])[0]
            
        jac = d.log_partition(.5 *nus[0:1,:] + .5*nus[1:2,:],
                [False,True,False])[1]
        self.assertAlmostEqual(diff, (jac.reshape(-1)*(nu2-nu1)).sum())
        

    def test_gniw(self):
        p = 2
        d = GaussianNIW(p)

        np.random.seed(1)
        x  = np.random.randn(100,p)

        mu0 = np.random.randn(10,p)
        Psi = np.random.randn(10,p,p)
        Psi = np.einsum('tki,tkj->tij',Psi,Psi)
        k = np.random.rand(10)*10
        nu = p - 1 + k
        
        nus = d.prior.usual2nat(mu0,Psi,k,nu)

        ll, gr, hs= ConjugatePair.posterior_ll(d,x,nus,
                    [True,True,False],usual_x=True)
        ll_,gr_,hs_= d.posterior_ll(x,nus,
                    [True,True,True],usual_x=True)
        np.testing.assert_array_almost_equal(ll,ll_)
        
        al = 1e-10
        x1 = al*x[1,:] -al *x[0,:] + .5 *x[0,:] + .5*x[1,:]
        x2 = al*x[0,:] -al *x[1,:] + .5 *x[0,:] + .5*x[1,:]
        
        diff = (d.posterior_ll(x2[np.newaxis,:],nus,usual_x=True)[0] - 
                d.posterior_ll(x1[np.newaxis,:],nus,usual_x=True)[0])
        jac = d.posterior_ll(.5 *x[0:1,:] + .5*x[1:2,:],nus,
                [False,True,False], usual_x=True)[1]
        diff_ = (jac *(x2-x1)[np.newaxis,np.newaxis,:]).sum(2)

        np.testing.assert_array_almost_equal(diff,diff_)


