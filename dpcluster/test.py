import unittest
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



@unittest.skip("")
class SlowTests(unittest.TestCase): 
    def test_batch_vdp(self):

        np.random.seed(1)
        def gen_data(A, mu, n=10):
            xs = np.random.multivariate_normal(mu,np.eye(mu.size),size=n)
            ys = (np.einsum('ij,j->i',A,mu)
                + np.random.multivariate_normal(
                        np.zeros(A.shape[0]),np.eye(A.shape[0]),size=n))
            
            return np.hstack((ys,xs))


        As = np.array([[[1,2,5],[2,2,2]],
                       [[-4,3,-1],[2,2,2]],
                       [[-4,3,1],[-2,-2,-2]],
                        ])
        mus = np.array([[10,0,0],
                        [0,10,0],
                        [0,0,10],
                        ])

        n = 120
        data = np.vstack([ gen_data(A,mu,n=n) for A,mu in zip(As,mus)])
        d = data.shape[1]
        # can forget mus, As
            
        prob = VDP(GaussianNIW(d), k=50,w=.4)
        x = prob.distr.sufficient_stats(data)
        prob.batch_learn(x, verbose = False)
        
        print prob.cluster_sizes()        
        
        np.testing.assert_almost_equal((prob.al-1)[:3], n*np.ones(3))
        
        # Log likelihood of training data under model
        print prob.ll(x)[0].sum()
        


    def test_ll(self):

        np.random.seed(1)
        def gen_data(A, mu, n=10):
            xs = np.random.multivariate_normal(mu,np.eye(mu.size),size=n)
            ys = (np.einsum('ij,j->i',A,mu)
                + np.random.multivariate_normal(
                        np.zeros(A.shape[0]),np.eye(A.shape[0]),size=n))
            
            return np.hstack((ys,xs))


        As = np.array([[[1,2,5],[2,2,2]],
                       [[-4,3,-1],[2,2,2]],
                       [[-4,3,1],[-2,-2,-2]],
                        ])
        mus = np.array([[10,0,0],
                        [0,10,0],
                        [0,0,10],
                        ])

        n = 120
        x = np.vstack([ gen_data(A,mu,n=n) for A,mu in zip(As,mus)])
        d = x.shape[1]
        
        # done generating test data
            
        # k is the max number of clusters
        # w is the prior parameter. 
        prob = VDP(GaussianNIW(d), k=30,w=0.1)

        xt = prob.distr.sufficient_stats(x)
        prob.batch_learn(xt, verbose = False)
        
        ll , gr, hs = prob.ll(x,[True,True,True], usual_x=True)

        al = 1e-10
        x1 = al*x[1,:] -al *x[0,:] + .5 *x[0,:] + .5*x[1,:]
        x2 = al*x[0,:] -al *x[1,:] + .5 *x[0,:] + .5*x[1,:]
        
        diff = (prob.ll(x2[np.newaxis,:],usual_x=True)[0] - 
                prob.ll(x1[np.newaxis,:],usual_x=True)[0])

        jac = prob.ll(.5 *x[0:1,:] + .5*x[1:2,:],
                [False,True,False], usual_x=True)[1]

        diff_ = (jac *(x2-x1)[np.newaxis,np.newaxis,:]).sum(2)
        np.testing.assert_array_almost_equal(diff,diff_[0])



    def test_resp(self):

        np.random.seed(1)
        def gen_data(A, mu, n=10):
            xs = np.random.multivariate_normal(mu,np.eye(mu.size),size=n)
            ys = (np.finsum('ij,j->i',A,mu)
                + np.random.multivariate_normal(
                        np.zeros(A.shape[0]),np.eye(A.shape[0]),size=n))
            
            return np.hstack((ys,xs))


        As = np.array([[[1,2,5],[2,2,2]],
                       [[-4,3,-1],[2,2,2]],
                       [[-4,3,1],[-2,-2,-2]],
                        ])
        mus = np.array([[10,0,0],
                        [0,10,0],
                        [0,0,10],
                        ])

        n = 120
        x = np.vstack([ gen_data(A,mu,n=n) for A,mu in zip(As,mus)])
        d = x.shape[1]
            
        prob = VDP(GaussianNIW(d), k=30,w=0.1)

        xt = prob.distr.sufficient_stats(x)
        prob.batch_learn(xt, verbose = False)
        
        ps = prob.resp(x, usual_x=True,slc=[2,3,4])

        grad = prob.distr.prior.log_partition(prob.tau,[False,True,False])[1]
        print grad

        ll1 = np.einsum('ki,ni->nk',prob.glp,xt)
        mus,sgs = prob.distr.evidence.nat2usual(prob.glp[:,:-2])
         
        dx = x[:,np.newaxis,:] - mus[np.newaxis,:,:]
        sgi = np.array(map(np.linalg.inv,sgs)) # this is the hessian
        
        # TODO: einsum wrong:
        ll2 = -np.einsum('kij,nki,nkj->nk',.5*sgi,dx,dx)
        
        
        gr = prob.glp[:,:d]
        hs = prob.glp[:,d:d*(d+1)].reshape(-1,d,d)
        # TODO: einsum wrong:
        ll3 = np.einsum('ki,ni->nk', gr,x) +np.einsum('kij,ni,nj->nk',hs,x,x)
        


    def test_online_vdp(self):
        
        hvdp = OnlineVDP(GaussianNIW(3), w=1e-2, k = 20, tol=1e-3, max_items = 100 )
        
        for t in range(1000):
            x = np.mod(np.linspace(0,2*np.pi*3,134),2*np.pi)
            t1 = time.time()
            data = np.vstack((x,np.sin(x),np.cos(x))).T
            hvdp.put(hvdp.distr.sufficient_stats(data))
            hvdp.get_model()
            print time.time()-t1

if __name__ == '__main__':
    single_test = 'test_batch_vdp'
    if hasattr(SlowTests, single_test):
        dev_suite = unittest.TestSuite()
        dev_suite.addTest(Tests(single_test))
        unittest.TextTestRunner().run(dev_suite)
    else:
        unittest.main()


