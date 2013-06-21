import pycuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda import gpuarray
import scikits.cuda.cublas as cublas

from jinja2 import Template
import numpy as np
import time

cublas_handle = cublas.cublasCreate()
 
def bptrs(a):
    """
    Pointer array when input represents a batch of matrices.
    """
    return gpuarray.arange(a.ptr,a.ptr+a.shape[0]*a.strides[0],a.strides[0],
                dtype=cublas.ctypes.c_void_p)


def prepare(m):
    permute_template = Template("""
    __global__ void permute_inds(int x[][{{ m }}]) {
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

        int a[{{ m }}];

        for (int i=0; i<{{ m }}; i++) a[i]=i;
            
        for (int i=0; i<{{ m }}; i++){
            int i_=x[idx][i]-1; 
            int t = a[i]; 
            a[i] = a[i_]; 
            a[i_] = t; };

        for (int i=0; i<{{ m }}; i++) x[idx][i]=a[i];
    }
    __global__ void batch_permute_rows(int s[][{{ m }}], 
                        float a[][{{ m }}][{{ m }}],
                        float b[][{{ m }}][{{ m }}]) {
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;
        unsigned int idz = blockIdx.z * blockDim.z + threadIdx.z;
        
        b[idx][idy][idz] = a[idx][idy][s[idx][idz]];
    }

    """)

    perm_mod = SourceModule(permute_template.render(m=m))

    pi = perm_mod.get_function("permute_inds").prepare('P')
    pr = perm_mod.get_function("batch_permute_rows").prepare('PPP')
        
    return pi.prepared_call, pr.prepared_call


from scipy.linalg import lu_factor
l,m = 100*(32+1),32
perm, pmatrix = prepare(m)

A = np.random.rand(l,m, m).astype(np.float32)
A = np.array([np.matrix(a)*np.matrix(a).T for a in A])

a_gpu = gpuarray.to_gpu(A)
a_arr = bptrs(a_gpu)
s_gpu = gpuarray.empty((l,m), np.int32)
i_gpu = gpuarray.zeros(1, np.int32)

po_gpu = gpuarray.to_gpu(np.repeat(np.eye(m)[np.newaxis,:,:],l,axis=0).astype(np.float32))
p_gpu = gpuarray.empty_like(po_gpu)

for k in range(10):
    t = time.time()
    cublas.cublasSgetrfBatched(cublas_handle,
                m, a_arr.gpudata, m, s_gpu.gpudata, i_gpu.gpudata, l)
    print time.time()-t,
    perm((l,1,1),(1,1,1),s_gpu.gpudata)
    pmatrix((l,1,1),(1,m,m),s_gpu.gpudata,po_gpu.gpudata,p_gpu.gpudata)
    print time.time()-t

    #print s_gpu.get()[0]
    #print p_gpu.get()[0]


cublas.cublasDestroy(cublas_handle)

