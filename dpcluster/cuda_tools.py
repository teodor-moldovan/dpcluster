import pycuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda import gpuarray
import scikits.cuda.cublas as cublas
from pytools import memoize
import numpy as np

from jinja2 import Template
import atexit

cublas_handle = cublas.cublasCreate()
atexit.register(lambda:cublas.cublasDestroy(cublas_handle))
 
def bptrs(a):
    """
    Pointer array when input represents a batch of matrices.
    """
    return gpuarray.arange(a.ptr,a.ptr+a.shape[0]*a.strides[0],a.strides[0],
                dtype=cublas.ctypes.c_void_p)


@memoize
def k_perm_conv_batched(m):
    permute_template = Template("""
    #define M {{ m }}
    __global__ void permute_inds(int x[][M]) {
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

        int a[M];

        for (int i=0; i<M; i++) a[i]=i;
            
        for (int i=0; i<M; i++){
            int i_=x[idx][i]-1; 
            int t = a[i]; 
            a[i] = a[i_]; 
            a[i_] = t; };

        for (int i=0; i<M; i++) x[idx][i]=a[i];
    }
    """)

    perm_mod = SourceModule(permute_template.render(m=m))
    return  perm_mod.get_function("permute_inds").prepare('P')
    
def perm_conv_batched(p):
    
    l,m = p.shape
    return k_perm_conv_batched(m).prepared_call((l,1,1),(1,1,1),p.gpudata)



@memoize
def k_perm_rows_batched(m,n):
    permute_template = Template("""
    #define M {{ m }}
    #define N {{ n }}
    __global__ void batch_permute_rows(int s[][M], 
                float a[][M][N], float b[][M][N]) {
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;
        unsigned int idz = blockIdx.z * blockDim.z + threadIdx.z;
        
        b[idx][idy][idz] = a[idx][idy][s[idx][idz]];
    }

    """)

    perm_mod = SourceModule(permute_template.render(m=m,n=n))
    return perm_mod.get_function("batch_permute_rows").prepare('PPP')
        

def perm_rows_batched(p,s,d):
    l,m,n = s.shape
    return k_perm_rows_batched(m,n).prepared_call((l,1,1),(1,m,n),
            p.gpudata, s.gpudata, d.gpudata)



@memoize
def k_perm_mat_batched(m):
    permute_template = Template("""
    #define M {{ m }}
    __global__ void batch_permute_id(int s[][M], 
                float a[][M][M]) {
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;
        unsigned int idz = blockIdx.z * blockDim.z + threadIdx.z;
        
        a[idx][idy][idz] = s[idx][idz]==idy ;
    }

    """)

    perm_mod = SourceModule(permute_template.render(m=m))
    return perm_mod.get_function("batch_permute_id").prepare('PP')
        

def perm_mat_batched(p,d):
    l,m,m = d.shape
    return k_perm_mat_batched(m).prepared_call((l,1,1),(1,m,m),
            p.gpudata, d.gpudata)


@memoize
def temp_perm(l,m):
    return gpuarray.empty((l,m),np.int32) 

@memoize
def temp_ret_code():
    return gpuarray.empty(1,np.int32) 

def binv(s,d):
    l,m,m = s.shape
    p = temp_perm(l,m)
    i = temp_ret_code()
        
    cublas.cublasSgetrfBatched(cublas_handle,
                m, s.bptrs.gpudata, m, p.gpudata, i.gpudata, l)
    
    perm_conv_batched(p)
    perm_mat_batched(p,d)
    
    cublas.cublasStrsmBatched(cublas_handle,'l','l','n','u',m,m,
                np.float32(1.0), 
                s.bptrs.gpudata, m, 
                d.bptrs.gpudata, m, l
                )

    cublas.cublasStrsmBatched(cublas_handle,'l','u','n','n',m,m,
                np.float32(1.0), 
                s.bptrs.gpudata, m, 
                d.bptrs.gpudata, m, l
                )
 
