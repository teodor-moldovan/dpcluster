import numpy as np
import hashlib

enabled=True
cache = {}

def h(a):
    try:
        return hash(a)
    except:
        if isinstance(a,np.ndarray):
            rt = hashlib.sha1(a).hexdigest() 
            return rt


def all_read_only(rt):   
    try:
        for a in rt:
            if isinstance(a,np.ndarray):
                a.setflags(write=False)
            #else:
            #    for b in a:
            #        if isinstance(b,np.ndarray):
            #            b.setflags(write=False)

    except TypeError:
        if isinstance(rt,np.ndarray):
            rt.setflags(write=False)
    return rt
                    

hsh = lambda(x) : hash(tuple((h(a) for a in x)))


def cached(f):
    def new_f(*args):
         
        fh = hsh((f,))
        key = hsh(args)
        
        if not fh in cache:
            cache[fh] = [None,None]
        
        if cache[fh][0]!= key:
            #print f.__name__+' cache miss'
            rt = all_read_only(f(*args) )
            cache[fh] = (key,rt)
        else:
            #print f.__name__+' cache hit'
            pass

        return cache[fh][1]
    if enabled:
        return new_f
    else:
        return f


