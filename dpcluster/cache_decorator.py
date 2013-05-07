from hashlib import sha1

def cached(f):
    def h(a):
        try:
            return a.__hash__()
        except:
            return sha1(a).hexdigest()  
    hsh = lambda(x) : tuple((h(a) for a in x))
    def new_f(obj,*args):
         
        if not hasattr(obj,'__decorator_cache__'):
            obj.__decorator_cache__ = {}
        cache = obj.__decorator_cache__.setdefault(f.__name__,(None,None))
        
        ah = hsh(args)
        
        if cache[0] != ah:
            rt = f(obj,*args) 
            obj.__decorator_cache__[f.__name__] = (
                    ah,
                    rt
            )
                 
        return obj.__decorator_cache__[f.__name__][1]
    return new_f


