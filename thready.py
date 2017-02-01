import threading

def task(f,params):
    t = threading.Thread(target=f,args=params)
    return t

def fib(x):
    return 1 if x<=2 else fib(x-1)+fib(x-2)

import threadpool as tp
pool = tp.ThreadPool(4)

# async map function with multithreading support.
# returned mapresult is a map with integer indices.
def amap(f,plist):
    mapresult = {}

    def wrapper(idx):
        param = plist[idx]
        return idx,f(param)

    idxlist = range(len(plist))

    def taskend(request,result):
        idx,res = result
        mapresult[idx] = res

    reqs = tp.makeRequests(wrapper, idxlist, taskend)

    [pool.putRequest(req) for req in reqs]
    pool.wait()

    return mapresult
