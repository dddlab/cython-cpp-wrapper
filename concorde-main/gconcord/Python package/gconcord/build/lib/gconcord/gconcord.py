import numpy as np
import numpy.ctypeslib as npct
import site
import os.path
import sys
from ctypes import c_int, c_double

array_1d_double = npct.ndpointer(dtype=np.double, ndim=1, flags='CONTIGUOUS')
array_1d_int = npct.ndpointer(dtype=c_int, ndim=1, flags='CONTIGUOUS')


def gconcord(S, lam1, lam2, method = "ista", tol = 1e-5, maxit = 100, steptype = 0):
    
    path = findpath()              ## find path of shared library
    p = S.shape[0]                 ## data dimensionality
    sdiv = np.sqrt( 1/np.diag(S) ) ## inverse of standard deviation
    SamCor = (sdiv * S).T * sdiv   ## sample correlation matrix
    s = SamCor.reshape(p**2,)      ## flatten sample correlation

    libcd = npct.load_library("sharedlib.so", path)
    libcd.gconcord.restype = None
    libcd.gconcord.argtypes = [array_1d_double, c_int, c_int, 
                               array_1d_double, c_double, c_double, c_int, c_int,
                               array_1d_double, array_1d_int, array_1d_int]
    
    lambda1 = checklam(lam1, p)    ## convert lambda1
    mth = checkmethod(method)      ## determine optimization method
    
    out, outi, outj = createout(p)
    
    libcd.gconcord(s, p, mth, lambda1, lam2, tol, maxit, steptype, out, outi, outj)
    
    omegacor = createomega(out, outi, outj, p)
    omega = (sdiv * omegacor).T * sdiv
    
    return omega
    
    
    
def createomega(out, outi, outj, p):
    
    omega = np.zeros((p,p))
    for j in range(p**2):
        if outi[j] != -1:
            omega[outi[j], outj[j]] = out[j]
        else:
            break
    return omega
    
    
def createout(p):
    
    out = np.diag(np.repeat(1.0, p)).reshape(p**2,) + np.repeat(1.0, p**2)
    outi = np.array(np.repeat(-1, p**2), dtype = np.int32)
    outj = np.array(np.repeat(-1, p**2), dtype = np.int32)
    return out, outi, outj

    
def checkmethod(method):
    
    if method.lower() == "coordinatewise":
        return 1
    elif method.lower() == "ista":
        return 2
    elif method.lower() == "fista":
        return 3
    else:
        raise Exception('Incorrect method input.')
        
    
def checklam(lam1, p):
    
    if isinstance(lam1, float) | isinstance(lam1, int):
        lambda1 = ( np.repeat(1.0, p**2).reshape(p, p) - np.identity(p) ) * lam1
    elif isinstance(lam1, np.ndarray):
        lambda1 = lam1 - np.diag( np.diag(lam1) )
    else:
        raise Exception('lam1 should be either a number or an ndarray.')
    return lambda1.reshape(p**2,)
        
    
def findpath():
    
    paths = sys.path
    if len(paths) == 0:
        raise Exception('No path exists. Check with sys.path.')
    for path in paths:
        if os.path.isfile(path+"/gconcord/sharedlib.so"):
            return path+"/gconcord/sharedlib.so"
    raise Exception("No shared library in any paths. Please check the installation.")