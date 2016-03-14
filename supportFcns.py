import numpy as np

def dims(X):
    try:
        (a,b) = X.shape
    except:
        a = 1
        b = X.shape[0]
    return (a,b)

def isequalwithnans(A,B):
    idA = np.isnan(A)
    idB = np.isnan(B)
    Avalid = A[idA]
    Bvalid = B[idB]
    sameLen = len(A)==len(B)
    if sameLen:
        sameVal = all(Avalid==Bvalid)
        samenan = all(idA==idB)
    else:
        sameVal = False
        samenan = False

    return sameLen and sameVal and samenan

def selectalongfirstdimension(IN,f):
    sz = IN.shape()
    dim1 = sz[-1]
    if len(sz)>1:
        dimRest = sz[:-2]
    else:
        dimRest = 1
    tmpMatrix = np.reshape(IN,(dim1, np.prod(np.array(dimRest))))
    yesnan = np.isnan(f)
    if any(yesnan):
        f1 = f
        f1[yesnan] = 1
        tmpMatrix = tmpMatrix[f1,:]
        tmpMatrix[yesnan,:] = np.nan
    else:
        tmpMatrix = tmpMatrix[f,:]

    newsz = np.concatenate((len(f),np.array(dimRest)))
    return np.reshape(tmpMatrix,tuple(newsz))

def selectalonglastdimension(IN,f):
    sz = IN.shape
    dim1 = sz[-1]
    if len(sz)>1:
        dimRest = np.array(sz[:-1])
    else:
        dimRest = np.array([1])

    tmpMatrix = np.reshape(IN,(np.prod(dimRest), dim1))
    yesnan = np.isnan(f)
    if any(yesnan):
        f1 = f[:]
        f1[yesnan] = 1
        tmpMatrix = tmpMatrix[:,f1]
        tmpMatrix[:,yesnan] = np.nan
    else:
        tmpMatrix = tmpMatrix[:,f]
    newsz = list()
    for iR in range(len(dimRest)):
        newsz.append(dimRest[iR])
    newsz.append(len(f))
    out = np.reshape(tmpMatrix,tuple(newsz))
    return out

def normpdf(x,mu,sigma):

    if isinstance(x,(int,float)):
        x = [x]

    x = np.array(x)

    prec = 1/(sigma**2+10**-16)

    dev = x - mu

    denom = ( ((2* np.pi)**(1/2)) * (sigma) )+10**-16

    part1 = 1 / denom
    part2 = (-1/2) * (dev**2)*prec
    p = part1 * np.exp(part2)
    return p


