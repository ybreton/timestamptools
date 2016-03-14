import numpy as np
from supportFcns import *
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import convolve2d

class ts:
    def shape(self):
        (d,n) = dims(self.data())
        if isinstance(d,(int,float)):
            d = (d,)
        return (d,n)

    def tsOK(self):
        (d,n) = dims(self.range())
        return n==0 or d==1

    def OK(self):
        return self.tsOK()

    def starttime(self):
        (n,d) = dims(self.t)
        if n<1:
            return np.nan
        else:
            return float(self.t[0])

    def endtime(self):
        (n,d) = dims(self.t)
        if n<1:
            return np.nan
        else:
            return float(self.t[-1])

    def range(self):
        return np.unique(self.t)


    def dt(self):
        return np.nanmedian(np.diff(self.range()))


    def data(self,t=None,extrapolate=np.nan,return_indices=False):
        if t is None:
            d = self.range()
        else:
            T = self.range()
            nT = len(T)
            idrange = np.arange(0,nT)
            fint = interp1d(T,idrange)
            ix = (np.round(fint(t))).astype(int)
            in_range = np.logical_and(ix>=0, ix<nT)
            if extrapolate is None or extrapolate is False:
                ix = ix[in_range]
                d = T[ix]
            elif extrapolate is True:
                ix[t<T[0]] = 1
                ix[t>T[-1]] = nT
                d = T[ix]
            else:
                d = np.ones(ix.shape)*extrapolate
                d[in_range] = T[ix[in_range]]

        if return_indices:
            return (d,ix)
        else:
            return d


    def restrict(self,t0,t1):
        if not (isinstance(t0,(int,float)) and isinstance(t1,(int,float))):
            assert len(t0) == len(t1), 't0 and t1 must have equal length.'
            nT = len(t0)
        else:
            t0 = [t0]
            t1 = [t1]
            nT = 1

        n = self.nD
        keep = np.ones((n,1))*False
        for iT in range(nT):
            keep = np.logical_and(self.t>=t0[iT],self.t<=t1[iT])

        return ts(self.t[keep])


    def removeNaNs(self):
        T = self.range()
        D = self.data()
        (d,n) = self.shape()
        idnan = np.isnan(D)
        D = D[np.logical_not(idnan)]
        T = T[np.logical_not(idnan)]

        if isinstance(self,ts):
            tso = ts(T)
        if isinstance(self,(tsd,ctsd)):
            tso = tsd(T,D)
        return tso


    def smooth(self,sigma=0.1,window=0.2):

        nW = np.ceil(window/self.dt())
        w = [iN*self.dt() for iN in np.linspace(-nW,nW,2*nW+1)]
        SmoothingWindow = normpdf(w,0,sigma)
        SmoothingWindow = SmoothingWindow/np.nansum(SmoothingWindow)
        SmoothingWindow = SmoothingWindow.reshape((SmoothingWindow.size,1))
        Data = self.data()
        sd = convolve2d(Data,SmoothingWindow,mode='same')
        tso = tsd(self.range(),sd)
        return tso

    def __eq__(self,other):
        return (len(self.t)==len(other.t)) and all(self.t==other.t)


    def __init__(self,T):
        if isinstance(T,tsd):
            T = tsd.t
        if isinstance(T,ctsd):
            T = [ctsd.t0+iN*ctsd.dT for iN in range(ctsd.nD)]

        # if isinstance(T,(float,int)):
        #     T = n[T]
        #
        # if not isinstance(T,(np.ndarray, np.generic)):
        #     T = np.array(T)

        T = np.array(list(T))

        (d,n) = dims(T)
        if n==1:
            T = T.reshape(n,d)
            (d,n) = dims(T)

        T.sort()
        self.t = T
        self.nD = n
        assert(self.tsOK), 'ts component not OK'


    def __str__(self):
        string = '<{0:d}-timestamp object>\n'.format(self.nD)
        string = string+'starttime:{0:.2f}, endtime:{1:.2f}, dt:{2:.2f}'.format(self.starttime(),self.endtime(),self.dt())
        return string

    def __repr__(self):
        string = '<{0:d}-timestamp object>\n'.format(self.nD)
        string = string+'t='+str(self.range())+'\n'
        return string

    def plot(self):
        ph=plt.scatter(self.range(),np.ones((self.range()).shape))
        return ph


class tsd(ts):
    def tsdOK(self):
        ok = self.tsOK()
        (d,n) = dims(self.data())
        return ok and n==len(self.range())

    def data(self,t=None,extrapolate=np.nan,return_indices=False):
        if t is None:
            t = self.range()
        (_t, ix) = ts.data(self, t, extrapolate=extrapolate, return_indices=True)
        d = selectalonglastdimension(self.D, ix)

        if return_indices:
            return (d,ix)
        else:
            return d

    def restrict(self,t0,t1):
        trestrict = ts.restrict(self,t0,t1)
        Rt = trestrict.t
        Rd = self.data(Rt)
        return tsd(Rt,Rd)


    def __init__(self,T,D):

        ts.__init__(self,T)

        self.D = D
        (dims,n) = self.shape()
        self.dims = dims
        assert self.tsdOK(), 'tsd not OK.'

    def __eq__(self,other):
        return (len(self.t)==len(other.t)) and (all(self.t==other.t)) and (len(self.D)==len(other.D)) and (all(self.D[idvalid]==other.D[idvalid])) and isequalwithnans(self.D,other.D)


    def __str__(self):
        ndim = len(self.dims)
        string = '<{1:d}-D, {0:d}-timestamped data object>\n'.format(self.nD,ndim)
        string = string+'starttime:{0:.2f}, endtime:{1:.2f}, dt:{2:.2f}'.format(self.starttime(),self.endtime(),self.dt())
        return string

    def __repr__(self):
        string = '<{1:d}-D, {0:d}-timestamped data object>\n'.format(self.nD,ndim)
        string = string+'t='+str(self.range())+'\n'
        string = string+'d='+str(self.data())+'\n'
        return string

    def plot(self):
        ph=plt.scatter(self.range(),self.data())
        return ph

class ctsd(tsd):
    def isOK(self):
        ok1 = isinstance(self.t0,(int,float))
        ok2 = isinstance(self.dT,(int,float))
        if not ok1:
            ok1 = len(self.t0)==1
        if not ok2:
            ok2 = len(self.dT)==1
        return ok1 and ok2

    def starttime(self):
        return self.t0

    def endtime(self):
        (d,n) = dims(self.D)
        return self.t0+n*self.dT


    def range(self):
        (d,n) = dims(self.D)
        return np.array([self.t0+self.dT*iN for iN in range(n)])


    def dt(self):
        return self.dT

    def data(self, t=None, extrapolate=np.nan, return_indices=False):
        if t is None:
            d = self.D
            ix = np.arange(len(d))
        else:
            t = np.array(t)
            t0 = self.starttime()
            t1 = self.endtime()
            ix = np.array(np.round((t-t0)/self.dT)).astype(int)
            in_range = np.logical_and(t>=t0, t<= t1)

            if extrapolate is None:
                ix = ix[in_range]
            elif extrapolate is False:
                ix[t<t0] = 0
                ix[t>t1] = self.nD
            else:
                if any(np.logical_not(in_range)):
                    ix[np.logical_not(in_range)] = extrapolate

            d = selectalonglastdimension(self.D,ix)
        if return_indices:
            return (d,ix)
        else:
            return d


    def restrict(self,t0,t1):
        t = self.range()
        d = self.data()
        tso = tsd(t,d)
        return tso.restrict(t0,t1)


    def __init__(self,t0,dt=None,d=None):
        if d is not None and dt is not None:
            self.t0 = t0
            self.dT = dt
            d = np.array(d)
            try:
                (m,n) = d.shape
            except:
                m = 1
                n = d.shape[0]
            assert m==1, 'ctsd only supports 1d continuous data.'
            d = d.reshape(n)

            n = len(d)
            t = np.array([t0+iN*dt for iN in range(n)])
            self.f = interp1d(t,d)
            self.D = (self.f(t)).reshape((1,t.size))
        elif isinstance(t0,(ts,tsd,ctsd)):
            self.t0 = t0.starttime()
            self.dT = t0.dt()
            self.f = interp1d(t0.range(),t0.data())
            n = np.ceil((t0.endtime()-t0.starttime())/t0.dt()).astype(int)
            t = np.array([t0.starttime()+iN*t0.dt() for iN in range(n)])
            self.D = (self.f(t)).reshape((1,t.size))

        (d,n) = dims(self.D)
        self.dims = d
        self.nD = n
        assert self.isOK()


    def __eq__(self,other):
        if not isinstance(other,tsd.tsd):
            other = ctsd.ctsd(other)
        return self.t0==other.t0 and self.dT==other.dT and isequalwithnans(self.D,other.D)


    def __str__(self):
        string = '<continuously-timestamped data object>\n'
        string = string+'starttime:{0:.2f}, endtime:{1:.2f}, dt:{2:.2f}'.format(self.starttime(),self.endtime(),self.dt())
        return string

    def __repr__(self):
        string = '<continuously-timestamped data object>\n'
        string = string+'t0='+str(self.starttime())+'\n'
        string = string+'dt='+str(self.dt())+'\n'
        string = string+'d='+str(self.data())+'\n'
        return string

    def plot(self):
        t = self.range()
        d = (self.data()).reshape(t.shape)

        ph=plt.plot(t,d,'bo-')
        return ph



if __name__ == '__main__':
    print('Self test...')
    print('*'*80)
    t = np.linspace(0,10,1001)
    print('Time stamps:')
    print(t)
    d = np.random.randn(1,1001)
    print('Data:')
    print(d)

    print('\n')
    print('Time stamped array ts:')
    print('*'*80)
    timeStampObj = ts(t)
    print('timeStampObj = ts(t)')
    print(timeStampObj)
    print('timeStampObj.shape() = ')
    print(timeStampObj.shape())
    print('timeStampObj.range() = ')
    print(timeStampObj.range())
    print('timeStampObj.restrict(3,7) = ')
    print(timeStampObj.restrict(3,7))
    print('timeStampObj.data([5,6]) = ')
    print(timeStampObj.data([5,6]))
    print('*'*80)
    print('Time-stamped data tsd:')
    print('\n')
    timeStampData = tsd(t,d)
    print('timeStampData = tsd(t,d)')
    print(timeStampData)
    print('timeStampData.shape() = ')
    print(timeStampData.shape())
    print('timeStampData.range() = ')
    print(timeStampData.range())
    print('timeStampData.restrict(3,7) = ')
    print(timeStampData.restrict(3,7))
    print('timeStampData.data([5,6]) = ')
    print(timeStampData.data([5,6]))
    print('timeStampData.smooth() = ')
    print(timeStampData.smooth())
    print('*'*80)
    print('Continuously sampled data ctsd:')
    print('\n')
    csd = ctsd(timeStampData)
    print('csd = ctsd(timeStampData)')
    print(csd)
    print('csd.data() = ')
    print(csd.data())
    print('csd.range() = ')
    print(csd.range())
    print('csd.data([5,6]) = ')
    print(csd.data([5,6]))
    print('csd.restrict(3,7) = ')
    print(csd.restrict(3,7))
    print('csd.smooth() = ')
    print(csd.smooth())
    print('*'*80)
    print('Continuously sampled data ctsd:')
    print('\n')
    csd = ctsd(t[0],dt=0.01,d=d)
    print('csd = ctsd(t[0],dt=0.01,d=d)')
    print(csd)
    print('csd.data() = ')
    print(csd.data())
    print('csd.range() = ')
    print(csd.range())
    print('csd.data([5,6]) = ')
    print(csd.data([5,6]))
    print('csd.restrict(3,7) = ')
    print(csd.restrict(3,7))
    print('csd.smooth() = ')
    print(csd.smooth())

    fh1 = plt.figure()
    fh1.suptitle('Time stamp')
    timeStampObj.plot()

    fh2 = plt.figure()
    fh2.suptitle('Time-stamped data')
    timeStampData.plot()

    fh3 = plt.figure()
    fh3.suptitle('Continuous time-stamped data')
    csd.plot()

    fh4 = plt.figure()
    fh4.suptitle('Smoothed, continuous time-stamped data')
    (ctsd(csd.smooth())).plot()

    plt.show()