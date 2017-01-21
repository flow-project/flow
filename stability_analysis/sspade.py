"""
State-Space Pade approximation of time delays

Copyright 2016 Jason M. Sachs

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
import scipy.signal

def conjugate_pairs(poles,reltol=1e-8,abstol=None):
    ''' 
    Find conjugate pole pairs.
    Assume they can probably be sorted by imaginary parts,
    and take advantage of this to reduce execution time,
    but don't rely on it.
    
    Returns a list of tuples i,j
    where poles[i] = conj(poles[j])
    '''
    poles = poles.copy()
    nan=float('nan')
    if abstol is None:
        abstol=reltol
    def ispair(p1,p2):
        return np.abs(np.conj(p2)/p1 - 1) < reltol
    isort = np.argsort(np.imag(poles))
    result = []
    i = 0
    j = len(poles)-1
    poles = poles[isort]
    for i,pi in enumerate(poles):
        if np.imag(pi) > -abstol:
            break
            # stop when we get too close to the real axis
        for k in xrange(j,i,-1):
            pk = poles[k]
            if ispair(pi,pk):
                break
            if np.imag(pk) > -np.imag(pi)+abstol:
                poles[k] = nan
            # not a match, but poles[k] has an imaginary part that is 
            # larger than this or subsequent poles, so we're never going to match it
            # So mark it with a nan 
        else:
            # pole not found
            continue
        result.append((isort[i],isort[k]))
        # Mark the poles we have found
        poles[k] = nan
        if np.isnan(poles[k:j+1]).all():
            # update j if all the trailing poles have been marked
            j = k-1
        i += 1
    return result

def transform_to_real(poles, B=None, verify=False):
    '''
    construct similarity transform T which is unitary,
    and T.H*diag(poles)*T is a real matrix
    '''
    poles = np.array(poles)
    T = np.eye(len(poles),dtype=poles.dtype)
    a = 1/np.sqrt(2)
    b = 1j/np.sqrt(2)
    cpairs = conjugate_pairs(poles)
    for i,j in cpairs:
        T[i,i] =  a
        T[i,j] =  b
        T[j,i] =  b
        T[j,j] =  a
    T = np.matrix(T)
    if verify:
        A = np.diag(poles)
        A2 = T.H*A*T
        assert np.amax(np.abs(np.imag(A2))) < 1e-13
    if B is None:
        return T
    # extended version: B is the B matrix in state-space
    KB = np.ones_like(B)
    for i,j in cpairs:
        ki = 1.0
        kj = 1j*np.conj(B[i,0]*ki)/B[j,0]
        KB[i,0] = ki
        KB[j,0] = kj
    return T, KB

def ss2_wellcond(p,num,as_abcd=False):
    # state-space 2nd order system
    # with well-conditioned basis
    # p = one of the complex poles
    # num = numerator polynomial
    nnum = len(num)
    assert 1 <= nnum <= 3
    num = np.array(num)
    sigma = -np.real(p)
    omega = np.imag(p)
    den = np.array([1,2*sigma,sigma**2 + omega**2])
    if nnum == 3:
        D = [num[0]]
        num -= D*den
        a = num[1]
        b = num[2]
    else:
        D = [0]
        if nnum == 2:
            a,b = num
        else:
            a,b = 0,num[0]
    A = np.array([[-sigma,omega],[-omega,-sigma]])
    M = a
    K = 1.0*(a*sigma-b)/omega
    B = np.transpose([[1.0,0]])
    C = np.array([M,K])
    if as_abcd:
        return A,B,C,D
    else:
        return scipy.signal.lti(A,B,C,D)

def get_ABCD(sys):
    if isinstance(sys, scipy.signal.lti):
        return sys.A,sys.B,sys.C,sys.D
    else:
        # A,B,C,D
        return sys

def cascade(*systems, **kwargs):
    def mmul(A,B):
        return np.asarray(np.matrix(A)*np.matrix(B))
    A1,B1,C1,D1 = [],[],[],1
    first = True
    for S in systems:
        if first:
            A1,B1,C1,D1 = [np.atleast_2d(M) for M in get_ABCD(S)]
            nx1 = A1.shape[0]
            first = False
            continue
        A2,B2,C2,D2 = get_ABCD(S)
        nx2 = A2.shape[0]
        NW=A1
        NE=np.zeros((nx1,nx2))
        SW=mmul(B2, C1)
        SE=A2
        A = np.vstack([np.hstack([NW,NE]),np.hstack([SW,SE])])
        B = np.vstack([B1,mmul(B2,D1)])
        C = np.hstack([mmul(D2,C1), np.atleast_2d(C2)])
        D = mmul(D2,D1)
        A1,B1,C1,D1 = A,B,C,D
        nx1 += nx2
    if kwargs.get('as_abcd',False):
        S1 = A1,B1,C1,D1
    else:
        S1 = scipy.signal.lti(A1,B1,C1,D1)
    return S1
    
def _calc_residuals(zeros,poles,k):
    ''' calculate complex residues of zeros, poles, gain '''
    q = len(poles)
    p = len(zeros)
    n = max(p,q)
    c = 1.0*k
    for j in xrange(n):
        if j < q:
            a = poles - poles[j]
            a[j] = 1.0
            c /= a
        if j < p:
            c *= (poles - zeros[j])
    return c

def zpk2ss_modal(zeros, poles, k, residuals=None, as_matrices=False):
    '''
     convert zeros poles k representation to state-space
     with a modal system (diagonal A matrix)
    '''
    if residuals is None:
        residuals = _calc_residuals(zeros, poles, k)
    npoles = len(poles)
    nzeros = len(zeros)
    assert npoles >= nzeros
    kfeedthrough = 0 if npoles > nzeros else k

    Am = np.diag(poles)
    rsqrt = np.sqrt(residuals)
    Bm = np.matrix(rsqrt).T
    Cm = rsqrt
    Dm = kfeedthrough
    if as_matrices:
        return Am,Bm,Cm,Dm
    else:
        return scipy.signal.lti(Am,Bm,Cm,Dm)
    
def zpk2ss_realmodal(zeros, poles, k, residuals=None, imag_limit=None, 
                as_matrices=False):
    '''
     convert zeros poles k representation to state-space
     but with real coefficients only (tridiagonal matrix)
     See https://ccrma.stanford.edu/~jos/StateSpace/Similarity_Transformations.html
        and http://math.stackexchange.com/questions/1598826
        and http://math.stackexchange.com/questions/1598919
        '''
    Am,Bm,Cm,Dm = zpk2ss_modal(zeros, poles, k, residuals, as_matrices=True)
    T,KB = transform_to_real(poles, B=Bm)
    Ar = np.asarray(T.H*Am*T)
    Br = np.asarray(T.H*np.multiply(Bm,KB))
    Cr = np.asarray(np.divide(Cm,KB.T)*T)
    Dr = Dm
    if imag_limit is not None:
        if imag_limit == True:
            imag_limit = 1e-9
        def assert_real(M):
            maximag = np.amax(np.abs(np.imag(M)))
            assert maximag < imag_limit,maximag
        assert_real(Ar)
        assert_real(Br)
        assert_real(Cr)
        assert_real(Dr)
    Ar = np.real(Ar)
    Br = np.real(Br)
    Cr = np.real(Cr)
    Dr = np.real(Dr)
    if as_matrices:
        return Ar,Br,Cr,Dr
    else:
        return scipy.signal.lti(Ar,Br,Cr,Dr)

def _calc_ss_matrices(z,p):
    if np.imag(p) < 1e-9:
        # 1st-order system
        assert z is None or np.imag(z) < 1e-9
        p = np.real(p)
        A = np.array([p])
        B = np.array([1.0])
        if z is None:
            C = -A
            D = 0
        else:
            z = np.real(z)
            D = p/z
            C = D*(p-z)*B
    else:
        # 2nd-order system
        p2 = np.abs(p)**2
        if z is None:
            num = np.array([p2])
        elif np.abs(np.imag(z)) < 1e-9:
            # numerator is 1st-order
            z = np.real(z)
            num = -np.array([p2/z,-p2])
        else:
            # numerator is 2nd-order
            z2 = np.abs(z)**2
            sigma = -np.real(z)
            num = p2/z2*np.array([1,2*sigma,z2])
        A,B,C,D=ss2_wellcond(p,num,as_abcd=True)            
    return A,B,C,D
    
def _canonicalize_roots(roots, rootsorter=None):
    if len(roots) < 2:
        return roots
    if rootsorter is None:
        rootsorter = lambda z: np.imag(z)
    return sorted(roots[np.imag(roots) >= -1e-9], key=rootsorter)

def zpk2ss_cascade(zeros, poles, k, as_matrices=False, rootsorter=None):
    '''
    Final approach: use state-space cascade of 2nd-order systems
    by sorting poles and zeros
    '''
    zlist = _canonicalize_roots(zeros, rootsorter)
    plist = _canonicalize_roots(poles, rootsorter)
    zlist = [None]*(len(plist)-len(zlist)) + list(zlist)
    speclist = [_calc_ss_matrices(z,p) for z,p in zip(zlist,plist)]
    return cascade(*speclist)

class PadeExponential(object):
    def __init__(self, p, q, f0=1.0, rootsorter=None):
        builder = PadeExponentialBuilder(p,q,f0)
        self.p = builder.p
        self.q = builder.q
        self.f0 = f0
        self.num, self.den = builder.ratcoeffs
        self.zeros, self.poles, self.k = builder.zpk
        self.residuals = _calc_residuals(self.zeros,self.poles,self.k)
        self.rootsorter = (rootsorter if rootsorter is not None
                           else lambda z: np.imag(z)) 
    @property
    def lti_scipy_coeff(self):
        '''
        First implementation: brute-force LTI system via coefficients 
        (very ill-conditioned)
        '''
        return scipy.signal.lti(self.num,self.den)
    @property
    def lti_scipy_zpk(self):
        '''
        Second implementation: LTI via zeros,poles,gain
        '''
        return scipy.signal.lti(self.zeros, self.poles, self.k)
    @property
    def lti_ssmodal(self):
        '''
        Third implementation: LTI via state-space modal representation
        '''
        return zpk2ss_modal(self.zeros, self.poles, self.k, self.residuals)
    def get_lti_realmodal(self, imag_limit=None):
        '''
        Same as lti_modal() but with a similarity transform 
        to keep all coefficients real.
        '''
        return zpk2ss_realmodal(self.zeros, self.poles, self.k, self.residuals,
                             imag_limit=imag_limit)
    @property
    def lti_ssrealmodal(self):
        return self.get_lti_realmodal()


    @property
    def lti_sscascade(self):
        '''
        Final approach: use state-space cascade of 2nd-order systems
        by sorting poles and zeros
        '''
        return zpk2ss_cascade(self.zeros, self.poles, self.k,
                                rootsorter=self.rootsorter)
        
    @property
    def zpk(self):
        return self.zeros, self.poles, self.k
    def _test_residuals(self, x, max_err=1e-10):
        pp = self.poles
        pz = self.zeros
        k = self.k
        r = self.residuals

        y1 = sum(r_j/(x-pp_j) for r_j,pp_j in zip(r,pp))
        y2 = k
        for pp_j in pp:
            y2 /= (x-pp_j)
        for pz_j in pz:
            y2 *= (x-pz_j)
        err = np.max(np.abs(y2-y1))
        assert err < max_err,  err
        return y1,y2
    def __call__(self, s):
        '''
        Evaluate transfer function
        '''
        pp = self.poles
        pz = self.zeros
        k = self.k
        y = 1.0
        for i in xrange(max(len(pp),len(pz))):
            if i < len(pz):
                k *= pz[i]
                y *= (1 - s/pz[i])
            if i < len(pp):
                k /= pp[i]
                y /= (1 - s/pp[i])
        return y

class PadeExponentialBuilder(object):
    def __init__(self,p,q,f0=1.0):
        self.p = p
        self.q = q
        self.f0 = f0
    def _h0(self,v,p,q):
        ''' find F(v) where F(v)=0 for vector of roots '''
        vj,vk=np.meshgrid(v,v)
        d = vj-vk
        np.fill_diagonal(d,float('inf'))
        H = 1.0/d
        F = sum(H, 0) - 0.5 - (p+q)/(2.0*v) 
        return F
    def _h1(self,v,p,q):
        ''' find Jacobian of F(v) '''
        vj,vk=np.meshgrid(v,v)
        d = vj-vk
        np.fill_diagonal(d,float('inf'))
        H = 1.0/d
        H2 = H**2
        dF = H2 + np.diag(-sum(H2,0)+(p+q)/2.0/(v**2))
        return dF
    def _roots_iteration(self,v,p,q):
        ''' 
        one step of Newton's method for solving roots v;
        combines the _h0 and _h1 methods
        '''
        vj,vk=np.meshgrid(v,v)
        d = vj-vk
        np.fill_diagonal(d,float('inf'))
        H = 1.0/d
        F = sum(H, 0) - 0.5 - (p+q)/(2.0*v) 
        H2 = H**2
        dF = H2 + np.diag(-sum(H2,0)+(p+q)/2.0/(v**2))
        dv = np.linalg.solve(dF, F)
        return v - dv
    def poles(self,p,q,nstep=15):
        ''' find poles of Pade approximation of e^-s ''' 
        u = (2.0*np.arange(q) - (q-1))/q
        rx = -p - q/3.0 + 2*u*u*q/3.0
        ry = 2*u*(p+q)/3.0
        v = rx + 1j*ry
        for k in xrange(nstep):
            v = self._roots_iteration(v,p,q)
        return v*self.f0
    def get_zpk(self,p,q):
        ''' find zeros, poles, gain '''
        poles = self.poles(p,q)
        zeros = -self.poles(q,p)
        K = 1.0
        for j in xrange(max(p,q)):
            if j < q:
                K *= -poles[j]
            if j < p:
                K /= -zeros[j]
        return zeros, poles, K
    @property
    def zpk(self):
        ''' zeros, poles, gain '''
        return self.get_zpk(self.p, self.q)
    @property
    def ratcoeffs(self):
        ''' polynomial coefficients of num/den '''
        p,q = self.p,self.q
        n = max(p,q)
        c = 1
        d = 1
        f = self.f0
        clist = [c]
        dlist = [d]
        for k in xrange(1,n+1):
            c *= -1.0/f*(p-k+1)/(p+q-k+1)/k
            if k <= p:
                clist.append(c)
            d *= 1.0/f*(q-k+1)/(p+q-k+1)/k
            if k <= q:
                dlist.append(d)
        return np.array(clist[::-1]),np.array(dlist[::-1])
        
class Bessel(object):
    def __init__(self,n,f0=1.0):
        self.n = n
        self.f0 = f0
        peb = PadeExponentialBuilder(n,n,f0/2.0)
        poles = peb.poles(n,n)
        self.poles = poles
        k = 1.0
        for p in poles:
            k *= p
        self.k = np.real(k)
        self.zeros = np.array([])
    @property
    def zpk(self):
        return self.zeros, self.poles, self.k
    @property
    def lti_sscascade(self):
        return zpk2ss_cascade(self.zeros, self.poles, self.k)

        