# Basic SINDy regression tools

import numpy as np
import scipy as sc
from tqdm import tqdm,tnrange
import copy
from matplotlib import pyplot as plt
    
##########################################################################################################################################

def prinfo(correct,Xnf,X,dt,Ni,M,reg=0,lam=0,div=0):
    
    """
    Returns features relating to data's linear regression solvability.
    
    Inputs:
    Returns: (n datapoints, model order, )
    """
    
    info={}
    
    # 1) # of data points
    N=X.shape[0]
    info['N']=N
    
    # 2) Sampling rate (dt)
    info['dt']=dt
    
    # 3) Model order
    info['M']=M
    
    # 4) Ridge penalty
    info['reg']=reg
    
    # 5) ST threshold
    info['lam']=lam
    
    # 7) Noise magnitude
    info['nlevel']=np.ptp(X-Xnf)
    
    # 8) # of initial conditions
    info['Ni']=Ni
    
    # 9) # of parameters
    Theta=polypool(X,M)
    n_col=Theta.shape[1]
    info['npar']=n_col
    
    # 10) ratio of data to unkown parameters
    info['N/npar']=N/n_col
    
    # 11) Data info (mean, std, range, min/max)
    info['Dmean']=np.average(X,axis=0)
    info['Dstd']=np.std(X,axis=0)
    info['Dmin']=np.min(X,axis=0)
    info['Dmax']=np.max(X,axis=0)
    info['Drange']=np.ptp(X,axis=0)
    
    # 12) Theta info (mean, std, range, min/max)
    info['Tmean']=np.average(Theta,axis=0)
    info['Tstd']=np.std(Theta,axis=0)
    info['Tmin']=np.min(Theta,axis=0)
    info['Tmax']=np.max(Theta,axis=0)
    info['Trange']=np.ptp(Theta,axis=0)
    
    dXdt=np.gradient(X,dt,axis=0)
    n=dXdt.shape[-1]
    for i in range(Ni):
        dXdt[int(i*N/Ni):int((1+i)*N/Ni)]=np.gradient(X[int(i*N/Ni):int((1+i)*N/Ni)],dt,axis=0)
    
    # 14) (log10 of) colleration matrix conditon
    W=np.diag(1/np.max(np.abs(Theta),axis=0))
    Thetaw=Theta@W
    ThetawTThetaw=Thetaw.T.dot(Thetaw)
    ThetawTdXdt=Thetaw.T.dot(dXdt)
    info['l10cond']=np.log10(np.linalg.cond(ThetawTThetaw+reg*np.identity(n_col)))
    
    # 15) LS solution residual
    Xi=W@np.linalg.lstsq(ThetawTThetaw+reg*np.identity(n_col),ThetawTdXdt,rcond=-1)[0]
    info['LSres']=np.sum(np.abs(dXdt-Theta@Xi))/N
    if not correct.shape==Xi.shape:
        correct=np.vstack((np.nan_to_num(correct/correct),np.zeros((Xi.shape[0]-correct.shape[0],correct.shape[1]))))
    info['LSmerr']=np.sum(np.abs(correct-Xi))
    #print(Xi)
    
    # 16) ST solution residual
    Xip=np.copy(Xi)
    
    flag=False
    while not flag:
        smallinds=abs(Xi)<=lam
        Xi[smallinds]=0
        for ind in range(n):
            biginds=(smallinds==False)[:,ind]
            Xi[biginds,ind]=np.linalg.lstsq(ThetawTThetaw[:,biginds][biginds,:]+reg*np.identity(biginds.sum())
                                            ,ThetawTdXdt[biginds,ind],rcond=-1)[0]
        Xi=W@Xi
        flag=np.array_equal(Xip,Xi)
        Xip=np.copy(Xi)
        
    info['STres']=np.sum(np.abs(dXdt-Theta@Xi))/N
    info['STcorr']=np.all(np.nan_to_num(correct/correct)==np.nan_to_num(Xi/Xi))
    info['STmerr']=np.sum(np.abs(correct-Xi))
    #print(Xi)
    
    # 17) CV solution residual
    if div==0:
        info['CVres']=0
        info['CVcorr']=0
        info['CVmerr']=0
    else:
        Xi=SINDyCV(Theta,dXdt,0.01,div,reg,1e-16,0.9,0.1,ptf=1,normalise='auto')
        info['CVres']=np.sum(np.abs(dXdt-Theta@Xi))/N
        info['CVcorr']=np.all(np.nan_to_num(correct/correct)==np.nan_to_num(Xi/Xi))
        info['CVmerr']=np.sum(np.abs(correct-Xi))
    #print(Xi)
    
    #print(info)
    return info

##########################################################################################################################################

def polypool(X,mo):
    
    """
    Generate polynomial pool of given data X, uptill maximum power, mo.
    
    Inputs: (X,mo)
    Retruns: Theta
    
    'X' is a set of datapoints stacked vertically.
    
    'Theta' is returned as a vertically stacked polynomial augmentation of datapoints in 'X'.
    """
    
    mo=mo+1
    
    Theta=np.ones((len(X),1))
    
    from itertools import combinations_with_replacement
    
    for crl in range(1,mo):
        coms = combinations_with_replacement(range(len(X[0])), crl)
        
        for com in list(coms):            
            new=np.ones((len(X),1))
            
            for j in com:
                new=new*X[:,j].reshape(-1,1)
            
            Theta=np.hstack((Theta,new))
    
    return Theta

##########################################################################################################################################

def polypool_names(states,mo):
    
    """
    Generate names of polynomial terms of given state names, uptill maximum power, mo.
    
    Inputs: (states,mo)
    Retruns: theta_names
    """
    
    mo=mo+1
    
    theta_names=[1]
    
    from itertools import combinations_with_replacement
    
    for crl in range(1,mo):
        coms = list(combinations_with_replacement(range(len(states)), crl))
        
        for com in coms:
            name='$'
            
            for i in range(len(states)):
                if (np.array(com)==i).sum()!=0:
                    name=name+states[i]+'^'+str(int((np.array(com)==i).sum()))
                
            name=name+'$'
            
            theta_names.append(name)
    
    return len(states)*theta_names

##########################################################################################################################################

def polypool_(X,mo):
    
    """
    Generate polynomial pool of given data X (without variable interaction), uptill maximum power, mo.
    
    Inputs: (X,mo)
    Retruns: Theta
    
    'X' is a set of datapoints stacked vertically.
    
    'Theta' is returned as a vertically stacked polynomial augmentation of datapoints in 'X'.
    """
    
    mo=mo+1
    
    Theta=np.ones((len(X),1))
    
    for crl in range(1,mo):
        
        Theta=np.hstack((Theta,X**crl))
    
    return Theta

##########################################################################################################################################

def polypool_names_(states,mo):
    
    """
    Generate names of polynomial terms (generated without variable interaction) of given state names, uptill maximum power, mo.
    
    Inputs: (states,mo)
    Retruns: theta_names
    """
    
    mo=mo+1
    
    theta_names=[1]
    
    for crl in range(1,mo):
        for statevar in states:
            name='$'
            name=name+statevar+'^'+str(crl)
            name=name+'$'
        
            theta_names.append(name)
    
    return len(states)*theta_names

##########################################################################################################################################

def rpolypool(X,mo):
    
    """
    Generate rational polynomial pool of given data X, uptill maximum power, mo.
    
    Inputs: (X,mo)
    Retruns: Theta
    
    'X' is a set of datapoints stacked vertically.
    
    'Theta' is returned as a vertically stacked polynomial augmentation of datapoints in 'X'.
    """
    
    mo=mo+1
    
    Theta=np.ones((len(X),1))
    
    from itertools import combinations_with_replacement
    
    for crl in range(1,mo):
        coms = combinations_with_replacement(range(len(X[0])), crl)
        
        for com in list(coms):            
            new=np.ones((len(X),1))
            
            for j in com:
                new=new*X[:,j].reshape(-1,1)
            
            Theta=np.hstack((Theta,new,1/new))
    
    return Theta

##########################################################################################################################################

def rpolypool_names(states,mo):
    
    """
    Generate names of rational polynomial terms of given state names, uptill maximum power, mo.
    
    Inputs: (states,mo)
    Retruns: theta_names
    """
    
    mo=mo+1
    
    theta_names=[1]
    
    from itertools import combinations_with_replacement
    
    for crl in range(1,mo):
        coms = list(combinations_with_replacement(range(len(states)), crl))
        
        for com in coms:
            name='$'
            
            for i in range(len(states)):
                if (np.array(com)==i).sum()!=0:
                    name=name+states[i]+'^'+str(int((np.array(com)==i).sum()))
                
            name=name+'$'
            
            theta_names.append(name)
            
            name='$1/'
            
            for i in range(len(states)):
                if (np.array(com)==i).sum()!=0:
                    name=name+states[i]+'^'+str(int((np.array(com)==i).sum()))
                
            name=name+'$'
            
            theta_names.append(name)
    
    return len(states)*theta_names

##########################################################################################################################################

def hc_dxc_polypool(X,mo,l1,l2):
    """
    Generate polynomial pool of...
    
    Inputs: (X,mo)
    Retruns: Theta
    
    'X' is a set of datapoints stacked vertically.
    
    'Theta' is returned as a vertically stacked polynomial augmentation of datapoints in 'X'.
    """
    
    #kf*(-x[0]+x[4]-x[2]*l1)+cf*(-x[1]+x[5]-x[3]*l1)+kr*(-x[0]+x[6]+x[2]*l2)+cr*(-x[1]+x[7]+x[3]*l2)
    hcdX=np.array([-X[:,0]+X[:,4]-X[:,2]*l1,-X[:,1]+X[:,5]-X[:,3]*l1,-X[:,0]+X[:,6]+X[:,2]*l2,-X[:,1]+X[:,7]+X[:,3]*l2]).T
    
    mo=mo+1
    
    Theta=np.ones((len(X),1))
    
    for crl in range(1,mo):
        
        Theta=np.hstack((Theta,hcdX**crl))
    
    return Theta

def hc_dthetac_polypool(X,mo,l1,l2):
    """
    Generate polynomial pool of...
    
    Inputs: (X,mo)
    Retruns: Theta
    
    'X' is a set of datapoints stacked vertically.
    
    'Theta' is returned as a vertically stacked polynomial augmentation of datapoints in 'X'.
    """
    
    #l1*(kf*(-x[0]+x[4]-x[2]*l1)+cf*(-x[1]+x[5]-x[3]*l1))+l2*(kr*(x[0]-x[6]-x[2]*l2)+cr*(x[1]-x[7]-x[3]*l2))
    hcdX=np.array([-X[:,0]+X[:,4]-X[:,2]*l1,-X[:,1]+X[:,5]-X[:,3]*l1,X[:,0]-X[:,6]-X[:,2]*l2,X[:,1]-X[:,7]-X[:,3]*l2]).T
    
    mo=mo+1
    
    Theta=np.ones((len(X),1))
    
    for crl in range(1,mo):
        
        Theta=np.hstack((Theta,hcdX**crl))
    
    return Theta

def hc_dxf_polypool(X,mo,l1,l2):
    """
    Generate polynomial pool of...
    
    Inputs: (X,mo)
    Retruns: Theta
    
    'X' is a set of datapoints stacked vertically.
    
    'Theta' is returned as a vertically stacked polynomial augmentation of datapoints in 'X'.
    """
    
    #kf*(x[0]-x[4]+x[2]*l1)+cf*(x[1]-x[5]+x[3]*l1)+kt*(-x[4])+ct*(-x[5])
    hcdX=np.array([X[:,0]-X[:,4]+X[:,2]*l1,X[:,1]-X[:,5]+X[:,3]*l1,-X[:,4],-X[:,5]]).T
    
    mo=mo+1
    
    Theta=np.ones((len(X),1))
    
    for crl in range(1,mo):
        
        Theta=np.hstack((Theta,hcdX**crl))
    
    return Theta

def hc_dxr_polypool(X,mo,l1,l2):
    """
    Generate polynomial pool of...
    
    Inputs: (X,mo)
    Retruns: Theta
    
    'X' is a set of datapoints stacked vertically.
    
    'Theta' is returned as a vertically stacked polynomial augmentation of datapoints in 'X'.
    """
    
    #kr*(x[0]-x[6]-x[2]*l2)+cr*(x[1]-x[7]-x[3]*l2)+kt*(-x[6])+ct*(-x[7])
    hcdX=np.array([X[:,0]-X[:,6]-X[:,2]*l2,X[:,1]-X[:,7]-X[:,3]*l2,-X[:,6],-X[:,7]]).T
    
    mo=mo+1
    
    Theta=np.ones((len(X),1))
    
    for crl in range(1,mo):
        
        Theta=np.hstack((Theta,hcdX**crl))
    
    return Theta

##########################################################################################################################################

def SINDy(Theta,dXdt,lam,reg=0,normalise='auto'):
    
    """
    Basic SINDy algorithm solving dXdt=Theta@Xi for Xi using STLSQ. This function uses numpy.linalg.lstsq(rcond=-1), solving for the normal equations given the data. 'Theta' can be normalised using a weighted diagonal matrix, thus normalising the correlation matrix to reduce numerical precision problems.
    
    Input: (Theta,dXdt,lam,reg=0,normalise=False,ssings=False)
    Output: Xi
    
    'Theta' and 'dXdt' are dtapoints stacked vertically.
    'lam' is the hard therholding value used by STLSQ.
    'reg' defines how strongly the correlation matrix diagonals are stabalised (L2 regularisation).
    'normalise' is the vector used to normalise 'Theta'; is converted to a diagonal matrix prior to use.
    """
    
    n=dXdt.shape[-1]
    n_col=Theta.shape[1]
    
    if normalise==False:
        W=np.identity(n_col)
    elif normalise=='auto':
        W=np.diag(1/np.max(np.abs(Theta),axis=0))
    else:
        W=np.diag(normalise)
        W=W.astype('float64')
    
    Thetaw=Theta@W
    #print(np.log10(np.linalg.cond(Thetaw)).round())
    Q,R=np.linalg.qr(Thetaw)
    RTR=R.T@R
    RTQTb=R.T@Q.T@dXdt
    
    RTR=RTR+reg*np.identity(n_col)
    
    Xi=W@np.linalg.lstsq(RTR,RTQTb,rcond=-1)[0]
    Xip=np.copy(Xi)
    
    if lam!=0:
        flag=False
        while not flag:
            smallinds=abs(Xi)<=lam
            Xi[smallinds]=0
            for ind in range(n):
                biginds=(smallinds==False)[:,ind]
                Xi[biginds,ind]=np.linalg.lstsq(RTR[:,biginds][biginds,:]
                                                ,RTQTb[biginds,ind],rcond=-1)[0]
            Xi=W@Xi
            flag=np.array_equal(Xip,Xi)
            Xip=np.copy(Xi)
    
    return Xi

##########################################################################################################################################

def SINDy_prec(Theta,dXdt,normalise='auto'):
    
    """
    Basic SINDy algorithm solving dXdt=Theta@Xi for Xi using STLSQ. This function uses numpy.linalg.lstsq(rcond=-1), solving for the normal equations given the data. 'Theta' can be normalised using a weighted diagonal matrix, thus normalising the correlation matrix to reduce numerical precision problems.
    
    Input: (Theta,dXdt,lam,reg=0,normalise=False,ssings=False)
    Output: Xi
    
    'Theta' and 'dXdt' are dtapoints stacked vertically.
    'lam' is the hard therholding value used by STLSQ.
    'reg' defines how strongly the correlation matrix diagonals are stabalised (L2 regularisation).
    'normalise' is the vector used to normalise 'Theta'; is converted to a diagonal matrix prior to use.
    """
    
    n_col=Theta.shape[1]
    
    if normalise==False:
        W=np.identity(n_col)
    elif normalise=='auto':
        W=np.diag(1/np.max(np.abs(Theta),axis=0))
    else:
        W=np.diag(normalise)
        W=W.astype('float64')
    
    Thetaw=Theta@W
    Q,R=np.linalg.qr(Thetaw)
    RTR=R.T@R
    RTQTb=R.T@Q.T@dXdt
    
    return W,RTR,RTQTb

##########################################################################################################################################

def post_SINDy(W,RTR,RTQTb,lam,reg=0):
    
    """
    Basic SINDy algorithm solving dXdt=Theta@Xi for Xi using STLSQ. This function uses numpy.linalg.lstsq(rcond=-1), solving for the normal equations given the data. 'Theta' can be normalised using a weighted diagonal matrix, thus normalising the correlation matrix to reduce numerical precision problems.
    
    Input: (Theta,dXdt,lam,reg=0,normalise=False,ssings=False)
    Output: Xi
    
    'Theta' and 'dXdt' are dtapoints stacked vertically.
    'lam' is the hard therholding value used by STLSQ.
    'reg' defines how strongly the correlation matrix diagonals are stabalised (L2 regularisation).
    'normalise' is the vector used to normalise 'Theta'; is converted to a diagonal matrix prior to use.
    """
    
    n=RTQTb.shape[-1]
    n_col=RTR.shape[1]
    
    RTR=RTR+reg*np.identity(n_col)
    
    Xi=W@np.linalg.lstsq(RTR,RTQTb,rcond=-1)[0]
    Xip=np.copy(Xi)
    
    if lam!=0:
        flag=False
        while not flag:
            smallinds=abs(Xi)<=lam
            Xi[smallinds]=0
            for ind in range(n):
                biginds=(smallinds==False)[:,ind]
                Xi[biginds,ind]=np.linalg.lstsq(RTR[:,biginds][biginds,:]
                                                ,RTQTb[biginds,ind],rcond=-1)[0]
            Xi=W@Xi
            flag=np.array_equal(Xip,Xi)
            Xip=np.copy(Xi)
    
    return Xi

##########################################################################################################################################

def wSINDy(Theta,dXdt,lam,reg=0,normalise=False):
    
    """
    SINDy STLSQ with weighted least squares, with inverse gradient magnitude weights per data point.
    
    Input: (Theta,dXdt,lam,reg=0,normalise=False,ssings=False)
    Output: Xi
    
    'Theta' and 'dXdt' are dtapoints stacked vertically.
    'lam' is the hard therholding value used by STLSQ.
    'reg' defines how strongly the correlation matrix diagonals are stabalised (L2 regularisation).
    'normalise' is the vector used to normalise 'Theta'; is converted to a diagonal matrix prior to use.
    """
    
    wXi=np.zeros((dXdt.shape[1],Theta.shape[1]))
    
    for i in range(dXdt.shape[1]):
        wXi[i]=SINDy(Theta/np.abs(dXdt[:,[i]]),dXdt[:,[i]]/np.abs(dXdt[:,[i]]),lam,reg,normalise)[:,0]
    
    return wXi.T

##########################################################################################################################################

def ESINDy(Theta,dXdt,lam,nstr,nlap,rp0,pt0,normalise='auto'):
    
    """
    
    """
    
    randomise = np.arange(len(dXdt))
    
    penalty=rp0

    mcs=[]
    inc=[]
    wts=None
    for i in range(nstr):
        np.random.shuffle(randomise)
        mc=SINDy(Theta[randomise][:int(nlap*len(dXdt))],dXdt[randomise][:int(nlap*len(dXdt))]
                 ,lam,penalty,normalise)
        mcs.append(mc)
        inc.append(np.nan_to_num(np.abs(mc/mc)))
        
    pt=pt0
    MC=WAS(mcs,wts)[0]*((np.sum(inc,axis=0)/nstr)>pt)*(np.abs(WAS(mcs,wts)[0])>lam)
    
    mc=re_SINDy(Theta,dXdt,copy.deepcopy(MC),lam,penalty,normalise)
    MC=mc
    
    return MC

##########################################################################################################################################

def dispmymod(Xi,statenames,termnames):
    
    """
    Display SINDy model.
    
    Input: (Xi,statenames,termnames)
    Output: 
    
    'Xi' model coefficients.
    'statenames' refers to the list of names on the left hand side terms of the ODE.
    'termnames' is a list of names for all the model parameters present (length should equate to number of elements in Xi).
    """
    
    from IPython.display import display, Latex
    model=np.char.add(Xi.round(2).astype(str),np.array(termnames).reshape(Xi.shape[1],-1).T)
    for i in range(Xi.shape[1]):
        l='$'+statenames[i]+'$'+' = '
        for j in range(Xi.shape[0]):
            if Xi[j,i]!=0 and l==statenames[i]+' = ':
                l=l+model[j,i]
            elif Xi[j,i]!=0 and l!=statenames[i]+' = ':
                l=l+' + '+model[j,i]
        display(Latex(l))

##########################################################################################################################################

def predict(Theta,Xi):
    """
    Input: (Theta,Xi)
    Output: Theta@Xi
    """
    return Theta@Xi

##########################################################################################################################################

def modelg(t,x,Xi,library_type="polypool",mo=3):
    
    """
    This function returns the gradient of the SINDy model at (t,x), given all other SINDy model info. Can be called by odeint. If multiple (t.x) values are provided, the function returns a list of gradients in the same shape as x.
    
    Input: (t,x,Xi,library_type="polypool",mo=3)
    Output: globals()[library_type](x,mo)@Xi
    """
    
    if len(x.shape)==1:
        return (globals()[library_type](x.reshape(1,-1),mo)@Xi)[0]
    else:
        return globals()[library_type](x,mo)@Xi

##########################################################################################################################################

def re_SINDy(Theta,dXdt,Xii,lam=0,reg=0,normalise='auto'):
    
    """
    Continue the SINDy STLSQ procedure given current sparsity. This cannot reduce sparsity.
    
    Input: re_SINDy(Theta,dXdt,Xii,lam=0,reg=0,normalise=False)
    Output: Xi
    
    'Xii' is the initial Xi solution which may already be sparse.
    Other arguments are identical to 'SINDy()'.
    """
    
    n=dXdt.shape[-1]
    n_col=Theta.shape[1]
    
    if normalise==False:
        W=np.identity(n_col)
    elif normalise=='auto':
        W=np.diag(1/np.max(np.abs(Theta),axis=0))
    else:
        W=np.diag(normalise)
        W=W.astype('float64')
    
    Thetaw=Theta@W
    #ThetawTThetaw=Thetaw.T.dot(Thetaw)
    #ThetawTdXdt=Thetaw.T.dot(dXdt)
    Q,R=np.linalg.qr(Thetaw)
    RTR=R.T@R
    RTQTb=R.T@Q.T@dXdt
    
    RTR=RTR+reg*np.identity(n_col)
    
    Xi=np.copy(Xii)
    Xip=np.copy(Xi)
    
    smallinds=abs(Xi)==0
    smallinds.tolist()
    
    for ind in range(n):
        biginds=(smallinds==False)[:,ind]
        Xi[biginds,ind]=np.linalg.lstsq(RTR[:,biginds][biginds,:]
                                        ,RTQTb[biginds,ind],rcond=-1)[0]
    Xi=W@Xi
    Xip=np.copy(Xi)
    
    if lam!=0:
        flag=False
        while not flag:
            smallinds=abs(Xi)<=lam
            Xi[smallinds]=0
            for ind in range(n):
                biginds=(smallinds==False)[:,ind]
                Xi[biginds,ind]=np.linalg.lstsq(RTR[:,biginds][biginds,:]
                                                ,RTQTb[biginds,ind],rcond=-1)[0]
            Xi=W@Xi
            flag=np.array_equal(Xip,Xi)
            Xip=np.copy(Xi)
    
    return Xi

##########################################################################################################################################

def post_re_SINDy(W,RTR,RTQTb,Xii,lam=0,reg=0,normalise='auto'):
    
    """
    Continue the SINDy STLSQ procedure given current sparsity. This cannot reduce sparsity.
    
    Input: re_SINDy(Theta,dXdt,Xii,lam=0,reg=0,normalise=False)
    Output: Xi
    
    'Xii' is the initial Xi solution which may already be sparse.
    Other arguments are identical to 'SINDy()'.
    """
    
    n=RTQTb.shape[-1]
    n_col=RTR.shape[1]
    
    RTR=RTR+reg*np.identity(n_col)
    
    Xi=np.copy(Xii)
    Xip=np.copy(Xi)
    
    smallinds=abs(Xi)==0
    smallinds.tolist()
    
    for ind in range(n):
        biginds=(smallinds==False)[:,ind]
        Xi[biginds,ind]=np.linalg.lstsq(RTR[:,biginds][biginds,:]
                                        ,RTQTb[biginds,ind],rcond=-1)[0]
    Xi=W@Xi
    Xip=np.copy(Xi)
    
    if lam!=0:
        flag=False
        while not flag:
            smallinds=abs(Xi)<=lam
            Xi[smallinds]=0
            for ind in range(n):
                biginds=(smallinds==False)[:,ind]
                Xi[biginds,ind]=np.linalg.lstsq(RTR[:,biginds][biginds,:]
                                                ,RTQTb[biginds,ind],rcond=-1)[0]
            Xi=W@Xi
            flag=np.array_equal(Xip,Xi)
            Xip=np.copy(Xi)
    
    return Xi

##########################################################################################################################################

def opt_SINDy_i(X,ts,smodel,omethod='Nelder-Mead',maxiter=0,eo=2):
    
    """
    Improves a pre-fit 'pysindy.SINDy' model coefficients by minimising error in the integral space (i.e. actual measurement data) using 'scipy.optimize.minimize' as compared to the standard least squares errors of the derivatives. This method does not change the model sparsity. This may significantly improve the model if the data is very noisy or has many weakily dynamic regions in the data.
    
    Inputs: (X,ts,smodel,omethod='Nelder-Mead',maxiter=0,eo=2)
    Retruns: Xi
    
    'X' is a set of datapoints stacked vertically.
    'ts' is a 1D array of timestamps for X.
    'smodel' is the given 'pysindy.SINDy' model to optimise.
    'omethod' is the optimisation method to be used by 'scipy.optimize.minimize'.
    If 'maxiter' is set to the default value of 0, it is set as int(7500/len(X)) to limit computational time.
    'eo' is the error order to minimise.
    
    Note that 'Xi' returned in this function is formatted as a transpose of the 'pysindy.SINDy' format, but equavilent to the typical SINDy definition. Take note when overwritting model coefficients externally.
    """
    
    XT=np.copy(X.T)
    smat=np.copy(smodel.coefficients())
    
    dt=ts[1]-ts[0]
    
    if maxiter==0:
        maxiter=int(7500/XT.shape[1])
    
    def obj(xl,xs,xp,data,times):
        
        xm=np.zeros(xs)
        for i in range(len(xp)):
            xm[xp[i][0],xp[i][1]]=xl[i]
        
        smodel.coefficients()[:]=xm
        
        x0=data[:,0]
        sol=smodel.simulate(x0,times)
        sol=sol.T
            
        return np.sum(np.abs(sol-data)**eo)

    xl0=smat[np.nonzero(smat)]
    mci=np.transpose(np.nonzero(smat))
    
    mco=sc.optimize.minimize(obj,xl0,args=(smat.shape,mci,XT,ts)\
                                 ,method=omethod,options={'maxiter': maxiter})
    
    
    MCO=np.copy(smat)
    for i in range(len(mci)):
        MCO[mci[i][0],mci[i][1]]=mco['x'][i]
    
    return np.copy(MCO.T)

##########################################################################################################################################

def opt_SINDy(X,dXdt,smodel,omethod='Nelder-Mead',maxiter=0,eo=2):
    
    """
    Improves a pre-fit 'pysindy.SINDy' model coefficients by method of iterative numerical optimisation, using the current model as the initial estimate. This does not affect model sparsity. This may improve the model if the data is very noisy.
    
    Inputs: (X,ts,smodel,omethod='Nelder-Mead',maxiter=0,eo=2)
    Retruns: Xi
    
    'X' and 'dXdt' are a set of datapoints stacked vertically.
    'smodel' is the given 'pysindy.SINDy' model to optimise.
    'omethod' is the optimisation method to be used by 'scipy.optimize.minimize'.
    If 'maxiter' is set to the default value of 0, it is set as int(7500/len(X)) to limit computational time.
    'eo' is the error order to minimise.
    
    Note that 'Xi' returned in this function is formatted as a transpose of the 'pysindy.SINDy' format, but equavilent to the typical SINDy definition. Take note when overwritting model coefficients externally.
    """
    
    XT=np.copy(X.T)
    smat=np.copy(smodel.coefficients())
    
    if maxiter==0:
        maxiter=int(7500/XT.shape[1])
    
    def obj(xl,xs,xp,data,datag):
        
        xm=np.zeros(xs)
        for i in range(len(xp)):
            xm[xp[i][0],xp[i][1]]=xl[i]
        
        smodel.coefficients()[:]=xm
        solg=smodel.predict(data.T)
            
        return np.sum(np.abs(solg-datag)**eo)

    xl0=smat[np.nonzero(smat)]
    mci=np.transpose(np.nonzero(smat))
    
    mco=sc.optimize.minimize(obj,xl0,args=(smat.shape,mci,XT,dXdt)\
                                 ,method=omethod,options={'maxiter': maxiter})
    
    MCO=np.copy(smat)
    for i in range(len(mci)):
        MCO[mci[i][0],mci[i][1]]=mco['x'][i]
    
    return np.copy(MCO.T)

##########################################################################################################################################

def SINDyL1(X,Theta,dXdt,lam,smodel,reg=0):
    
    """
    Basic SINDy algorithm with L1 regression (STLAE; numerical optimisation).
    
    Input: (X,Theta,dXdt,lam,smodel,reg=0)
    Output: Xi
    
    All arguments are identical in use as in 'SINDy', however this function requires the non-augmented datapoints 'X' as well as a pre-fit 'pysindy.SINDy' model. The pre-fit model is completely overwritten. The 'reg' is only used when generating an initial estimate using the standard least squares L2 regularised solution.
    """
    
    n=dXdt.shape[-1]
    n_col=Theta.shape[1]
    
    Xi=np.linalg.lstsq(Theta.T.dot(Theta)+reg*np.identity(n_col),Theta.T.dot(dXdt),rcond=-1)[0]
    Xip=np.copy(Xi)
        
    flag=False
    while not flag:
        smallinds=abs(Xi)<lam
        Xi[smallinds]=0
        smodel.coefficients()[:,:]=Xi.T
        Xi=opt_SINDy(X,dXdt,smodel,maxiter=None,eo=1)
        flag=np.allclose(Xip,Xi)
        Xip=np.copy(Xi)
    
    return Xi

##########################################################################################################################################

def SINDyL2(X,Theta,dXdt,lam,smodel,reg=0):
    
    """
    Basic SINDy algorithm with L2 regression (STLSQ; numerical optimisation).
    
    Input: (X,Theta,dXdt,lam,smodel,reg=0)
    Output: Xi
    
    All arguments are identical in use as in 'SINDy', however this function requires the non-augmented datapoints 'X' as well as a pre-fit 'pysindy.SINDy' model. The pre-fit model is completely overwritten. The 'reg' is only used when generating an initial estimate using the standard least squares L2 regularised solution.
    """
    
    n=dXdt.shape[-1]
    n_col=Theta.shape[1]
    
    Xi=np.linalg.lstsq(Theta.T.dot(Theta)+reg*np.identity(n_col),Theta.T.dot(dXdt),rcond=-1)[0]
    Xip=np.copy(Xi)
        
    flag=False
    while not flag:
        smallinds=abs(Xi)<lam
        Xi[smallinds]=0
        smodel.coefficients()[:,:]=Xi.T
        Xi=opt_SINDy(X,dXdt,smodel,maxiter=None,eo=2)
        flag=np.allclose(Xip,Xi)
        Xip=np.copy(Xi)
    
    return Xi

##########################################################################################################################################

def WAS(values, weights):
    """
    Return the weighted average and standard deviation.

    Input: (values, weights)
    Output: (average, np.sqrt(variance))
    """
    average = np.average(values,axis=0,weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2,axis=0,weights=weights)
    return (average, np.sqrt(variance))

##########################################################################################################################################

def SINDyCV0(cmodel,Theta,dXdt,lam,nstr,rp0,rp,pt0,ptstep,ptf=1,normalise=False,ssings=False):
    
    """
    STLSQCV SINDy
    
    Input: (cmodel,Theta,dXdt,lam,nstr,rp0,rp,pt0,ptstep,ptf=1,normalise=False,ssings=False)
    Output: Xi
    """
    
    randomise = np.arange(len(dXdt))
    np.random.shuffle(randomise)
    Theta=Theta[randomise];dXdt=dXdt[randomise]
    
    penalty=rp0

    mcs=[]
    wts=None
    for i in tnrange(nstr):
        mc=SINDy(Theta[int(i*len(dXdt)/nstr):int((i+1)*len(dXdt)/nstr)],dXdt[int(i*len(dXdt)/nstr):int((i+1)*len(dXdt)/nstr)]
                 ,lam,penalty,normalise)
        mc=mc.T
        mcs.append(mc)
        
    pt=pt0
    cmodel.coefficients()[:,:]=WAS(mcs,wts)[0]*(np.abs(WAS(mcs,wts)[0]/WAS(mcs,wts)[1])>pt)*(np.abs(WAS(mcs,wts)[0])>lam)
    currentsigns=np.sign(cmodel.coefficients()[:,:])
    cmodel.print()
    
    pts=np.arange(pt0,ptf+0.0001,ptstep)
    rps=np.geomspace(rp0,rp,len(pts))
    
    for j in tnrange(len(pts)):
        modelchange=True
        while modelchange:
            penalty=rps[j]

            mcs=[]
            for i in np.arange(nstr):
                mc=re_SINDy(Theta[int(i*len(dXdt)/nstr):int((i+1)*len(dXdt)/nstr)],dXdt[int(i*len(dXdt)/nstr):int((i+1)*len(dXdt)/nstr)]
                            ,copy.deepcopy(cmodel.coefficients()).T,lam,penalty,normalise)
                mc=mc.T
                mcs.append(mc)

            pt=pts[j]
            cterms=cmodel.coefficients()[:,:]
            nterms=WAS(mcs,wts)[0]*(np.abs(WAS(mcs,wts)[0]/WAS(mcs,wts)[1])>pt)*(np.abs(WAS(mcs,wts)[0])>lam)
            #\*(np.sign(WAS(mcs,wts)[0])==currentsigns)
            modelchange=np.any(np.nan_to_num(cterms/cterms)!=np.nan_to_num(nterms/nterms))
            #print(modelchange)
            cmodel.coefficients()[:,:]=WAS(mcs,wts)[0]*(np.abs(WAS(mcs,wts)[0]/WAS(mcs,wts)[1])>pt)*(np.abs(WAS(mcs,wts)[0])>lam)
            #\*(np.sign(WAS(mcs,wts)[0])==currentsigns)
            currentsigns=np.sign(cmodel.coefficients()[:,:])
            cmodel.print()
            print()
    
    mc=re_SINDy(Theta,dXdt,copy.deepcopy(cmodel.coefficients()).T,lam,penalty,normalise)
    mc=mc.T
    cmodel.coefficients()[:,:]=mc#*(np.sign(mc)==currentsigns)
    
    return cmodel.coefficients().T

##########################################################################################################################################

def SINDyCV(Theta,dXdt,lam,nstr,rp0,rp,pt0,ptstep,ptf=1,normalise='auto'):
    
    """
    STLSQCV SINDy
    
    Input: (cmodel,Theta,dXdt,lam,nstr,rp0,rp,pt0,ptstep,ptf=1,normalise='auto')
    Output: Xi
    """
    
    randomise = np.arange(len(dXdt))
    np.random.shuffle(randomise)
    Theta=Theta[randomise];dXdt=dXdt[randomise]
    
    penalty=rp0

    mcs=[]
    wts=None
    for i in range(nstr):
        mc=SINDy(Theta[int(i*len(dXdt)/nstr):int((i+1)*len(dXdt)/nstr)],dXdt[int(i*len(dXdt)/nstr):int((i+1)*len(dXdt)/nstr)]
                 ,lam,penalty,normalise)
        mcs.append(mc)
        #print(mc[1,2])
        
    pt=pt0
    MC=WAS(mcs,wts)[0]*(np.abs(WAS(mcs,wts)[0]/WAS(mcs,wts)[1])>pt)*(np.abs(WAS(mcs,wts)[0])>lam)
    currentsigns=np.sign(MC)
    
    pts=np.arange(pt0,ptf+0.0001,ptstep)
    rps=np.geomspace(rp0,rp,len(pts))
    
    for j in range(len(pts)):
        modelchange=True
        while modelchange:
            penalty=rps[j]

            mcs=[]
            for i in np.arange(nstr):
                mc=re_SINDy(Theta[int(i*len(dXdt)/nstr):int((i+1)*len(dXdt)/nstr)],dXdt[int(i*len(dXdt)/nstr):int((i+1)*len(dXdt)/nstr)]
                            ,copy.deepcopy(MC),lam,penalty,normalise)
                mcs.append(mc)

            pt=pts[j]
            cterms=MC
            nterms=WAS(mcs,wts)[0]*(np.abs(WAS(mcs,wts)[0]/WAS(mcs,wts)[1])>pt)*(np.abs(WAS(mcs,wts)[0])>lam)
            #\*(np.sign(WAS(mcs,wts)[0])==currentsigns)
            modelchange=np.any(np.nan_to_num(cterms/cterms)!=np.nan_to_num(nterms/nterms))
            #print(modelchange)
            MC=WAS(mcs,wts)[0]*(np.abs(WAS(mcs,wts)[0]/WAS(mcs,wts)[1])>pt)*(np.abs(WAS(mcs,wts)[0])>lam)
            #\*(np.sign(WAS(mcs,wts)[0])==currentsigns)
            currentsigns=np.sign(MC)
    
    mc=re_SINDy(Theta,dXdt,copy.deepcopy(MC),lam,penalty,normalise)
    MC=mc#*(np.sign(mc)==currentsigns)
    
    return MC.T

##########################################################################################################################################

def SINDyCV_t(Theta,dXdt,lam,rp0,rp,pt0,nsteps,ptf=1,normalise='auto',reticv=False):
    
    """
    STLSQCV SINDy
    
    Input: (cmodel,Theta,dXdt,lam,nstr,rp0,rp,pt0,ptstep,ptf=1,normalise=False,ssings=False)
    Output: Xi
    """
    
    penalty=rp0
    
    #mc=SINDy(Theta,dXdt,0,penalty,normalise)
    lw,lrtr,lrtqtb=SINDy_prec(Theta,dXdt,normalise)
    mc=post_SINDy(lw,lrtr,lrtqtb,0,reg=penalty)
    #print(mc)
    beta=1/((np.sum(np.square(Theta@mc-dXdt),axis=0))/dXdt.shape[0])
    #print(beta)
    
    ThetaTTheta=Theta.T.dot(Theta)
    ThetaTThetan=ThetaTTheta/(dXdt.shape[0]**0)
    #ThetaTTheta=np.cov(Theta.T)
    #print(np.diag(ThetaTTheta))
    #SN=1/(np.diag(ThetaTTheta).reshape(-1,1)@beta.reshape(1,-1)+penalty)
    cstds=[]
    for betai in beta:
        SN=np.linalg.inv((penalty*betai)*np.eye(len(ThetaTThetan))+betai*ThetaTThetan)
        #print(SN)
        cstds.append(np.sqrt(np.diag(SN)))
    cstds=np.array(cstds).T
    #print(cstds)
    #SN=np.sqrt(SN)
    #print(penalty*np.eye(len(ThetaTTheta))+beta*ThetaTTheta)
    #print(cstds.shape)
    #print(mc/cstds)
    #print(':',np.sqrt(np.diag(SN).reshape(-1,1)))
    #print(beta*(np.linalg.inv(beta*ThetaTTheta))@(Theta.T)@dXdt)
    
    pt=pt0
    #print(np.abs(mc/cstds))
    MC=mc*(np.abs(mc/cstds/(dXdt.shape[0]**0.5))>pt)*(np.abs(mc)>lam)
    #currentsigns=np.sign(MC)
    pts=np.linspace(1e-16+pt0,1e-16+ptf,nsteps)
    rps=np.linspace(rp0,rp,nsteps)
    
    for j in range(len(pts)):
        modelchange=True
        while modelchange:
            penalty=rps[j]

            mc=post_re_SINDy(lw,lrtr,lrtqtb,copy.deepcopy(MC),lam,penalty)
            beta=1/((np.sum(np.square(Theta@mc-dXdt),axis=0))/dXdt.shape[0])
            cstds=[]
            for ibeta in range(len(beta)):
                #print((np.abs(mc)!=0).T[ibeta])
                #ThetaTTheta=(Theta).T.dot(Theta)
                ThetaTThetan=(((ThetaTTheta*((np.abs(mc)!=0).T[ibeta])).T*((np.abs(mc)!=0).T[ibeta])).T)/(dXdt.shape[0]**0)
                SN=np.linalg.inv((penalty*beta[ibeta])*np.eye(len(ThetaTThetan))+beta[ibeta]*ThetaTThetan)
                cstds.append(np.sqrt(np.diag(SN)))
            cstds=np.array(cstds).T

            pt=pts[j]
            #print(pt,np.abs(mc/cstds/(dXdt.shape[0]**0.5)))
            cterms=MC
            nterms=mc*(np.abs(mc/cstds/(dXdt.shape[0]**0.5))>pt)*(np.abs(mc)>lam)
            #\*(np.sign(mc)==currentsigns)
            modelchange=np.any(np.nan_to_num(cterms/cterms)!=np.nan_to_num(nterms/nterms))
            #print(modelchange)
            MC=mc*(np.abs(mc/cstds/(dXdt.shape[0]**0.5))>pt)*(np.abs(mc)>lam)
            #\*(np.sign(mc)==currentsigns)
            #currentsigns=np.sign(MC)
    
    mc=post_re_SINDy(lw,lrtr,lrtqtb,copy.deepcopy(MC),lam,penalty)
    beta=1/((np.sum(np.square(Theta@mc-dXdt),axis=0))/dXdt.shape[0])
    cstds=[]
    for ibeta in range(len(beta)):
        #print((np.abs(mc)!=0).T[ibeta])
        #ThetaTTheta=(Theta).T.dot(Theta)
        ThetaTThetan=(((ThetaTTheta*((np.abs(mc)!=0).T[ibeta])).T*((np.abs(mc)!=0).T[ibeta])).T)/(dXdt.shape[0]**0)
        SN=np.linalg.inv((penalty*beta[ibeta])*np.eye(len(ThetaTThetan))+beta[ibeta]*ThetaTThetan)
        cstds.append(np.sqrt(np.diag(SN)))
    cstds=np.array(cstds).T
    MC=mc#*(np.sign(mc)==currentsigns)
    #print(mc/cstds)
    
    if reticv==True:
        return MC,(mc/cstds/(dXdt.shape[0]**0.5))*(np.abs(mc)>lam)
    else:
        return MC
##########################################################################################################################################

def nanminidx_2d(x):
    #x=np.abs(x)*x/x
    k = np.nanargmin(x)
    ncol = x.shape[1]
    return k//ncol, k%ncol

##########################################################################################################################################

def SINDyCV_t_1(Theta,dXdt,lam,rp0,rp,pt0,nsteps,ptf=1,normalise='auto',reticv=False):
    
    """
    STLSQCV SINDy
    
    Input: (cmodel,Theta,dXdt,lam,nstr,rp0,rp,pt0,ptstep,ptf=1,normalise=False,ssings=False)
    Output: Xi
    """
    
    penalty=rp0
    
    mc=SINDy(Theta,dXdt,0,penalty,normalise)
    beta=1/((np.sum(np.square(Theta@mc-dXdt),axis=0))/dXdt.shape[0])
    
    ThetaTTheta=Theta.T.dot(Theta)
    ThetaTThetan=ThetaTTheta/(dXdt.shape[0]**0)
    cstds=[]
    for betai in beta:
        SN=np.linalg.inv((penalty*betai)*np.eye(len(ThetaTThetan))+betai*ThetaTThetan)
        cstds.append(np.sqrt(np.diag(SN)))
    cstds=np.array(cstds).T
    
    pt=pt0
    pp=mc/cstds/(dXdt.shape[0]**0.5)
    MC=mc
    pts=np.linspace(1e-16+pt0,1e-16+ptf,nsteps)
    rps=np.linspace(rp0,rp,nsteps)
    
    for j in range(len(pts)):
        penalty=rps[j]
        pt=pts[j]
        
        mc=re_SINDy(Theta,dXdt,copy.deepcopy(MC),lam,penalty,normalise)
        beta=1/((np.sum(np.square(Theta@mc-dXdt),axis=0))/dXdt.shape[0])
        cstds=[]
        for ibeta in range(len(beta)):
            ThetaTThetan=(((ThetaTTheta*((np.abs(mc)!=0).T[ibeta])).T*((np.abs(mc)!=0).T[ibeta])).T)/(dXdt.shape[0]**0)
            SN=np.linalg.inv((penalty*beta[ibeta])*np.eye(len(ThetaTThetan))+beta[ibeta]*ThetaTThetan)
            cstds.append(np.sqrt(np.diag(SN)))
        cstds=np.array(cstds).T
        #print(mc)
        #print(cstds)
        pp=mc/cstds/(dXdt.shape[0]**0.5)
        #print(pp)
        #print()
        MC=mc
        
        while np.nanmin((pp/pp)*np.abs(pp))<pt:
            
            mc[nanminidx_2d((pp/pp)*np.abs(pp))]=0
            MC=mc
            
            mc=re_SINDy(Theta,dXdt,copy.deepcopy(MC),lam,penalty,normalise)
            beta=1/((np.sum(np.square(Theta@mc-dXdt),axis=0))/dXdt.shape[0])
            cstds=[]
            for ibeta in range(len(beta)):
                ThetaTThetan=(((ThetaTTheta*((np.abs(mc)!=0).T[ibeta])).T*((np.abs(mc)!=0).T[ibeta])).T)/(dXdt.shape[0]**0)
                SN=np.linalg.inv((penalty*beta[ibeta])*np.eye(len(ThetaTThetan))+beta[ibeta]*ThetaTThetan)
                cstds.append(np.sqrt(np.diag(SN)))
            cstds=np.array(cstds).T
            #print(mc)
            #print(cstds)
            pp=mc/cstds/(dXdt.shape[0]**0.5)
            #print(pp)
            #print()
            MC=mc
            #print(MC)
            #print(mc);print(pt,penalty);print(pp);print()
            #print(MC);print()
            
    #print(MC);print()
    mc=re_SINDy(Theta,dXdt,copy.deepcopy(MC),lam,penalty,normalise)
    beta=1/((np.sum(np.square(Theta@mc-dXdt),axis=0))/dXdt.shape[0])
    cstds=[]
    for ibeta in range(len(beta)):
        ThetaTThetan=(((ThetaTTheta*((np.abs(mc)!=0).T[ibeta])).T*((np.abs(mc)!=0).T[ibeta])).T)/(dXdt.shape[0]**0)
        SN=np.linalg.inv((penalty*beta[ibeta])*np.eye(len(ThetaTThetan))+beta[ibeta]*ThetaTThetan)
        cstds.append(np.sqrt(np.diag(SN)))
    cstds=np.array(cstds).T
    #print(mc)
    #print(cstds)
    pp=mc/cstds/(dXdt.shape[0]**0.5)
    #print(pp)
    MC=mc
    
    if reticv==True:
        return MC,pp
    else:
        return MC

##########################################################################################################################################

def SINDyCV_lap(Theta,dXdt,lam,nstr,nlap,rp0,rp,pt0,nsteps,ptf=1,normalise='auto'):
    
    """
    STLSQCV SINDy
    
    Input: (cmodel,Theta,dXdt,lam,nstr,rp0,rp,pt0,ptstep,ptf=1,normalise=False,ssings=False)
    Output: Xi
    """
    
    randomise = np.arange(len(dXdt))
    #np.random.shuffle(randomise)
    #Theta=Theta[randomise];dXdt=dXdt[randomise]
    
    penalty=rp0

    mcs=[]
    wts=None
    for i in range(nstr):
        np.random.shuffle(randomise)
        mc=SINDy(Theta[randomise][:int(nlap*len(dXdt))],dXdt[randomise][:int(nlap*len(dXdt))]
                 ,lam,penalty,normalise)
        mcs.append(mc)
        
    pt=pt0
    #print(WAS(mcs,wts)[0]/WAS(mcs,wts)[1]/np.sqrt(nlap*len(dXdt)))
    MC=WAS(mcs,wts)[0]*(np.abs(WAS(mcs,wts)[0]/WAS(mcs,wts)[1]/np.sqrt(nlap*len(dXdt)))>pt)*(np.abs(WAS(mcs,wts)[0])>lam)
    #currentsigns=np.sign(MC)
    pts=np.linspace(1e-16+pt0,1e-16+ptf,nsteps)
    rps=np.geomspace(rp0,rp,nsteps)
    
    for j in range(len(pts)):
        modelchange=True
        while modelchange:
            penalty=rps[j]

            mcs=[]
            for i in np.arange(nstr):
                np.random.shuffle(randomise)
                mc=re_SINDy(Theta[randomise][:int(nlap*len(dXdt))],dXdt[randomise][:int(nlap*len(dXdt))]
                            ,copy.deepcopy(MC),lam,penalty,normalise)
                mcs.append(mc)

            pt=pts[j]
            cterms=MC
            nterms=WAS(mcs,wts)[0]*(np.abs(WAS(mcs,wts)[0]/WAS(mcs,wts)[1]/np.sqrt(nlap*len(dXdt)))>pt)*(np.abs(WAS(mcs,wts)[0])>lam)
            #\*(np.sign(WAS(mcs,wts)[0])==currentsigns)
            modelchange=np.any(np.nan_to_num(cterms/cterms)!=np.nan_to_num(nterms/nterms))
            #print(modelchange)
            MC=WAS(mcs,wts)[0]*(np.abs(WAS(mcs,wts)[0]/WAS(mcs,wts)[1]/np.sqrt(nlap*len(dXdt)))>pt)*(np.abs(WAS(mcs,wts)[0])>lam)
            #\*(np.sign(WAS(mcs,wts)[0])==currentsigns)
            #currentsigns=np.sign(MC)
    
    mc=re_SINDy(Theta,dXdt,copy.deepcopy(MC),lam,penalty,normalise)
    MC=mc#*(np.sign(mc)==currentsigns)
    #print(WAS(mcs,wts)[0]/WAS(mcs,wts)[1]/np.sqrt(nlap*len(dXdt)))
    
    return MC

##########################################################################################################################################

def SINDyCV_rep(Theta,dXdt,lam,nstr,nlev,rp0,rp,pt0,ptstep,ptf=1,normalise=False):
    
    """
    STLSQCV SINDy which does not divide data but rather generates noisy copies.
    
    Input: (cmodel,Theta,dXdt,lam,nstr,rp0,rp,pt0,ptstep,ptf=1,normalise=False,ssings=False)
    Output: Xi
    """
    
    penalty=rp0

    mcs=[]
    wts=None
    for i in range(nstr):
        mc=SINDy(np.random.normal(Theta,nlev/2),np.random.normal(dXdt,nlev/2)
                 ,lam,penalty,normalise)
        mcs.append(mc)
        
    pt=pt0
    MC=WAS(mcs,wts)[0]*(np.abs(WAS(mcs,wts)[0]/WAS(mcs,wts)[1])>pt)*(np.abs(WAS(mcs,wts)[0])>lam)
    currentsigns=np.sign(MC)
    
    pts=np.arange(pt0,ptf+0.0001,ptstep)
    rps=np.geomspace(rp0,rp,len(pts))
    
    for j in range(len(pts)):
        modelchange=True
        while modelchange:
            penalty=rps[j]

            mcs=[]
            for i in np.arange(nstr):
                mc=re_SINDy(np.random.normal(Theta,nlev/2),np.random.normal(dXdt,nlev/2)
                            ,copy.deepcopy(MC),lam,penalty,normalise)
                mcs.append(mc)

            pt=pts[j]
            cterms=MC
            nterms=WAS(mcs,wts)[0]*(np.abs(WAS(mcs,wts)[0]/WAS(mcs,wts)[1])>pt)*(np.abs(WAS(mcs,wts)[0])>lam)
            #\*(np.sign(WAS(mcs,wts)[0])==currentsigns)
            modelchange=np.any(np.nan_to_num(cterms/cterms)!=np.nan_to_num(nterms/nterms))
            #print(modelchange)
            MC=WAS(mcs,wts)[0]*(np.abs(WAS(mcs,wts)[0]/WAS(mcs,wts)[1])>pt)*(np.abs(WAS(mcs,wts)[0])>lam)
            #\*(np.sign(WAS(mcs,wts)[0])==currentsigns)
            currentsigns=np.sign(MC)
    
    mc=re_SINDy(Theta,dXdt,copy.deepcopy(MC).T,lam,penalty,normalise)
    MC=mc#*(np.sign(mc)==currentsigns)
    
    return MC

##########################################################################################################################################

def STE(X,tstamps,Xi,library_type="polypool",mo=3,nststeps=1):
    
    """
    
    """
    
    possiblesttp=np.array([ele for ele in np.arange(len(tstamps))[::-1][::nststeps][::-1] if ele not in np.arange(nststeps)])
    unklen=len(tstamps)-len(possiblesttp)*nststeps
    
    st=X[0,:].reshape(1,-1)

    for it in possiblesttp:
        x0=X[it-nststeps,:]
        sti=sc.integrate.solve_ivp(modelg,[tstamps[it-nststeps],tstamps[it]],x0,
                                  args=tuple([Xi,library_type,mo]),method='Radau').y[:,-1].reshape(1,-1)
        st=np.vstack((st,sti))
    
    ste=st[1:]-X[possiblesttp]
    
    ste=np.repeat(ste,nststeps,axis=0)
    ste=np.vstack((np.inf*np.ones_like(X[:unklen,:].reshape(unklen,-1)),ste))
    
    return ste

##########################################################################################################################################

def GNSTE(X,tstamps,Xi,library_type="polypool",mo=3,nststeps=1,dist=1,boost=0):
    
    """
    
    """
    
    possiblesttp=np.array([ele for ele in np.arange(len(tstamps))[::-1][::nststeps][::-1] if ele not in np.arange(nststeps)])
    unklen=len(tstamps)-len(possiblesttp)*nststeps
    
    st=X[0,:].reshape(1,-1)
    abspgrad=np.ones((1,X.shape[1]))
    
    for it in possiblesttp:
        x0=X[it-nststeps,:]
        stiy=sc.integrate.solve_ivp(modelg,[tstamps[it-nststeps],tstamps[it]],x0,
                                  args=tuple([Xi,library_type,mo]),method='Radau').y
        sti=stiy[:,-1].reshape(1,-1)
        st=np.vstack((st,sti))
        abspgrad=np.append(abspgrad,[np.abs(modelg(0,stiy.T[:-1],Xi,library_type,mo).sum(axis=0))],axis=0)
        
    ste=st[1:]-X[possiblesttp]
    ste=np.repeat(ste,nststeps,axis=0)
    ste=np.vstack((np.inf*np.ones_like(X[:unklen,:].reshape(unklen,-1)),ste))
    
    abspgrad=abspgrad[1:].T
    
    ptstamps=[]
    peaks=[]
    for ivar in range(X.shape[1]):
        ptstamps.append(tstamps[possiblesttp][sc.signal.find_peaks(abspgrad[ivar],distance=dist)[0]])
        peaks.append(abspgrad[ivar][sc.signal.find_peaks(abspgrad[ivar],distance=dist)[0]])
    
    env=[]
    for iptstamp in range(len(ptstamps)):
        if len(ptstamps[iptstamp])==0:
            env.append(np.ones_like(tstamps[possiblesttp]))
        else:
            env.append(np.interp(tstamps[possiblesttp],ptstamps[iptstamp],peaks[iptstamp]))
    
    env=np.array(env)
    env=env.T
    env=np.repeat(env,nststeps,axis=0)
    env=np.vstack((np.ones_like(X[:unklen,:].reshape(unklen,-1)),env))+boost
    gne=ste/env
    
    return ste,gne

##########################################################################################################################################

def full_STE(X,tstamps,Xi,library_type="polypool",mo=3,nststeps=1):
    
    """
    
    """
    
    st=np.ones_like(X[:nststeps,:].reshape(nststeps,-1))*np.inf

    for it in range(len(tstamps))[nststeps:]:
        x0=X[it-nststeps,:]
        sti=sc.integrate.solve_ivp(modelg,[tstamps[it-nststeps],tstamps[it]],x0,
                                  args=tuple([Xi,library_type,mo]),method='Radau').y[:,-1].reshape(1,-1)
        st=np.vstack((st,sti))
    ste=st-X
    
    return ste

##########################################################################################################################################

def full_GNSTE(X,tstamps,Xi,library_type="polypool",mo=3,nststeps=1,smooth=1,dist=1,boost=0):
    
    """
    
    """
    
    '''print()
    print(X)
    print(tstamps)
    print(Xi)
    print(library_type)
    print(mo)
    print(nststeps)
    print(smooth)
    print(dist)
    print(boost)
    print()'''
    
    st=np.ones_like(X[:nststeps,:].reshape(nststeps,-1))*np.inf
    pgrad=np.ones_like(X[:nststeps,:].reshape(nststeps,-1))
    #abspgrad=np.ones_like(X[:nststeps,:].reshape(nststeps,-1))
    
    for it in np.arange(len(tstamps))[nststeps:]:
        x0=X[it-nststeps,:]
        stiy=sc.integrate.solve_ivp(modelg,[tstamps[it-nststeps],tstamps[it]],x0,
                                  args=tuple([Xi,library_type,mo]),method='Radau').y
        sti=stiy[:,-1].reshape(1,-1)
        st=np.vstack((st,sti))
        pgrad=np.append(pgrad,[modelg(0,stiy.T[:-1],Xi,library_type,mo).sum(axis=0)],axis=0)
        #abspgrad=np.append(abspgrad,[np.abs(modelg(0,stiy.T[:-1],Xi,library_type,mo).sum(axis=0))],axis=0)
        
    ste=st[nststeps:]-X[nststeps:]
    ste=np.vstack((np.inf*np.ones_like(X[:nststeps,:].reshape(nststeps,-1)),ste))
    
    #abspgrad=abspgrad[nststeps:].T
    #print(np.abs(sc.signal.cspline1d(pgrad.T[1,:],1)))
    abspgrad=[]
    for ivar in range(X.shape[1]):
        abspgrad.append(np.abs(sc.signal.cspline1d(pgrad.T[ivar,:],smooth)))
    abspgrad=np.array(abspgrad)[:,nststeps:]
    #print(abspgrad)
    
    ptstamps=[]
    peaks=[]
    for ivar in range(X.shape[1]):
        ptstamps.append(tstamps[nststeps:][sc.signal.find_peaks(abspgrad[ivar],distance=dist)[0]])
        peaks.append(abspgrad[ivar][sc.signal.find_peaks(abspgrad[ivar],distance=dist)[0]])
    
    env=[]
    for iptstamp in range(len(ptstamps)):
        if len(ptstamps[iptstamp])==0:
            env.append(np.ones_like(tstamps[nststeps:]))
        else:
            env.append(np.interp(tstamps[nststeps:],ptstamps[iptstamp],peaks[iptstamp]))
    
    env=np.array(env)
    env=env.T
    env=np.vstack((np.ones_like(X[:nststeps,:].reshape(nststeps,-1)),env))+boost
    gne=ste/env
    
    return ste,gne

##########################################################################################################################################

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w

##########################################################################################################################################

def vfil_percent(earray,filfrac,bufflen,minsamlen):
    
    """
    
    """
    
    filout=np.array([])
    eargsorted=np.argsort(earray)[::-1]
    epos=0

    while len(filout)<filfrac*len(earray):
        apoint=eargsorted[epos]
        filout=np.append(filout,range(apoint-bufflen,apoint+bufflen))
        filout=np.unique(filout)
        epos+=1

    filout=filout.astype(np.int)
    filinds=np.delete(np.arange(len(earray)),filout)

    filindstps=np.append(np.array([filinds[0]]),filinds[np.append(np.diff(filinds),np.array([1]))>1])
    filindstps=np.append(filindstps,filinds[np.append(np.array([1]),np.diff(filinds))>1])
    filindstps=np.append(filindstps,np.array(filinds[-1]))
    filindstps=np.sort(filindstps)

    filinds=np.array([])

    for itps in range(len(filindstps))[::2]:
        if filindstps[itps+1]-filindstps[itps]>minsamlen:
            filinds=np.append(filinds,np.arange(filindstps[itps],filindstps[itps+1]+1))

    filinds=filinds.astype(np.int)
    
    return filinds

##########################################################################################################################################

def vfil_amp(earray,filamp,bufflen,minsamlen):
    
    """
    
    """
    
    filout=np.array([])
    eargsorted=np.argsort(earray)[::-1]
    epos=0
    apoint=eargsorted[epos]

    while earray[apoint]>filamp:
        filout=np.append(filout,range(apoint-bufflen,apoint+bufflen))
        filout=np.unique(filout)
        epos+=1
        apoint=eargsorted[epos]

    filout=filout.astype(np.int)
    filinds=np.delete(np.arange(len(earray)),filout)

    filindstps=np.append(np.array([filinds[0]]),filinds[np.append(np.diff(filinds),np.array([1]))>1])
    filindstps=np.append(filindstps,filinds[np.append(np.array([1]),np.diff(filinds))>1])
    filindstps=np.append(filindstps,np.array(filinds[-1]))
    filindstps=np.sort(filindstps)

    filinds=np.array([])

    for itps in range(len(filindstps))[::2]:
        if filindstps[itps+1]-filindstps[itps]>minsamlen:
            filinds=np.append(filinds,np.arange(filindstps[itps],filindstps[itps+1]+1))

    filinds=filinds.astype(np.int)
    
    return filinds

##########################################################################################################################################

def NA_SINDy_vonly(data,libraryinfo,stcvinfo,sfinfo):
    
    """
    
    """
    
    #unpack arguments
    #data
    X=data[0]
    dXdt=data[1]
    tstamps=data[2]
    #library
    librarytype=libraryinfo[0]
    deg=libraryinfo[1]
    #STCV
    Theta=globals()[librarytype](X,deg)
    lam=stcvinfo[0]
    rp0=stcvinfo[1]
    rp=stcvinfo[2]
    pt0=stcvinfo[3]
    nsteps=stcvinfo[4]
    ptf=stcvinfo[5]
    normalise=stcvinfo[6]
    #STE/SF
    nststeps=sfinfo[0]
    navgwin=sfinfo[1]
    smooth=sfinfo[2]
    dist=sfinfo[3]
    stesig=sfinfo[4]
    gnesig=sfinfo[5]
    buff=sfinfo[6]
    minsamlen=sfinfo[7]
    maxiter=sfinfo[8]
    
    niter=0
    filindsprev=np.array([])
    filinds=np.arange(len(X[:,0]))
    
    while niter<maxiter and not np.all(filindsprev==filinds):
        
        #preliminary model
        mc=SINDyCV_t_1(Theta[filinds],dXdt[filinds],lam,rp0,rp,pt0,nsteps,ptf,normalise)
        mc[:,0]=np.zeros_like(mc[:,0])
        mc[2,0]=np.median(dXdt[:,0]/X[:,1])

        #STE
        ste,gne=full_GNSTE(X,tstamps,mc,nststeps=nststeps,smooth=smooth,dist=dist)
        maste=moving_average(ste[:,1],navgwin)
        magne=moving_average(gne[:,1],navgwin)
        ster,gner=full_GNSTE(X[::-1],tstamps,-mc,nststeps=nststeps,smooth=smooth,dist=dist)
        ster=ster[::-1]
        gner=gner[::-1]
        master=moving_average(ster[:,1],navgwin)
        magner=moving_average(gner[:,1],navgwin)

        #filtering
        filindsprev=np.copy(filinds)
        filindsste=vfil_amp(np.abs(maste),1.4826*sc.stats.median_abs_deviation(maste[np.isfinite(maste)])*stesig,buff,minsamlen)
        filindsgne=vfil_amp(np.abs(magne),1.4826*sc.stats.median_abs_deviation(magne[np.isfinite(magne)])*gnesig,buff,minsamlen)
        filinds=np.intersect1d(filindsste,filindsgne)
        filindsster=vfil_amp(np.abs(master),1.4826*sc.stats.median_abs_deviation(master[np.isfinite(master)])*stesig,buff,minsamlen)
        filinds=np.intersect1d(filinds,filindsster)
        filindsgner=vfil_amp(np.abs(magner),1.4826*sc.stats.median_abs_deviation(magner[np.isfinite(magner)])*gnesig,buff,minsamlen)
        filinds=np.intersect1d(filinds,filindsgner)
        
        niter+=1
    
    return mc

##########################################################################################################################################

def AN_SINDy_vonly(data,libraryinfo,stcvinfo,sfinfo):
    
    """
    
    """
    
    #unpack arguments
    #data
    X=data[0]
    dXdt=data[1]
    #library
    librarytype=libraryinfo[0]
    deg=libraryinfo[1]
    #STCV
    Theta=globals()[librarytype](X,deg)
    lam=stcvinfo[0]
    rp0=stcvinfo[1]
    rp=stcvinfo[2]
    pt0=stcvinfo[3]
    nsteps=stcvinfo[4]
    ptf=stcvinfo[5]
    normalise=stcvinfo[6]
    #AN/ME
    nmeasteps=sfinfo[0]
    stesig=sfinfo[1]
    gnesig=sfinfo[2]
    buff=sfinfo[3]
    minsamlen=sfinfo[4]
    maxiter=sfinfo[5]
    
    niter=0
    filindsprev=np.array([])
    filinds=np.arange(len(X[:,0]))
    
    while niter<maxiter and not np.all(filindsprev==filinds):
        
        #preliminary model
        mc=SINDyCV_t_1(Theta[filinds],dXdt[filinds],lam,rp0,rp,pt0,nsteps,ptf,normalise)
        mc[:,0]=np.zeros_like(mc[:,0])
        mc[2,0]=np.median(dXdt[:,0]/X[:,1])

        #ME
        me=(Theta@mc-dXdt)[:,1]
        mea=moving_average(me,nmeasteps)

        #filtering
        filindsprev=np.copy(filinds)
        filinds=vfil_amp(np.abs(mea),1.4826*sc.stats.median_abs_deviation(mea)*stesig,buff,minsamlen)
        
        niter+=1
    
    return mc

##########################################################################################################################################



##########################################################################################################################################