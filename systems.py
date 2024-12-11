# Compilation of dynamic systems wheich can be simulated


# Lorenz ################################################################################################################################

def lorenz(x,t):
    
    """
    Funtion to simulate lorenz system with sigma, rho, beta = 10, 28, 8/3.
    
    Inputs: (x,t)
    Outputs: dx
    """
    
    dx=[10*(x[1]-x[0]),
        x[0]*(28-x[2])-x[1],
        x[0]*x[1]-(8/3)*x[2]]
    
    return dx

# Lorenz ################################################################################################################################

def lorenzmod(x,t):
    
    """
    Funtion to simulate lorenz system with sigma, rho, beta = 10, 28, 8/3.
    
    Inputs: (x,t)
    Outputs: dx
    """
    
    dx=[10*(x[1]-x[0])+0.1*x[1]**3,
        x[0]*(28-x[2])-x[1],
        x[0]*x[1]-(8/3)*x[2]]
    
    return dx

# Van der Pol ###########################################################################################################################

def pol(x,t):
    
    """
    Funtion to simulate Van der Pol system with mu = 5.
    
    Inputs: (x,t)
    Outputs: dx
    """
    
    dx=[x[1],#-x[0],
        5*(1-x[0]**2)*x[1]-x[0]]
    return dx

# Duffing Oscillator ####################################################################################################################

def duf(x,t):
    dx=[x[1],
        -x[0]-4*x[0]**3]
    return dx

# Rossler ###############################################################################################################################

def rossler(x,t):
    dx=[10*(-x[1]-x[2]),
        10*(x[0]+0.1*x[1]),
        10*(0.1+x[2]*(x[0]-14))]
    return dx

# Lorenz ################################################################################################################################

def lorenz_(t,x):
    
    """
    Funtion to simulate lorenz system with sigma, rho, beta = 10, 28, 8/3.
    
    Inputs: (x,t)
    Outputs: dx
    """
    
    dx=[10*(x[1]-x[0]),
        x[0]*(28-x[2])-x[1],
        x[0]*x[1]-(8/3)*x[2]]
    
    return dx

# Van der Pol ###########################################################################################################################

def pol_(t,x):
    
    """
    Funtion to simulate Van der Pol system with mu = 5.
    
    Inputs: (x,t)
    Outputs: dx
    """
    
    dx=[x[1],#-x[0],
        5*(1-x[0]**2)*x[1]-x[0]]
    return dx

# Duffing Oscillator ####################################################################################################################

def duf_(t,x):
    dx=[x[1],
        -x[0]-4*x[0]**3]
    return dx

# Rossler ###############################################################################################################################

def rossler_(t,x):
    dx=[10*(-x[1]-x[2]),
        10*(x[0]+0.1*x[1]),
        10*(0.1+x[2]*(x[0]-14))]
    return dx

# bouncng ball function #################################################################################################################

def BB(tlim,dt,v0,g=-9.81,et=1):
    
    """
    Function to simulate a bouncing ball. Bounces are instantaneous and have consistent % velocity loss.
    
    Input: (tlim,dt,v0,g=-9.81,et=1)
    Returns (tstamps,Sol,(dt,g,impt[:-1],impv[:-1]))
    
    'tlim' is the length of the simulation in seconds.
    'dt' is the sampling rate.
    'v0' is the initial vetrical velocity (positive).
    'et' is the kinetic energy transfer efficiency.
    
    Returns timestamps and integrated solution, along with system/measurement specific constants. In this case, the impact times and respective velocity are returned.
    """
    
    def st(a,u,t):
        return u*t+0.5*a*t**2
    
    def vt(a,u,t):
        return u+a*t
    
    tstamps=np.arange(0,tlim,dt)
    
    t0=0
    tb=-2*v0/g
    
    Sol=np.array([st(g,v0,tstamps[np.sum(tstamps<t0):np.sum(tstamps<tb)]),
                  vt(g,v0,tstamps[np.sum(tstamps<t0):np.sum(tstamps<tb)])])
    
    v0=np.sqrt(et)*v0
    
    impt=[tb]
    impv=[v0]
    
    while tb<tlim:
        
        t0=tb
        tb=tb-2*v0/g
        
        sol=np.array([st(g,v0,tstamps[np.sum(tstamps<t0):np.sum(tstamps<tb)]-t0),
                      vt(g,v0,tstamps[np.sum(tstamps<t0):np.sum(tstamps<tb)]-t0)])
        Sol=np.hstack((Sol,sol))
        
        v0=np.sqrt(et)*v0
        
        impt.append(tb)
        impv.append(v0)
    
    Sol=Sol[:,:len(tstamps)]
    
    return tstamps,Sol,(dt,g,impt[:-1],impv[:-1])

