"""
Coupled CENTURY water-carbon model for all eight sites.
"""

# Packages
import numpy as np
import pandas as pd
import tqdm

def poisson_process(n, lbd, alpha):
    """
    Creates rainfall series based on Poisson process.

    Parameters
    ----------
    n: array_like
       The number of events
    lbd: array_like
       The rain events mean distribution
    alpha: array_like
       Mean rain depth

    Returns
    -------
    samples: ndarray
       n Rain events in the same unit of alpha.
    """
    rain = np.zeros(n)
    for i in np.arange(n):
        n1 = np.random.uniform(0, 1, 1)
        if n1 < lbd:
            n2 = np.random.uniform(0, 1, 1)
            rain[i] = -alpha * np.log((1 - n2))
        else:
            rain[i] = 0
    return rain


def canopy(rain, rstar):
    """
    Returns the amount of water intercepted by the vegetation canopy
    """
    if rain <= rstar:
        return rain
    else:
        return rstar


def troughfall(rain, rstar):
    return rain - canopy(rain, rstar)


def evapotranspiration(s, sh, sw, sstar, emax, ew):
    """
    Computes evapotranspiration
    Parameters
    ----------
        s: soil moisture
        sh: soil moisture at hygroscopic point
        sw: soil moisture at wilting point
        sstar: soil moisture where ET decreases with s
        ew: evaporation rate
        emax: maximum evapotranspiration rate
    """
    if s <= sh:
        et = 0
    elif sh < s <= sw:
        et = ew * (s - sh) / (sw - sh)
    elif sw < s <= sstar:
        et = ew + (emax - ew) * (s - sw) / (sstar - sw)
    else:
        et = emax
    return et


def leakage_carbon(s, sh, phitex, phi, kstex, m_prime, ksstr, m):
    """
    Compute the total leakage given the soil conditions based on carbon dynamics
    Parameters
    ----------
    s: soil moisture
    sh: soil moisture at hygroscopic point
    phitex: textural porosity
    phi : total porosity
    kstex : textural hydraulic conductivity
    m_prime : shape parameter
    ksstr : structural hydraulic conductivity
    m : shape parameter

    """
    s_m = phitex / phi
    if s <= s_m:
        se_prime = (s - sh) / (s_m - sh)
        y = kstex * ((se_prime ** 0.5) * (1 - (1 - se_prime ** (1 / m_prime)) ** m_prime) ** 2)
    else:
        se = (s - s_m) / (1 - s_m)
        y = kstex + ksstr * ((se ** 0.5) * (1 - (1 - se ** (1 / m)) ** m) ** 2)
        pass
    return y


def swb_f(s, rain, npp, sh, sw, sstar, emax, ew, phi, zr, rstar, phitex, kstex, m_prime, ksstr, m, dt):
    """
    Computes soil water balance for one time step
    """

    # Soil water storage capacity
    swc = phi * zr

    # Canopy interception
    ci = canopy(rain, rstar)
    tf = rain - ci  # Througfall

    # Add througfall
    s = s + tf / swc

    # Verify if there is runoff
    if s > 1:
        q = (s - 1) * swc
        s = 1
    else:
        s = s
        q = 0
        
    I = tf - q  # m
    I_rate = (I * 1000)*dt       # converting I from 'm' to 'mm/day', since I_rate is fed as XX in EPIC in 'mm/day'

    et = evapotranspiration(s, sh, sw, sstar, emax, ew) * dt
    lk = leakage_carbon(s, sh, phitex, phi, kstex, m_prime, ksstr, m) * dt

    s = s - (et + lk) / swc
    npp_ = npp
    
    output = {'Rain': rain, 'CI': ci, 'Q': q, 's': s, 'ET': et, 'Lk': lk, 'npp_': npp_, 'I' : I, 'I_rate' : I_rate, 'swc' : swc}
    return output

# CENTURY model -------------------------------------------------------------

def f_tex(c1, c2, CS_Fr):
    
    """
    Parameters:
    Ftex: Soil texture effect on decomposition of active pool
    c1 : Intercept of clay fraction relationship
    c2 : Slope of clay fraction relationship
    CS_Fr : percent of soil that is in the silt + clay fractions (in %)
    """
    Ftex = (c1 - (c2 * (CS_Fr * 0.01)))
    
    return Ftex 

def st(t):
    """Temperature scalar using site-specific temperature."""
    if t <= 35:
        return t / (t + np.exp(5.058 - 0.2504 * t))
    else:
        return 1 - 0.04 * t  
        
def sut(s, sfc):
    
    w1 = 30.0
    w2 = 9.0

    return 1.0 / (1.0 + w1 * np.exp(-w2 * s / sfc))

def ox(z5, TF):
    
    """
    ox: oxygen control on biological processes with soil depth
    z5 : depth (m) from soil surface to center of soil layer under consideration
    """
    if TF == 1 or TF == 2:
        OX =  1 - (0.9 * z5) / (z5 + np.exp(16.79 - 0.0196 * z5))
        
    else:
        OX = 1
    
    return OX

def bulkdensity(ACTIVE, SLOW, PASSIVE, db, dbm):
    
    """
    db: bulk density of soil
    dbm: bulk density of mineral
    SOC: Soil organic carbon (gC/kg)
    d: depth of layer
    """
    d = 30                 # in centimeters, Sun et al., 2019
    
    ACTIVE = float(ACTIVE)/100     # converting g-C/m2 into MgC/ha to get SOC concentration in g/kg
    SLOW = float(SLOW)/100         # converting g-C/m2 into MgC/ha to get SOC concentration in g/kg
    PASSIVE = float(PASSIVE)/100   # converting g-C/m2 into MgC/ha to get SOC concentration in g/kg
    
    SOC_stock_Mgha = ACTIVE + SLOW + PASSIVE
    
    SOC = (((SOC_stock_Mgha) * 10) / (float(db) * d))
    SOC = SOC/10                   # To honour equation 63 Adams Equation, it needs % of SOC, not SOC in g/kg
    
    if SOC >= 0 and SOC < 5.8:
        db = 100 / (((SOC * (1.724/0.244)) + (100 - (SOC * 1.724))/dbm))
        
    else:
        db = 0.244
        
    output = {'db' : db, 'SOC' : SOC}
    return output 

def bulkdensityplow(db, I_rate, rain, TF, dbp, SAN):   
    """
    lbd: frequency of rainfall event
    alpha: mean depth of rainfall event
    SAN: Percentage of Sand
    Z: depth of plow layer in (m)
    dbp0: bulk density of plow layer before tillage
    EF: Efficiency of tillage mixing
    """   
    Z = 0.3
    EF = 0.5
    
    if TF == 0 or TF == 3 or TF == 4:                          # no-tillage
        dbp = db
    
    elif TF == 1:
        dbp = dbp - (dbp - (0.667 * db))*EF
        
    elif TF == 2 and rain != 0:                  
        SZ = 0.2 * I_rate * (1 + (2 * SAN / (SAN + np.exp(8.597 - 0.075 * SAN)))) / (Z**0.6)
        F = SZ / (SZ + np.exp(3.92 - 0.0226 * SZ))
        dbp = dbp + F * (db - dbp)
        
    elif TF == 2 and rain == 0:                     # tillage + no rain  
        dbp = dbp
    
    return dbp
        
def x1(db, dbp, TF): 
    """
    x1: tillage control on residue decomposition
    db: Bulk Density
    dbp: Bulk density of plow layer
    """
    
    if TF == 0 or TF == 1 or TF == 2:
        X1 = np.exp(6.0 * (db - dbp))        
    else:
        X1 = 0
    return X1

def x3(s, sfc, TF):
    """Allow gradual CT/NT residue entry unless soil is too wet."""
    if TF == 2 or TF == 3 or TF == 4:
        if s > 1.1 * sfc:
            X3 = 0
        else:
            X3 = 1
    else:
        X3 = 0

    return X3

def cs(t, db, dbp, TF, sfc, z5, s):
    """
    cs: combined factor controlling biological processes
    stt : Temperature scalar
    sww : Water scalar
    ox: oxygen control on biological processes with soil depth
    x1: tillage control on residue decomposition
    """
    stt = st(t)
    sww = sut(s, sfc)
    OX = ox(z5, TF)
    X1 = x1(db, dbp, TF)
    
    if TF == 1 or TF == 2:
        CS = np.sqrt(stt * sww) * 0.8 * OX * X1
        CS = min(CS, 10)

    elif TF == 0:
        CS = 0
    
    else:
        CS = np.sqrt(stt * sww) * OX
        CS = CS
        
    return CS

def residue(t, db, dbp, sfc, NPP, NPP_c, RSD, TF, s, z5, dt):
    """Track surface residue separately from the tillage incorporation amount.

    Incoming RSD is the previous hour's surface residue. RSD_surface is
    returned for carryover. At TF=1, returned RSD supplies the existing
    87%/13% litter additions. Fi remains a daily flux for gradual NT entry.
    """
    HI = 0.45
    TLD = 0.3
    EF = 0.5

    # Between CT events, use the same gradual incorporation rule as TF=4.
    # Other CT decomposition/tillage multipliers elsewhere are unchanged.
    CS = cs(t, db, dbp, 4 if TF == 2 else TF, sfc, z5, s)
    X3 = x3(s, sfc, TF)
    RSD_surface = RSD
    NPP_c = NPP * dt + NPP_c

    if TF == 0:
        # Initial pre-harvest period: no incorporation.
        Fi = 0
        WBMX = 0
        F = 0

    elif TF == 1:
        # Add harvest residue to the carried-over surface stock.
        RSD_surface = RSD_surface + (1 - HI) * NPP_c
        NPP_c = 0

        # Keep your existing incorporation equation and variable RSD.
        RSD = RSD_surface
        RSD = RSD * (1 - np.exp(-56.9 * TLD * EF))
        RSD_surface = RSD_surface - RSD
        Fi = 0
        WBMX = 0
        F = 0

    else:
        if TF == 3:
            RSD_surface = RSD_surface + (1 - HI) * NPP_c
            NPP_c = 0

        F = RSD_surface / (RSD_surface + np.exp(4.59 - 0.00713 * RSD_surface))
        WBMX = CS * X3 * F
        Fi = WBMX * RSD_surface
        RSD_surface = RSD_surface - Fi * dt
        RSD = RSD_surface

    return {
        'Fi': Fi, 'NPP_c': NPP_c, 'WBMX': WBMX,
        'F': F, 'X3': X3, 'RSD': RSD, 'RSD_surface': RSD_surface,
    }

def f_StrLitter(StrLitter, kls, t, LigFrac, s, sfc):
    """Structural litter decomposition flux."""
    stt = st(t)
    sww = sut(s, sfc)

    Fls = StrLitter * kls * stt * sww * np.exp(-3 * LigFrac)
    return Fls
            
def f_MetLitter(MetLitter, klm, t, s, sfc):    
    """
    Flm: Metabolic litter decomposition flux
    MetLitter : Structural Litter pool
    klm : Turnover rate of metabolic litter pool
    stt : Temperature scalar
    sww : Water scalar
    """
    stt = st(t)
    sww = sut(s, sfc)
    
    Flm = MetLitter * klm * stt * sww
    return Flm

def f_ACTIVE(ka, t, ACTIVE, c1, c2, CS_Fr, s, sfc):    
    """
    Fa: Active pool decomposition flux
    ACTIVE : Active pool
    ka : Turnover rate of active pool
    stt : Temperature scalar
    sww : Water scalar
    """
    stt = st(t)
    sww = sut(s, sfc)    
    Ftex = f_tex(c1, c2, CS_Fr)
    
    Fa = ACTIVE * ka * stt * sww * Ftex
    return Fa
            
def f_SLOW(ks, t, SLOW, s, sfc):    
    """
    Fs: Slow pool decomposition flux
    SLOW : Slow pool
    ks : Turnover rate of slow pool
    stt : Temperature scalar
    sww : Water scalar
    """
    stt = st(t)
    sww = sut(s, sfc)
    
    Fs = SLOW * ks * stt * sww
    return Fs

def f_PASSIVE(kp, t, PASSIVE, s, sfc):    
    """
    Fp: Passive pool decomposition flux
    PASSIVE : Passive pool
    kp : Turnover rate of passive pool
    stt : Temperature scalar
    sww : Water scalar
    """ 
    stt = st(t)
    sww = sut(s, sfc)
    
    Fp = PASSIVE * kp * stt * sww   
    return Fp  

def change_StrLitter(pli, Fi, NPP_c, RSD, TF, StrLitter, kls, t, db, dbp, sfc, LigFrac, dt, s, z5):
    """
    Computes change in structural Litter with time   
    pli : Proportion of plant residue to structural litter pool
    STD : C input or residue
    kls : Turnover rate of structural litter pool
    Fls: Structural litter decomposition flux
    """

    Fls = f_StrLitter(StrLitter, kls, t, LigFrac, s, sfc)
    CS = cs(t, db, dbp, TF, sfc, z5, s)

    if TF == 1:
        StrLitter = StrLitter + ((pli * Fi) - (Fls * CS)) * dt + RSD * 0.87
    elif TF == 2:
        StrLitter = StrLitter + ((pli * Fi) - (Fls * CS)) * dt
    else:
        StrLitter = StrLitter + ((pli * Fi) - Fls) * dt
        
    output = {'Bulk Density' : db, 'Bulk Density Plow' : dbp, 'Fls' : Fls, 'StrLitter' : StrLitter, 'NPP_c' : NPP_c}
    return output
            
def change_MetLitter(pli, Fi, RSD, TF, MetLitter, klm, t, db, dbp, sfc, dt, s, z5):
    """Update metabolic litter using the supplied incorporation flux."""
    Flm = f_MetLitter(MetLitter, klm, t, s, sfc)
    CS = cs(t, db, dbp, TF, sfc, z5, s)

    if TF == 1:
        MetLitter = MetLitter + (((1 - pli) * Fi) - (Flm * CS)) * dt + 0.13 * RSD
    elif TF == 2:
        MetLitter = MetLitter + (((1 - pli) * Fi) - (Flm * CS)) * dt
    else:
        MetLitter = MetLitter + ((1 - pli) * Fi - Flm) * dt

    output = {'Flm' : Flm, 'MetLitter' : MetLitter}
    return output

def change_ACTIVE(StrLitter, kls, t, ACTIVE, SLOW, PASSIVE, db, dbp, sfc, LigFrac, MetLitter, klm, ka, c1, c2, CS_Fr, ks, kp, z5, plsa, psa, plma, ppa, TF,
                  dt, s):
    """Update the active pool from the current hourly pools."""
    Fls = f_StrLitter(StrLitter, kls, t, LigFrac, s, sfc)
    Flm = f_MetLitter(MetLitter, klm, t, s, sfc)
    Fa = f_ACTIVE(ka, t, ACTIVE, c1, c2, CS_Fr, s, sfc)
    Fs = f_SLOW(ks, t, SLOW, s, sfc)
    Fp = f_PASSIVE(kp, t, PASSIVE, s, sfc)

    OX = ox(z5, TF)
    X1 = x1(db, dbp, TF)
    CS = cs(t, db, dbp, TF, sfc, z5, s)

    inputs = ((1 - LigFrac) * plsa * Fls + Fs * psa + Flm * plma + Fp * ppa)

    if TF == 1 or TF == 2:
        ACTIVE = ACTIVE + (inputs - (Fa * CS)) * dt
    else:
        ACTIVE = ACTIVE + (inputs - Fa) * dt        
                     
    output = {'Fs': Fs, 'Fp': Fp, 'Fa': Fa, 'CS': CS, 'ACTIVE': ACTIVE, 'OX': OX, 'X1': X1}
    return output
                     
def change_slow(StrLitter, kls, t, ACTIVE, SLOW, db, dbp, sfc, LigFrac, ka, c1, c2, CS_Fr, ks, z5, plss, pap, TF, dt, s):
    """Update the slow pool from the current hourly pools."""
    Fls = f_StrLitter(StrLitter, kls, t, LigFrac, s, sfc)
    Fa = f_ACTIVE(ka, t, ACTIVE, c1, c2, CS_Fr, s, sfc)
    Fs = f_SLOW(ks, t, SLOW, s, sfc)

    Ftex = f_tex(c1, c2, CS_Fr)
    CS = cs(t, db, dbp, TF, sfc, z5, s)

    inputs = LigFrac * plss * Fls + Fa * (1 - Ftex - pap)

    if TF == 1 or TF == 2:
        SLOW = SLOW + (inputs - (Fs * CS)) * dt
    else:
        SLOW = SLOW + (inputs - Fs) * dt
                     
    output = {'Fa': Fa, 'Fs': Fs, 'SLOW': SLOW}
    return output
                     
def change_passive(ka, t, ACTIVE, SLOW, PASSIVE, db, dbp, sfc, c1, c2, CS_Fr, ks, kp, z5, pap, psp, TF, dt, s):
    """Update the passive pool from the current hourly pools."""
    Fa = f_ACTIVE(ka, t, ACTIVE, c1, c2, CS_Fr, s, sfc)
    Fs = f_SLOW(ks, t, SLOW, s, sfc)
    Fp = f_PASSIVE(kp, t, PASSIVE, s, sfc)

    CS = cs(t, db, dbp, TF, sfc, z5, s)

    inputs = Fa * pap + Fs * psp

    if TF == 1 or TF == 2:
        PASSIVE = PASSIVE + (inputs - (Fp * CS)) * dt
    else:
        PASSIVE = PASSIVE + (inputs - Fp) * dt        
                     
    output = {'Fa': Fa, 'Fs': Fs, 'Fp': Fp, 'PASSIVE': PASSIVE}       
    return output        

# hydraulic structures of the aggregates                     
                     
def kstot(sand, kstex):
    y =  kstex * (10 ** (0.64 - 0.35 * sand ** 0.13))
    return y


def phitot(sand, phitex): 
    y = phitex * (10 ** (0.64 - 0.35 * sand ** 0.13))
    return y


# ----------------------------------------------------------------------------
# Soil water balance coupled with CENTURY
# ----------------------------------------------------------------------------            
def swb_millennial(s, rain, npp, StrLitter0, NPP_c, RSD0, db0, dbp0, MetLitter0, ACTIVE0, SLOW0, PASSIVE0, soilp, r, h, c, w, u, v, TF0, dt):
    """
    Solve models together (one time step)
    """
    # --- soil params
    sh      = soilp['sh']
    sw      = soilp['sw']
    sstar   = soilp['sstar']
    emax    = soilp['emax']
    ew      = soilp['ew']
    zr      = soilp['zr']
    rstar   = soilp['rstar']
    kstex   = soilp['kstex']
    kstot   = soilp['kstot']
    phitex  = soilp['phitex']
    phitot  = soilp['phitot']
    m_prime = soilp['m_prime']
    m       = soilp['m']

    # Fixed site-specific hydraulic properties; independent of SLOW.
    ksstrp = 0
    phi = phitex

    # --- water balance
    sol_swb = swb_f(s, rain, npp, sh, sw, sstar, emax, ew, phi, zr, rstar, phitex, kstex, m_prime, ksstrp, m, dt)
    sr      = sol_swb['s']
    NPP     = sol_swb['npp_']
    I_rate  = sol_swb['I_rate']
    rain    = sol_swb['Rain']

    # --- UPDATED bulk densities for this step (from current pools)
    db_new  = bulkdensity(ACTIVE0, SLOW0, PASSIVE0, db0, soilp['dbm'])['db']
    dbp_new = bulkdensityplow(db_new, I_rate, rain, TF0, dbp0, soilp['SAN'])

    # --- CENTURY fluxes & pool updates
    Residue = residue(t=r["t"], db=db_new, dbp=dbp_new, sfc=r["sfc"], NPP=NPP, NPP_c=NPP_c, RSD=RSD0, TF=TF0, s=sr, z5=r["z5"], dt=dt)
    Fi        = Residue['Fi']
    RSD_new   = Residue['RSD']
    NPP_c_new = Residue['NPP_c']   

    strlitter = change_StrLitter(pli=h["pli"], Fi=Fi, NPP_c=NPP_c_new, RSD=RSD_new, TF=TF0, StrLitter=StrLitter0, kls=h["kls"], t=h["t"], db=db_new,                                            dbp=dbp_new, sfc=h["sfc"], LigFrac=h["LigFrac"], dt=dt, s=sr, z5=h["z5"])
    
    metlitter = change_MetLitter(pli=c["pli"],Fi=Fi,RSD=RSD_new, TF=TF0, MetLitter=MetLitter0, klm=c["klm"], t=c["t"], db=db_new, dbp=dbp_new, sfc=c["sfc"],
                                 dt=dt, s=sr, z5=c["z5"])

    active = change_ACTIVE(StrLitter=StrLitter0, kls=w["kls"], t=w["t"], ACTIVE=ACTIVE0, SLOW=SLOW0, PASSIVE=PASSIVE0, db=db_new, dbp=dbp_new, sfc=w["sfc"],
                           LigFrac=w["LigFrac"], MetLitter=MetLitter0, klm=w["klm"], ka=w["ka"], c1=w["c1"], c2=w["c2"], CS_Fr=w["CS_Fr"], ks=w["ks"], 
                           kp=w["kp"], z5=w["z5"], plsa=w["plsa"], psa=w["psa"], plma=w["plma"], ppa=w["ppa"], TF=TF0, dt=dt, s=sr)

    slow = change_slow(StrLitter=StrLitter0, kls=u["kls"], t=u["t"], ACTIVE=ACTIVE0, SLOW=SLOW0, db=db_new, dbp=dbp_new, sfc=u["sfc"], 
                       LigFrac=u["LigFrac"], ka=u["ka"], c1=u["c1"], c2=u["c2"], CS_Fr=u["CS_Fr"], ks=u["ks"], z5=u["z5"], plss=u["plss"], pap=u["pap"],  
                       TF=TF0, dt=dt, s=sr)

    passive = change_passive(ka=v["ka"], t=v["t"], ACTIVE=ACTIVE0, SLOW=SLOW0, PASSIVE=PASSIVE0, db=db_new, dbp=dbp_new, sfc=v["sfc"], c1=v["c1"], c2=v["c2"],
                             CS_Fr=v["CS_Fr"], ks=v["ks"], kp=v["kp"], z5=v["z5"], pap=v["pap"], psp=v["psp"], TF=TF0, dt=dt, s=sr)

    # --- promote new pools to state for return
    StrLitter0 = strlitter['StrLitter']
    MetLitter0 = metlitter['MetLitter']
    ACTIVE0    = active['ACTIVE']
    SLOW0      = slow['SLOW']
    PASSIVE0   = passive['PASSIVE']
    NPP_c0     = NPP_c_new
    RSD0       = Residue['RSD_surface']  # Carry only the surface remainder.
    
    Fa0  = active['Fa']
    Fs0  = slow['Fs']
    Fp0  = passive['Fp']
    Fls0 = strlitter['Fls']
    Flm0 = metlitter['Flm']
    CS0  = active['CS']
    OX0  = active['OX']
    X10  = active['X1']

    WBMX0 = Residue['WBMX']
    X30   = Residue['X3']
    F0    = Residue['F']
    
    # --- set updated bulk densities
    db0   = db_new
    dbp0  = dbp_new

    return {
        'StrLitter': StrLitter0, 'MetLitter': MetLitter0, 'ACTIVE': ACTIVE0, 'SLOW': SLOW0, 'PASSIVE': PASSIVE0,
        'NPP_c': NPP_c0, 'RSD': RSD0, 'RSD_surface': RSD0, 'Bulk Density': db0, 'Bulk Density Plow': dbp0, 'Fi': Fi,
        'Fa': Fa0, 'Fs': Fs0, 'Fp': Fp0, 'Fls': Fls0, 'Flm': Flm0,
        'npp_': NPP, 'CS': CS0, 'OX': OX0, 'X1': X10, 's': sr, 'WBMX': WBMX0, 'X3': X30, 'F': F0
    }  

def long_run(s, rain, npp, StrLitter0, NPP_c, RSD0, db0, dbp0, MetLitter0, ACTIVE0, SLOW0, PASSIVE0, soilp, r, h, c, w, u, v, dt):
    """
    Run the coupled swb and Century model for long-term
    -------

    """                
    nr = len(rain)
    s_out = np.zeros(nr)
    npp__out = np.zeros(nr)
    phi_out = np.zeros(nr)
    ksstr_out = np.zeros(nr)
    
    StrLitter_out = np.zeros(nr)
    MetLitter_out = np.zeros(nr)
    ACTIVE_out = np.zeros(nr)
    SLOW_out = np.zeros(nr)
    PASSIVE_out = np.zeros(nr)
    NPP_c_out = np.zeros(nr)
    RSD_out = np.zeros(nr)
    
    db_out = np.zeros(nr)
    dbp_out = np.zeros(nr)
    WBMX_out = np.zeros(nr)
    X3_out   = np.zeros(nr)
    F_out   = np.zeros(nr)
    
    CS_out = np.zeros(nr)
    OX_out = np.zeros(nr)
    X1_out = np.zeros(nr)

    Fi_out = np.zeros(nr)
    Fls_out = np.zeros(nr)
    Flm_out = np.zeros(nr)
    Fa_out = np.zeros(nr)
    Fs_out = np.zeros(nr)
    Fp_out = np.zeros(nr)
    
    # for i in range(nr):
    for i in tqdm.tqdm(range(nr)):
        if i <= 17520000:
            if i < 8760:
                TF0 = 0             #first 8759 hours
                
            elif i % 8760 == 0:
                TF0 = 1             #tillage at the end of each year
            
            else:                   #hours without tillage
                TF0 = 2
            
        else:                       #period of no-tillage
            if i % 8760 == 0:       #harvest during no-till period
                TF0 = 3
                
            else:                  #days without harvest during no-till period
                TF0 = 4


        y = swb_millennial(s, rain[i], npp[i], StrLitter0, NPP_c, RSD0, db0, dbp0, MetLitter0, ACTIVE0, SLOW0, PASSIVE0, soilp, r, h, c, w, u, v, TF0, dt)
                
        s_out[i] = y['s']
        npp__out[i] = y['npp_']
        #phi_out[i] = y['phi']
        #ksstr_out[i] = y['ksstr']
        
        StrLitter_out[i] = y['StrLitter']
        MetLitter_out[i] = y['MetLitter']
        ACTIVE_out[i] = y['ACTIVE']
        SLOW_out[i] = y['SLOW']
        PASSIVE_out[i] = y['PASSIVE']
        NPP_c_out[i] = y['NPP_c']
        RSD_out[i] = y['RSD']

        Fi_out[i] = y['Fi']
        Fa_out[i] = y['Fa']
        Fs_out[i] = y['Fs']
        Fp_out[i] = y['Fp']
        Fls_out[i] = y['Fls']
        Flm_out[i] = y['Flm']
        
        CS_out[i] = y['CS']
        OX_out[i] = y['OX']
        X1_out[i] = y['X1']
        db_out[i] = y['Bulk Density']
        dbp_out[i] = y['Bulk Density Plow']
        WBMX_out[i] = y['WBMX']
        X3_out[i] = y['X3']
        F_out[i] = y['F']
        
        s           = y['s']
        npp_        = y['npp_']
        
        StrLitter0  = y['StrLitter']
        MetLitter0  = y['MetLitter']
        RSD0        = y['RSD']
        ACTIVE0     = y['ACTIVE']
        SLOW0       = y['SLOW']
        PASSIVE0    = y['PASSIVE']
        
        NPP_c       = y['NPP_c']
        
        Fi          = y['Fi']
        Fa          = y['Fa']
        Fs          = y['Fs']
        Fp          = y['Fp']
        Fls         = y['Fls']
        Flm         = y['Flm']
        
        CS          = y['CS']
        OX          = y['OX']
        X1          = y['X1']
        db0         = y['Bulk Density']
        dbp0        = y['Bulk Density Plow']
        WBMX        = y['WBMX']
        X3          = y['X3']
        F           = y['F']
        
        pass                

    output = {'s': s_out, 'StrLitter': StrLitter_out, 'MetLitter': MetLitter_out, 'ACTIVE': ACTIVE_out, 'SLOW': SLOW_out, 'PASSIVE': PASSIVE_out, 'NPP_c': NPP_c_out, 'RSD': RSD_out, 'Fi': Fi_out, 'Fa': Fa_out, 'Fs': Fs_out, 'Fp': Fp_out, 'Fls' : Fls_out, 'Flm' : Flm_out, 'npp_': npp__out, 'CS': CS_out, 'OX': OX_out, 'X1': X1_out, 'Bulk Density' : db_out, 'Bulk Density Plow' : dbp_out, 'WBMX' : WBMX_out, 'X3' : X3_out, 'F' : F_out}
    
    output = pd.DataFrame(output)
    output["RSD_surface"] = output["RSD"]  # Explicit surface-stock column; do not sum both.

    y1 = int(len(output))
    hours_index = np.repeat(np.arange(1, y1+1), len(output) / y1)
    output["Hours"] = hours_index
    
    y2 = int((len(output))/24)
    days_index = np.repeat(np.arange(1, y2+1), len(output) / y2)
    output["Days"] = days_index
    
    y3 = int(len(output) / (365 * 1/dt))
    years_index = np.repeat(np.arange(1, y3+1), len(output) / y3)
    output["Years"] = years_index
    
    return output                    
                     
# Site-specific inputs, initial pools and original CSV destinations.
SITES = {
    "POI": {
        "annual_avg_npp": 456.3,
        "t": 6.3,
        "CS_Fr": 79,
        "SAN": 21,
        "sh": 0.36,
        "sw": 0.42,
        "sstar": 0.76,
        "sfc": 0.8,
        "ew": 0.019 / 100,
        "emax": 0.19 / 100,
        "zr": 20 / 100,
        "kstex": 0.16,
        "phitex": 0.5,
        "kstot": 0.21,
        "phitot": 0.66,
        "alpha": 14.42 / 1000,
        "lbd": 0.19,
        "dbm": 1.938,
        "db0": 1.52,
        "ACTIVE0": 1048.8,
        "SLOW0": 2097.6,
        "PASSIVE0": 7341.6,
        "output_files": [
            r"C:\Users\rishabh.singh\OneDrive - Texas A&M AgriLife\Desktop\Final Testing (Revision)\Poirier\POI_Century.csv",
            r"C:\Users\rishabh.singh\OneDrive - Texas A&M AgriLife\Desktop\Final Testing (Revision)\Results\POI_Century.csv",
        ],
    },
    "ZAN": {
        "annual_avg_npp": 996.5,
        "t": 19.4,
        "CS_Fr": 46,
        "SAN": 54,
        "sh": 0.19,
        "sw": 0.24,
        "sstar": 0.57,
        "sfc": 0.65,
        "ew": 0.031 / 100,
        "emax": 0.31 / 100,
        "zr": 20 / 100,
        "kstex": 0.88,
        "phitex": 0.55,
        "kstot": 0.99,
        "phitot": 0.62,
        "alpha": 16.64 / 1000,
        "lbd": 0.24,
        "dbm": 1.79,
        "db0": 1.589,
        "ACTIVE0": 553,
        "SLOW0": 1106,
        "PASSIVE0": 3871,
        "output_files": [
            r"C:\Users\rishabh.singh\OneDrive - Texas A&M AgriLife\Desktop\Final Testing (Revision)\Zanatta\ZAN_Century.csv",
            r"C:\Users\rishabh.singh\OneDrive - Texas A&M AgriLife\Desktop\Final Testing (Revision)\Results\ZAN_Century.csv",
        ],
    },
    "PGL": {
        "annual_avg_npp": 560,
        "t": 10.5,
        "CS_Fr": 75,
        "SAN": 25,
        "sh": 0.36,
        "sw": 0.42,
        "sstar": 0.76,
        "sfc": 0.8,
        "ew": 0.025 / 100,
        "emax": 0.25 / 100,
        "zr": 20 / 100,
        "kstex": 0.79,
        "phitex": 0.43,
        "kstot": 1.01,
        "phitot": 0.55,
        "alpha": 11.88 / 1000,
        "lbd": 0.22,
        "dbm": 1.6,
        "db0": 1.35,
        "ACTIVE0": 783.3,
        "SLOW0": 1567,
        "PASSIVE0": 5483,
        "output_files": [
            r"C:\Users\rishabh.singh\OneDrive - Texas A&M AgriLife\Desktop\Final Testing (Revision)\Puget & Lal\PGL_Century.csv",
            r"C:\Users\rishabh.singh\OneDrive - Texas A&M AgriLife\Desktop\Final Testing (Revision)\Results\PGL_Century.csv",
        ],
    },
    "BC": {
        "annual_avg_npp": 492.8,
        "t": 17.8,
        "CS_Fr": 70,
        "SAN": 30,
        "sh": 0.47,
        "sw": 0.52,
        "sstar": 0.78,
        "sfc": 1.0,
        "ew": 0.034 / 100,
        "emax": 0.34 / 100,
        "zr": 20 / 100,
        "kstex": 0.34,
        "phitex": 0.54,
        "kstot": 0.42,
        "phitot": 0.67,
        "alpha": 15.12 / 1000,
        "lbd": 0.17,
        "dbm": 1.442,
        "db0": 1.355,
        "ACTIVE0": 308.94,
        "SLOW0": 617.88,
        "PASSIVE0": 2162.58,
        "output_files": [
            r"C:\Users\rishabh.singh\OneDrive - Texas A&M AgriLife\Desktop\Final Testing (Revision)\Bhattacharya\BC_Century.csv",
            r"C:\Users\rishabh.singh\OneDrive - Texas A&M AgriLife\Desktop\Final Testing (Revision)\Results\BC_Century.csv",
        ],
    },
    "DOU": {
        "annual_avg_npp": 737.3,
        "t": 20,
        "CS_Fr": 53,
        "SAN": 47,
        "sh": 0.36,
        "sw": 0.42,
        "sstar": 0.76,
        "sfc": 0.8,
        "ew": 0.035 / 100,
        "emax": 0.35 / 100,
        "zr": 20 / 100,
        "kstex": 0.36,
        "phitex": 0.42,
        "kstot": 0.42,
        "phitot": 0.49,
        "alpha": 12.2 / 1000,
        "lbd": 0.22,
        "dbm": 1.507,
        "db0": 1.42,
        "ACTIVE0": 291.8,
        "SLOW0": 583.6,
        "PASSIVE0": 2043,
        "output_files": [
            r"C:\Users\rishabh.singh\OneDrive - Texas A&M AgriLife\Desktop\Final Testing (Revision)\Dou\DOU_Century.csv",
            r"C:\Users\rishabh.singh\OneDrive - Texas A&M AgriLife\Desktop\Final Testing (Revision)\Results\DOU_Century.csv",
        ],
    },
    "JIN": {
        "annual_avg_npp": 419.8,
        "t": 10.1,
        "CS_Fr": 70,
        "SAN": 30,
        "sh": 0.19,
        "sw": 0.24,
        "sstar": 0.57,
        "sfc": 0.65,
        "ew": 0.031 / 100,
        "emax": 0.31 / 100,
        "zr": 20 / 100,
        "kstex": 0.28,
        "phitex": 0.53,
        "kstot": 0.35,
        "phitot": 0.66,
        "alpha": 13.31 / 1000,
        "lbd": 0.13,
        "dbm": 1.405,
        "db0": 1.35,
        "ACTIVE0": 200.5,
        "SLOW0": 401,
        "PASSIVE0": 1403,
        "output_files": [
            r"C:\Users\rishabh.singh\OneDrive - Texas A&M AgriLife\Desktop\Final Testing (Revision)\JIN\JIN_Century.csv",
            r"C:\Users\rishabh.singh\OneDrive - Texas A&M AgriLife\Desktop\Final Testing (Revision)\Results\JIN_Century.csv",
        ],
    },
    "LOU": {
        "annual_avg_npp": 306.6,
        "t": 6.5,
        "CS_Fr": 64,
        "SAN": 36,
        "sh": 0.19,
        "sw": 0.24,
        "sstar": 0.57,
        "sfc": 0.65,
        "ew": 0.027 / 100,
        "emax": 0.27 / 100,
        "zr": 20 / 100,
        "kstex": 0.17,
        "phitex": 0.47,
        "kstot": 0.21,
        "phitot": 0.57,
        "alpha": 9.15 / 1000,
        "lbd": 0.15,
        "dbm": 1.464,
        "db0": 1.35,
        "ACTIVE0": 369.9,
        "SLOW0": 793.8,
        "PASSIVE0": 278,
        "output_files": [
            r"C:\Users\rishabh.singh\OneDrive - Texas A&M AgriLife\Desktop\Final Testing (Revision)\Lou\LOU_Century.csv",
            r"C:\Users\rishabh.singh\OneDrive - Texas A&M AgriLife\Desktop\Final Testing (Revision)\Results\LOU_Century.csv",
        ],
    },
    "AF": {
        "annual_avg_npp": 485.5,
        "t": 13.8,
        "CS_Fr": 62,
        "SAN": 38,
        "sh": 0.19,
        "sw": 0.24,
        "sstar": 0.57,
        "sfc": 0.65,
        "ew": 0.0389 / 100,
        "emax": 0.389 / 100,
        "zr": 20 / 100,
        "kstex": 0.08,
        "phitex": 0.45,
        "kstot": 0.1,
        "phitot": 0.54,
        "alpha": 5.59 / 1000,
        "lbd": 0.19,
        "dbm": 1.542,
        "db0": 1.43,
        "ACTIVE0": 366.4,
        "SLOW0": 732.7,
        "PASSIVE0": 2564.6,
        "output_files": [
            r"C:\Users\rishabh.singh\OneDrive - Texas A&M AgriLife\Desktop\Final Testing (Revision)\Results\AF_Century.csv",
            r"C:\Users\rishabh.singh\OneDrive - Texas A&M AgriLife\Desktop\Final Testing (Revision)\Alvaro-Fuentes\AF_Century.csv",
        ],
    },
}

# One shared decomposition-rate set is used in every pool equation at every site.
PARAMETERS = dict(kls=0.01, klm=0.16, ka=0.09, ks=0.0005, kp=0.00002)


def run_julia(rain, npp, kstex, kstot, phitex, phitot, site="AF"):
    """Run the selected site with the original hourly model."""
    parameters = SITES[site]
    s = 0.5
    StrLitter0 = 50
    MetLitter0 = 20
    ACTIVE0 = parameters["ACTIVE0"]
    SLOW0 = parameters["SLOW0"]
    PASSIVE0 = parameters["PASSIVE0"]
    NPP_c = 0
    RSD0 = 0
    db0 = parameters["db0"]
    dbp0 = 0.78
    dt = 1 / 24

    # Reuse each site's values in every pool to prevent conflicting copies.
    common = {name: parameters[name] for name in
              ["t", "sfc", "sw", "sh", "sstar", "emax", "ew", "zr"]}
    common.update(z5=0.3, rstar=0.0, kstex=kstex, phitex=phitex,
                  m_prime=0.286, m=1.5)
    r = dict(common)
    h = dict(common, pli=0.66, kls=PARAMETERS["kls"], LigFrac=0.3)
    c = dict(common, pli=0.66, klm=PARAMETERS["klm"])
    w = dict(common, **PARAMETERS, LigFrac=0.3, c1=0.85, c2=0.68, CS_Fr=parameters["CS_Fr"], plsa=0.5, psa=0.42, plma=0.45, ppa=0.45)
    u = dict(common, **PARAMETERS, LigFrac=0.3, c1=0.85, c2=0.68, CS_Fr=parameters["CS_Fr"], plss=0.7, pap=0.004)
    v = dict(common, **PARAMETERS, c1=0.85, c2=0.68, CS_Fr=parameters["CS_Fr"], pap=0.004, psp=0.03)
    soilp = dict(common, kstot=kstot, phitot=phitot, dbm=parameters["dbm"], SAN=parameters["SAN"])

    return long_run(s, rain, npp, StrLitter0, NPP_c, RSD0, db0, dbp0, MetLitter0, ACTIVE0, SLOW0, PASSIVE0, soilp, r, h, c, w, u, v, dt)
