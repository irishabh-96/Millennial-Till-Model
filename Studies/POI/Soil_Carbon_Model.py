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

    et = evapotranspiration(s, sh, sw, sstar, emax, ew) * dt
    lk = leakage_carbon(s, sh, phitex, phi, kstex, m_prime, ksstr, m) * dt

    s = s - (et + lk) / swc
    npp_ = npp
        
    output = {'Rain': rain, 'CI': ci, 'Q': q, 's': s, 'ET': et, 'Lk': lk, 'npp_': npp_}
    return output

# Millennial model -------------------------------------------------------------

def st(t1, t2, t3, t4, t, t_ref):
    """
    Computes temperature scalar of daily time-step version of Century model
    Parameters
    ----------
        t1: x-axis location of inflection point
        t2: y-axis location of inflection point
        t3: distance from maximum point to minimum point
        t4: Slope of line at inflection point
         t:  Current temperature
        t_ref : Reference temperature of temperature scalar
    """
    stt = (t2 + (t3 / 3.14) * np.arctan(3.14 * t4 * (t - t1))) / \
        (t2 + (t3 / 3.14) * np.arctan(3.14 * t4 * (t_ref - t1)))
    return stt

def sw (w1,w2,s,sfc,sp):
    """
    Computes moisture scalar of daily time-step version of Century model
    Parameters
    ----------
        w1 : empirical parameter
        w2 : emperical parameter
    """
    
    sww = 1/(1 + w1 * np.exp(-w2*(s-sp)/(sfc-sp)))
    return sww

def agg_break_POM(kb, t1, t2, t3, t4, t, t_ref, s, w1, w2, sfc, sp, A):
    """
    Computes aggregate Carbon breakdown to POM
    Parameters
    ----------
         Kb : rate of carbon breakdown
         A :  aggregated carbon
        st : temperature scalar taken from daily version of Century model
    """
    
    stt = st(t1, t2, t3, t4, t, t_ref)
    sww = sw(w1, w2, s, sfc, sp)
    
    Fa = kb * stt * sww * A
    return Fa

def agg_form_POM(vpa, P, kpa, A, amax, t1, t2, t3, t4, t, t_ref, s, w1, w2, sfc, sp):
    """
    Computes Formation of Aggregate C from POM
    Parameters
    ----------
        Vpa: maximum rate of aggregate formation
         P: POM
         Kpa : half-saturation constant of aggregate formation
         A : aggregated carbon
        Amax: Maximum capacity of Carbon in soil aggregates
        st : temperature scalar taken from daily version of Century model
    """
    
    stt = st(t1, t2, t3, t4, t, t_ref)
    sww = sw(w1, w2, s, sfc, sp)
    
    Fpa = vpa* stt * sww * (P / (P+kpa)) * (1 - (A / amax))
    return Fpa

def POM_dec_DOC(vpl, P, kpl, kpe, B, t1, t2, t3, t4, t, t_ref, s, w1, w2, sfc, sp):
    """
    Computes depolymerization of POM into DOC
    Parameters
    ----------
        Vpl: maximum rate of POM decomposition
         P: POM
         Kpl : half-saturation constant
         B : microbial biomass carbon
        Kpe: half-saturation constant of microbial control on POM mineralization
        st : temperature scalar
        sw : moisture scalar
    """
    
    stt = st(t1, t2, t3, t4, t, t_ref)
    sww = sw(w1, w2, s, sfc, sp)
    
    Fpl = vpl * stt * sww * (P / (kpl + P)) * (B / (kpe + B))
    return Fpl

def A_dec_DOC(val, A, kal, kae, B, t1, t2, t3, t4, t, t_ref, s, w1, w2, sfc, sp):
    """
    Computes depolymerization of A into DOC
    Parameters
    ----------
        Val: maximum rate of A depolymerization
         A : aggregated carbon 
         Kal : half-saturation constant
         B : microbial biomass carbon
        Kae: half-saturation constant of microbial control on A mineralization
        st : temperature scalar
        sw : moisture scalar
    """
    
    stt = st(t1, t2, t3, t4, t, t_ref)
    sww = sw(w1, w2, s, sfc, sp)
    
    Fal = val * stt * sww * ((A*0.001) / (kal + (A*0.001))) * (B / (kae + B))
    return Fal

def DOC_leaching(s, DOC, lk, phi, zr):
    """
    Computes leaching loss of DOC
    Parameters
    ----------
    DOC : Dissolved organic carbon
    lk is the leakage from the soil water balance model
    phi : total porosity
    zr : root zone depth
    """
    Fl = lk * DOC / (phi * s * zr)
    return Fl


def adsorption_DOC(klm, DOC, M, Qmax, s, t1, t2, t3, t4, t, t_ref, w1, w2, sfc, sp):
    """
    Computes adsorption of DOC to MAOM
    Parameters
    ----------
       klm : binding affinity based on pH
       Qmax : maximum sorption capacity
    """
    sww = sw(w1, w2, s, sfc, sp)
    stt = st(t1, t2, t3, t4, t, t_ref)
    
    Flm = stt * sww * ((klm * Qmax * DOC - klm * DOC * M) / Qmax)
    return Flm

def desorption_DOC(M, Qmax):
    """
    Computes desorption of DOC to MAOM
    Parameters
    ----------
       M : mineral associated organic matter
       Qmax : maximum sorption capacity
    """
    Fld = M / Qmax
    return Fld

def DOC_biomass(s, vlm, t1, t2, t3, t4, t, t_ref, a_star, ms_star, ns_star, mg_star, ng_star, DOC, KDOC, KO, phi, B):
    """
    Computes microbial uptake of DOC
    Parameters
    ----------
        vlm: potential uptake rate of DOC
        st : temperature scalar taken from daily version of Century model
        B : microbial biomass
        DOC : dissolved organic carbon
        Phi : total porosity
        a_star : SOC-microorganisms collocation factor
        ms_star : cementation exponent for DOC
        ns_star : saturation exponent for DOC
        mg_star : cementation exponent for oxygen
        ng_star : saturation exponent for oxygen
        KDOC : half-saturation constant for DOC
        KO :   half-saturation constant for oxygen        
    """    
    stt = st(t1, t2, t3, t4, t, t_ref)

    Flb = vlm * stt * B * (DOC * phi ** (a_star * (ms_star - ns_star)) * (s * phi) ** (a_star * ns_star)) / (
        (DOC * phi ** (a_star * (ms_star - ns_star)) * (s * phi) ** (a_star * ns_star)) + KDOC) * (
        (0.209 * (phi - s * phi) ** (4 / 3) * phi ** (mg_star - ng_star) * (s * phi - phi) ** ng_star) / (
            0.209 * (phi - s * phi) ** (4 / 3) * phi ** (mg_star - ng_star) * (s * phi - phi) ** ng_star + KO))    
    return Flb

def bio_MAOM(kmm, t1, t2, t3, t4, t, t_ref, B):
    """
    Computes Carbon flow from microbial biomass B to MAOM
    Parameters
    ----------
        kmm: adsorption rate of microbial biomass
        st : temperature scalar taken from daily version of Century model
        B  : microbial biomass
    """
    stt = st(t1, t2, t3, t4, t, t_ref)
    
    Fbm = (kmm * stt * B)
    return Fbm

def litter_f(t1, t2, t3, t4, t, t_ref, s, w1, w2, sfc, sp, pnpp, L, NPP, NPP_c, klit, p_dec, TF, dt):
    """
    Simulates surface residue dynamics
    Parameters:
    L : current litter C
    NPP : net primary productivity
    TF : tillage flag
    klit : decomposition rate of surface litter
    pnpp : Part of NPP which goes into DOC and remaining goes to litter layer
    TF : Tillage Flag
    """  
    
    stt = st(t1, t2, t3, t4, t, t_ref)
    sww = sw(w1, w2, s, sfc, sp)
    
    decomp = sww * stt * klit * L
    
    HI = 0.45     
    NPP_c = ((1-pnpp)*NPP)*dt + NPP_c                # since, part of NPP goes to DOC, the remaining part (1-pnpp)*NPP will go for                                                                                     accumulation
    if TF == 0 or TF == 2 or TF == 4:
        L = L - decomp*dt                            # litter pool getting updated since there is no new addition of NPP.
        Fi = p_dec * decomp                          # flux going after decomposition of litter pool into POM and DOC
        Flpom = 0                                    # no carbon going directly from litter to POM
        
    elif TF == 1:                                    # harvest during tillage window. Remaining residue after harvest gets added to surface residue
        L = L + ((1-HI) * NPP_c) - decomp*dt         # accumulated NPP fed in the surface residue
        Fi = p_dec * decomp                          # flux going after decomposition of litter pool into POM and DOC
        Flpom = 0.5 * L                              # C going from litter to the POM pool directly
        L = L - Flpom                               # updating the litter pool based on what fraction goes to POM 
        NPP_c = 0                                    # all the accumulated NPP is in the litter, hence setting it back to 0 for next year  
    
    elif TF == 3:                                    # no-tillage window, harvest at the end of the year  
        L = L + ((1-HI) * NPP_c) - decomp*dt         # accumulated NPP fed in the surface residue
        Fi = p_dec * decomp                          # flux going after decomposition of litter pool into POM and DOC
        NPP_c = 0                                    # all the accumulated NPP is in the residue, hence setting it back to 0 for next year
        Flpom = 0                                    # no carbon going from litter to the POM pool directly                                 
          
    output = {'Fi':Fi, 'L':L, 'NPP_c' : NPP_c, 'Flpom' : Flpom, 'decomp' : decomp}
    return output

def pom_f(pi, kb, t1, t2, t3, t4, t, t_ref, s, A, vpa, P, kpa, amax, vpl, kpl, kpe, B, w1, w2, sfc, sp, TF, TI, Fi, L, NPP_c, Flpom, dt):  
    """
    Computes change in POM P with time
         pi : Proportion of C input allocated to POM
         Fi : C input from aboveground plant litter, root litter and root exudates
         pa : proportion of C in aggregate breakdown allocated to POM
         Fa : aggregate carbon formation from POM
         Fpa : aggregate carbon formation from POM
         Fpl : decomposition of POM into DOC
         TI : Tillage Intensity
    """

    Fa = agg_break_POM(kb, t1, t2, t3, t4, t, t_ref, s, w1, w2, sfc, sp, A)
    Fpa = agg_form_POM(vpa, P, kpa, A, amax, t1, t2, t3, t4, t, t_ref, s, w1, w2, sfc, sp)
    Fpl = POM_dec_DOC(vpl, P, kpl, kpe, B, t1, t2, t3, t4, t, t_ref, s, w1, w2, sfc, sp)
    
    if TF == 1:                                                            
        TI_Calculated = TI
        P = ((pi * Fi) + Fa - Fpa - Fpl) * dt + P + ((TI_Calculated * A)) + Flpom
        
    else:                                                       
        P = ((pi * Fi) + Fa - Fpa - Fpl) * dt + P
              
    output = {'Fa': Fa, 'Fpa': Fpa, 'Fpl': Fpl,'P': P}    
    return output

def doc_f(vpl, vlm, P, kpl, kpe, A, DOC, klm, M, Qmax, B, a_star, ms_star, ns_star, mg_star, ng_star, KDOC, KO, phi, pi, s, t1, t2, t3, t4, t, t_ref, lk, zr, dt, kmm, pnpp, w1, w2, sfc, sp, val, kal, kae, Fi, L, NPP, TF):
    """
    Computes change in DOC with time
         pi : proportion of C input allocated to POM
         Fi : C input from aboveground plant litter, root litter and root exudates
         Fpl : decomposition of POM into DOC
         Fal : depolymerization of aggregate
         Fbm : turnover of microbial biomass
         Flb : uptake of DOC by microbial biomass
         Fl  : DOC leaching loss
         Flm : adsorption of DOC to MAOM 
         Fld : desorption of MAOM
    """

    Fl = DOC_leaching(s, DOC, lk, phi, zr)
    Fpl = POM_dec_DOC(vpl, P, kpl, kpe, B, t1, t2, t3, t4, t, t_ref, s,w1, w2, sfc, sp)
    Fal = A_dec_DOC(val, A, kal, kae, B, t1, t2, t3, t4, t, t_ref, s, w1, w2, sfc, sp)
    Flm = adsorption_DOC(klm, DOC, M, Qmax, s, t1, t2, t3, t4, t, t_ref,w1, w2, sfc, sp)
    Flb = DOC_biomass(s, vlm, t1, t2, t3, t4, t, t_ref, a_star, ms_star, ns_star, mg_star, ng_star, DOC, KDOC, KO, phi, B)
    Fbm = bio_MAOM(kmm, t1, t2, t3, t4, t, t_ref, B)
    Fld = desorption_DOC(M, Qmax)
                  
    DOC = ((Fi * (1 - pi)) + Fpl + Fal + Fbm - Flb - Fl- Flm + Fld + (pnpp * NPP)) * dt + DOC
    output = {'Fl': Fl, 'Fpl': Fpl, 'Fal': Fal, 'Flm': Flm, 'Flb': Flb, 'Fbm': Fbm, 'Fld': Fld,'DOC': DOC}
    return output

def agg_f(M, A, amax, s, vpa, P, kpa, kb, t1, t2, t3, t4, t, t_ref, w1, w2, sfc, sp, val, kal, kae, B, TF, TI, dt):  
    """
    Computes change in A with time
         Fpa : decomposition of POM into DOC
         Fa : adsorption of DOC to MAOM
         Fal : uptake of DOC by microbial biomass
         TI : Tillage Intensity
    """
    
    Fpa = agg_form_POM(vpa, P, kpa, A, amax, t1, t2, t3, t4, t, t_ref, s,w1, w2, sfc, sp)
    Fa = agg_break_POM(kb, t1, t2, t3, t4, t, t_ref, s, w1, w2, sfc, sp, A)
    Fal = A_dec_DOC(val, A, kal, kae, B, t1, t2, t3, t4, t, t_ref, s, w1, w2, sfc, sp)
    
    if TF == 1:
        TI_Calculated = TI
        A = (- Fa + Fpa - Fal) * dt + A - ((A * TI_Calculated))
         
    else:
        A = (- Fa + Fpa - Fal) * dt + A
        
    output = {'Fpa': Fpa, 'Fa': Fa, 'Fal': Fal, 'A': A}
    return output

def maom_f(klm, DOC, M, Qmax, s, t1, t2, t3, t4, t, t_ref, B,A, w1, w2, sfc, sp, dt):  
    """
    Computes change in MAOM with time
         Flm : adsorption of DOC to MAOM
         Fld : desorption of MAOM
    """
    Flm = adsorption_DOC(klm, DOC, M, Qmax, s, t1, t2, t3, t4, t, t_ref, w1, w2, sfc, sp)
    Fld = desorption_DOC(M, Qmax)
    M = (Flm - Fld) * dt + M
    output = {'Flm': Flm, 'Fld': Fld, 'M': M}
    return output

def bio_f(DOC, vlm, B, a_star, ms_star, ns_star, mg_star, ng_star, KDOC, KO, kmm, t1, t2, t3, t4, t, t_ref, s, phi, CUE, dt): 
    """
    Computes change in B with time
         CUE : carbon use efficiency
         Flb : adsorption of DOC to MAOM
         Fbm : carbon flow from microbial biomass to MAOM
    """
    Flb = DOC_biomass(s, vlm, t1, t2, t3, t4, t, t_ref, a_star, ms_star, ns_star, mg_star, ng_star, DOC, KDOC, KO, phi, B)
    Fbm = bio_MAOM(kmm, t1, t2, t3, t4, t, t_ref, B)

    B = (CUE*Flb - Fbm) * dt + B
    output = {'Flb': Flb, 'Fbm': Fbm, 'B': B}
    return output

def ksstr(kstex, kstot, A, alpha, beta):
    """
    Returns the structural hydraulic conductivity
    Parameters
    ----------
    kstex : textural hydraulic conductivity
    kstot : total hydraulic conductivity
    A : aggregated carbon
    alpha : shape parameter
    beta : shape parameter
    """
    y = kstot - ((kstot - kstex) / (1 + (A / alpha) ** beta)) - kstex
    return y

def phistr(phitot, phitex, A, alpha, beta):
    """
    Returns the structural porosity
    Parameters
    ----------
    phitot : total porosity
    phitex : textural porosity
    A : aggregated carbon
    alpha : shape parameter
    beta : shape parameter
    """
    y = phitot - ((phitot - phitex) / (1 + (A / alpha) ** beta)) - phitex
    return y

def kstot(sand, kstex):
    """
    Returns maximum possible hydraulic conductivity (i.e., fully structured soil)
    Parameters
    ----------
    sand : sand fraction (%)
    kstex : textural hydraulic conductivity

    Returns
    -------
    """
    y = 10 ** (3.5 - 1.5 * sand ** 0.13) * kstex
    return y

def phitot(sand, phitex):
    """
    Returns maximum possible porosity (i.e., fully structured soil)
    Parameters
    ----------
    sand : sand fraction(%)
    phitex : textural porosity

    Returns
    -------
#     """
    y = phitex / (10 ** (3.5 - 1.5 * sand ** 0.13))
    return y

# ----------------------------------------------------------------------------
# Soil water balance coupled with Millennial
# ----------------------------------------------------------------------------

def swb_millennial(s, rain, npp, P0, A0, B0, M0, DOC0, L0, NPP_c, alpha1, beta1, alpha2, beta2, soilp, w, u, v, h, c, l, TF0, TI, dt):
    """
    Solve models together
    Returns
    -------
    """

    # Textural parameters
    # Soil parameters
    sh = soilp['sh']
    sw = soilp['sw']
    sstar = soilp['sstar']
    emax = soilp['emax']
    ew = soilp['ew']
    zr = soilp['zr']
    rstar = soilp['rstar']
    kstex = soilp['kstex']
    kstot = soilp['kstot']
    phitex = soilp['phitex']
    phitot = soilp['phitot']
    m_prime = soilp['m_prime']
    m = soilp['m']

    # Hydraulic parameterization
    ksstrp = ksstr(kstex, kstot, A0, alpha1, beta1)
    phistrp = phistr(phitot, phitex, A0, alpha2, beta2)
    phi = phitex + phistrp

    # Water balance
    sol_swb = swb_f(s, rain, npp, sh, sw, sstar, emax, ew, phi, zr, rstar, phitex, kstex, m_prime, ksstrp, m, dt)
    sr = sol_swb['s']
    lk = sol_swb['Lk']
    NPP = sol_swb['npp_']

    # Millennial model equations
    lit = litter_f(s=sr, L=L0, NPP=NPP, NPP_c=NPP_c, TF=TF0, **l, dt=dt)
    Fi = lit['Fi']
    L_new = lit['L']
    NPP_c_new = lit['NPP_c']
    Flpom = lit['Flpom']
    decomp = lit['decomp']
        
    pom = pom_f(s=sr, P=P0, A=A0, B=B0, dt=dt, TF = TF0, TI=TI, Fi=Fi, L=L_new, NPP_c=NPP_c_new, Flpom=Flpom, **w)
    doc = doc_f(s=sr, P=P0, M=M0, B=B0, DOC=DOC0, A=A0, lk=lk, Fi=Fi, L=L_new, NPP = NPP, dt=dt, zr=zr, phi=phi, TF=TF0, **u)
    agg = agg_f(s=sr, M=M0, P=P0, A=A0, B=B0, dt=dt, TF=TF0, TI=TI, **v)
    maom = maom_f(s=sr, DOC=DOC0, A=A0, M=M0, B=B0, dt=dt, **h)
    bio = bio_f(s=sr, DOC=DOC0, B=B0, dt=dt, phi=phi, **c)
    
    L0 = L_new
    P0 = pom['P']
    DOC0 = doc['DOC']
    A0 = agg['A']
    M0 = maom['M']
    B0 = bio['B']

    Fa0 = pom['Fa']
    Fpa0 = pom['Fpa']
    Flb0 = doc['Flb']
    Fl0 = doc['Fl']
    Flm0 = doc['Flm']
    Fpl0 = doc['Fpl']
    Fbm0 = doc['Fbm']
    Fld0 = maom['Fld']
    Fal0 = doc['Fal']
    Flm0 = maom['Flm']
    Fi0 = Fi
    NPP_c = NPP_c_new
    Flpom0 = Flpom
    decomp = decomp
    
    output = {'s': sr, 'phi': phi, 'ksstr': ksstrp, 'P': P0, 'DOC': DOC0, 'A': A0, 'M': M0, 'B': B0,
              'Flb' : Flb0, 'Fl' : Fl0, 'Flm' : Flm0, 'Fpl' : Fpl0, 'Fbm' : Fbm0, 'lk':lk, 'Fa': Fa0,
              'Fpa': Fpa0, 'Fld': Fld0, 'Fal':Fal0, 'Flm':Flm0, 'npp_': NPP, 'L': L_new, 'Fi': Fi0, 'NPP_c' : NPP_c,
             'Flpom' : Flpom0, 'decomp' : decomp}
    return output


def long_run(s, rain, npp, P0, A0, B0, M0, DOC0, L0, NPP_c, alpha1, beta1, alpha2, beta2, TI, soilp, w, u, v, h, c, l, dt):
    """
    Run the coupled swb and millennial model for long-term
    -------

    """
    # print(f"Long run A0: {A0}")
    nr = len(rain)
    s_out = np.zeros(nr)
    npp__out = np.zeros(nr)
    phi_out = np.zeros(nr)
    ksstr_out = np.zeros(nr)

    P_out = np.zeros(nr)
    DOC_out = np.zeros(nr)
    A_out = np.zeros(nr)
    M_out = np.zeros(nr)
    B_out = np.zeros(nr)
    L_out = np.zeros(nr)
    NPP_c_out = np.zeros(nr)
    
    Flpom_out = np.zeros(nr)
    decomp_out = np.zeros(nr)
    Fi_out = np.zeros(nr)
    Fa_out = np.zeros(nr)
    Fpa_out = np.zeros(nr)
    Flb_out = np.zeros(nr)
    Fl_out = np.zeros(nr)
    Flm_out= np.zeros(nr)
    Fpl_out = np.zeros(nr)
    Fbm_out = np.zeros(nr)
    Fld_out = np.zeros(nr)
    Fal_out = np.zeros(nr)
    Flm_out = np.zeros(nr)
    lk_out = np.zeros(nr)
    
    for i in tqdm.tqdm(range(nr)):
        if i <= 7008000:            # tillage duration 0-7008000 hours (0-800 years)
            if i < 8760:
                TF0 = 0             # first 8759 hours at the beginning of the simulation
                
            elif i % 8760 == 0:
                TF0 = 1             # tillage activity at the last time step of each year
            
            else:                   # remaining time steps without tillage
                TF0 = 2
            
        else:                       # period of no-tillage (800-1000 years)
            if i % 8760 == 0:       # harvest operation at the end of each year during no-till period
                TF0 = 3
                
            else:                   # hours without harvest during no-till period
                TF0 = 4 
        
        y = swb_millennial(s, rain[i], npp[i], P0, A0, B0, M0, DOC0, L0, NPP_c, alpha1, beta1, alpha2, beta2, soilp, w, u, v, h, c, l, TF0, TI, dt)
        
        s_out[i] = y['s']
        npp__out[i] = y['npp_']
        phi_out[i] = y['phi']
        ksstr_out[i] = y['ksstr']
        
        P_out[i] = y['P']
        DOC_out[i] = y['DOC']
        A_out[i] = y['A']
        M_out[i] = y['M']
        B_out[i] = y['B']
        L_out[i] = y['L']
        
        NPP_c_out[i] = y['NPP_c']
        
        Flpom_out[i] = y['Flpom']
        decomp_out[i] = y['decomp']
        Fi_out[i] = y['Fi']
        Fa_out[i] = y['Fa']
        Fpa_out[i] = y['Fpa']
        Flb_out[i] = y['Flb']
        Fl_out[i] = y['Fl']
        Flm_out[i] = y['Flm']
        Fpl_out[i] = y['Fpl']
        Fbm_out[i] = y['Fbm']
        Fld_out[i] = y['Fld']
        Fal_out[i] = y['Fal']
        Flm_out[i] = y['Flm']
        lk_out[i] = y['lk']
        
        s = y['s']
        npp_ = y['npp_']
        
        L0 = y['L']
        P0 = y['P']
        DOC0 = y['DOC']
        A0 = y['A']
        M0 = y['M']
        B0 = y['B']
        
        NPP_c = y['NPP_c']
        
        Flpom = y['Flpom']
        decomp = y['decomp']
        Fi = y['Fi']
        Fa = y['Fa']
        Fpa = y['Fpa']
        Flb = y['Flb']
        Fl = y['Fl']
        Flm = y['Flm']
        Fpl = y['Fpl']
        Fbm = y['Fbm']
        Fld = y['Fld']
        Fal = y['Fal']
        Flm = y['Flm']
        lk = y['lk']
        
        pass

    output = {'s': s_out, 'phi': phi_out, 'ksstr': ksstr_out, 'P': P_out, 'DOC': DOC_out, 'A': A_out, 'M': M_out, 'B': B_out,
              'Fa': Fa_out, 'Fpa': Fpa_out, 'Flb': Flb_out, 'Fl': Fl_out, 'Flm': Flm_out, 'Fpl': Fpl_out, 'Fbm': Fbm_out,
              'Fld': Fld_out, 'Fal': Fal_out, 'Flm': Flm_out, 'lk': lk_out, 'npp_': npp__out, 'Fi':  Fi_out, 'L': L_out, 'NPP_c' : NPP_c_out, 'Flpom' : Flpom_out, 'decomp' : decomp_out}
    
    output = pd.DataFrame(output)

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

def run_julia(rain, npp, kstex, kstot, phitex, phitot, alpha1, beta1, alpha2, beta2, TI):
    
     # Initial conditions
    s = 0.5
    L0 = 200
    P0 = 139.416
    DOC0 = 46.094
    A0 = 786.332
    M0 = 1140.25
    B0 = 124.242
    dt = 1 / 24
    NPP_c = 0
    
    # Parameters  
    # Surface Residue Parameters ----------------------------
    l = {'t1' : 15.4,
         't2' : 11.75,
         't3' : 29.7,
         't4' : 0.0309,
         't_ref' : 30,
         't' : 6.3,
         'w1': 30,
         'w2': 9,
         'sfc': 0.80,
         'sp': 0.032,
         'klit' : 0.01,
         'p_dec' : 0.40,
         'pnpp' : 0.2}    

    # POM parameters ---------------------------------
    w = {'pi': 0.66,
         'kb': 0.0004,
         't1' : 15.4,
         't2' : 11.75,
         't3' : 29.7,
         't4' : 0.0309,
         't_ref' : 30,
         't' : 6.3,
         'w1': 30,
         'w2': 9,
         'sfc': 0.80,
         'sp': 0.032,
         'vpa': 10,
         'kpa': 50,
         'amax': 2000,
         'vpl': 2.5,
         'kpl': 200,
         'kpe': 24}

    # DOC parameters -----------------------------------------
    u = {'vpl': 2.5,
         'vlm': 0.3,
         'kpl': 200,
         'kpe': 24,
         'val': 3,
         'kal': 200,
         'kae': 24,
         'klm': 0.2,
         'Qmax': 3648,
         'a_star': 0.794,
         'ms_star': 1.5,
         'ns_star': 2,
         'mg_star': 1.5,
         'ng_star': 2,
         'KDOC': 190,
         'KO': 0.005,
         'pi': 0.66,
         't1' : 15.4,
         't2' : 11.75,
         't3' : 29.7,
         't4' : 0.0309,
         't_ref' : 30,
         't' : 6.3,
         'w1': 30,
         'w2': 9,
         'sfc': 0.80,
         'kmm': 0.01,
         'sp': 0.032,
         'pnpp': 0.2}

    # Aggregate parameters --------------------------
    v = {'amax': 2000,
         'vpa': 10,
         'kpa': 50,
         'val': 3,
         'kal': 200,
         'kae': 24,
         'kb': 0.0004,
         't1' : 15.4,
         't2' : 11.75,
         't3' : 29.7,
         't4' : 0.0309,
         't_ref' : 30,
         't' : 6.3,
         'w1': 30,
         'w2': 9,
         'sfc': 0.80,
         'sp': 0.032}

    # MAOM parameters -------------------------------------
    h = {'klm': 0.2,
         'Qmax': 3648,
         't1' : 15.4,
         't2' : 11.75,
         't3' : 29.7,
         't4' : 0.0309,
         't_ref' : 30,
         't' : 6.3,
         'w1': 30,
         'w2': 9,
         'sfc': 0.80,
         'sp': 0.032}

    # BIO parameters -----------------------------------
    c = {'vlm': 0.3,
         'a_star': 0.794,
         'ms_star': 1.5,
         'ns_star': 2,
         'mg_star': 1.5,
         'ng_star': 2,
         'KDOC': 190,
         'KO': 0.005,
         't1' : 15.4,
         't2' : 11.75,
         't3' : 29.7,
         't4' : 0.0309,
         't_ref' : 30,
         't' : 6.3,
         'CUE': 0.5,
         'kmm': 0.01}

    # Water balance model -------------------------------
    soilp = {'sh': 0.36,
             'sw': 0.42,
             'sstar': 0.76,
             'emax': 0.19/100,
             'ew': 0.019/100,
             'zr': 20/100,
             'rstar': 0.0,
             'kstex': kstex,  
             'kstot': kstot,  
             'phitex': phitex,  
             'phitot': phitot, 
             'm_prime': 0.286,  
             'm': 1.5}        
         
    x = long_run(s, rain, npp, P0, A0, B0, M0, DOC0, L0, NPP_c, alpha1, beta1, alpha2, beta2, TI, soilp, w, u, v, h, c, l, dt)
    return x