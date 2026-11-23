"""Non-perturbative source-flux tracer: exact hourly cohorts and age-aligned half-life.

Uses calibrated flux equations from Run 3 Model.py. Reuses stored 1000-year spin-up
end states; no change to calibrated parameters or physical background stocks.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from numba import njit
from Model import SITES, POOLS, crossing, save_csv

@njit(cache=False)
def trace(x, rain, npp, q, ct, TI, label, days_window, years):
    dt,t1,t2,t3,t4,tref,temp,w1,w2,sfc,sp,klit,pdec,pnpp,pi,kb,vpa,kpa,amax,vpl,kpl,kpe,val,kal,kae,vlm,klm,Qmax,astar,ms,ns,mg,ng,KDOC,KO,kmm,CUE,sh,sw,sstar,emax,ew,zr,mp,m,ktex,ktot,ptex,ptot,alpha1,beta1,alpha2,beta2,rstar=q
    st=(t2+t3/3.14*np.arctan(3.14*t4*(temp-t1)))/(t2+t3/3.14*np.arctan(3.14*t4*(tref-t1)))
    perday=int(round(1/dt)); n=len(rain); window_steps=days_window*perday
    total_steps=n*(years+1)+window_steps; start=n-1
    # At most one cohort per hourly timestep in the labeling window.
    z=np.zeros((window_steps,8)); dose=np.zeros(window_steps)
    # Index row = elapsed DAYS since cohort entered POC.
    aged=np.zeros((years*365+1,8))
    allocated=0.; nc=0
    for step in range(total_steps):
        i=step%n
        A,P,M,D,B,L,s,N=x
        ks=ktot-(ktot-ktex)/(1+(A/alpha1)**beta1)-ktex
        phi=ptot-(ptot-ptex)/(1+(A/alpha2)**beta2)
        through=rain[i]-min(rain[i],rstar); s=min(1.,s+through/(phi*zr))
        if s<=sh: et=0.
        elif s<=sw: et=ew*(s-sh)/(sw-sh)
        elif s<=sstar: et=ew+(emax-ew)*(s-sw)/(sstar-sw)
        else: et=emax
        sm=ptex/phi
        if s<=sm:
            se=(s-sh)/(sm-sh); lk=ktex*se**.5*(1-(1-se**(1/mp))**mp)**2*dt
        else:
            se=(s-sm)/(1-sm); lk=(ktex+ks*se**.5*(1-(1-se**(1/m))**m)**2)*dt
        s-=(et*dt+lk)/(phi*zr)
        if not np.isfinite(s) or s<=0 or s>1: raise ValueError('Invalid water state')
        ww=1/(1+w1*np.exp(-w2*(s-sp)/(sfc-sp)))
        decomp=ww*st*klit*L; Fi=pdec*decomp
        Fa=kb*st*ww*A; Fpa=vpa*st*ww*P/(P+kpa)*(1-A/amax)
        Fpl=vpl*st*ww*P/(kpl+P)*B/(kpe+B)
        Fal=val*st*ww*(A*.001)/(kal+A*.001)*B/(kae+B)
        Fl=lk*D/(phi*s*zr); Flm=st*ww*klm*D*(Qmax-M)/Qmax; Fld=M/Qmax
        dc=D*phi**(astar*(ms-ns))*(s*phi)**(astar*ns)
        ox=.209*(phi-s*phi)**(4/3)*phi**(mg-ng)*(s*phi-phi)**ng
        Flb=vlm*st*B*dc/(dc+KDOC)*ox/(ox+KO); Fbm=kmm*st*B
        rates=np.array([Fa,Fpa,Fpl,Fal,Fl,Fld,Flb,Fbm,decomp])
        if np.any(rates < -1e-12) or not np.all(np.isfinite(rates)) or not np.isfinite(Flm): raise ValueError('Invalid flux')
        N+=(1-pnpp)*npp[i]*dt; Ln=L-decomp*dt; event=i==n-1
        if event: Ln+=(1-.45)*N; N=0.
        inc=.5*Ln if event and ct else 0.
        disruption=TI*A if event and ct else 0.
        # Previously labeled cohorts advance one physical hour using background fluxes.
        for j in range(nc):
            fA=z[j,0]/A; fP=z[j,1]/P; fM=z[j,2]/M
            fD=z[j,3]/D; fB=z[j,4]/B; fL=z[j,5]/L
            za=Fa*fA*dt; zp=Fpa*fP*dt; zd=Fpl*fP*dt; zal=Fal*fA*dt
            zm=Flm*(fD if Flm>=0 else fM)*dt; zmd=Fld*fM*dt
            zb=Flb*fD*dt; zbm=Fbm*fB*dt; zl=decomp*fL*dt
            # Match Run 3 tracer bookkeeping exactly.
            z[j,0]+=zp-za-zal
            z[j,1]+=za-zp-zd+pi*pdec*zl
            z[j,2]+=zm-zmd
            z[j,3]+=zd+zal+zbm-zb-Fl*fD*dt-zm+zmd+(1-pi)*pdec*zl
            z[j,4]+=CUE*zb-zbm
            z[j,5]-=zl
            z[j,6]+=(1-CUE)*zb+(1-pdec)*zl
            z[j,7]+=Fl*fD*dt
            if event and ct:
                # Aggregate disruption uses PRE-timestep labeled AAC, as in Run 3.
                dis=TI*(fA*A)
                z[j,0]-=dis; z[j,1]+=dis
        # Advance physical background, exactly as in Model.year_run.
        x=np.array([A+(-Fa+Fpa-Fal)*dt-disruption,
                    P+(pi*Fi+Fa-Fpa-Fpl)*dt+inc+disruption,
                    M+(Flm-Fld)*dt,
                    D+((1-pi)*Fi+Fpl+Fal+Fbm-Flb-Fl-Flm+Fld+pnpp*npp[i])*dt,
                    B+(CUE*Flb-Fbm)*dt,Ln-inc,s,N])
        if np.any(x[:6]<=0) or not np.all(np.isfinite(x)): raise ValueError('Invalid background stock')
        # Source labeling happens AFTER the just-completed physical inflow.
        # AAC->POC recycling is not an eligible source. No extra physical C.
        if start<=step<start+window_steps and allocated<label-1e-14:
            source=pi*Fi*dt+inc
            take=min(source,label-allocated)
            if take>0:
                z[nc,1]=take
                dose[nc]=take
                nc+=1; allocated+=take
        # Sample each cohort on its exact integer-day ages, including age 0.
        # Cohort j enters at start+j; 1 day later is start+j+perday.
        for j in range(nc):
            elapsed=step-(start+j)
            if elapsed>=0 and elapsed%perday==0:
                day=elapsed//perday
                if day<len(aged):
                    for k in range(8): aged[day,k]+=z[j,k]
    return aged,dose,allocated,x

def run_source(end_states='End_States', output='Results_SourceFlux', sites=None,
               label=1., window_days=14, years=100):
    """Track real residue->POC inputs as hourly cohorts after the next CT event.

    Age=0 for each cohort on its own entry hour, not at end of labeling window.
    All source doses must sum to label or the simulation raises an error.
    """
    if label<=0 or not 1<=window_days<=30 or years<1:
        raise ValueError('Require label>0, 1<=window_days<=30, years>=1')
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    sites=SITES if sites is None else sites
    summaries=[];fates=[];labels=[]
    for site in sites:
        for ct in (True,False):
            treatment='CT' if ct else 'NT'
            path=Path(end_states)/f'{site}_{treatment}_End.npz'
            with np.load(path) as src:
                x=src['x'].copy();rain=src['rain'];npp=src['npp'];q=src['q'];TI=float(src['TI'])
            aged,dose,allocated,_=trace(x,rain,npp,q,ct,TI,label,window_days,years)
            if not np.isclose(allocated,label,rtol=0,atol=1e-10):
                raise ValueError(f'{site} {treatment}: only {allocated:.6g} g labeled in {window_days} days')
            if np.any(aged < -1e-7):
                raise ValueError(f'{site} {treatment}: negative tracer amount')
            df=pd.DataFrame(aged,columns=POOLS+['Respired','Leached'])
            df.insert(0,'Year',np.arange(len(df))/365)
            df['SOC']=df[POOLS[:5]].sum(axis=1)
            df['Total']=df.SOC+df.L
            df['Balance']=df.Total+df.Respired+df.Leached-label
            error=float(df.Balance.abs().max())
            if error>1e-7*label: raise ValueError(f'{site} {treatment}: balance {error}')
            save_csv(df,root/f'{site}_{treatment}_Age_Daily.csv.gz',compact=True)
            save_csv(df.iloc[::365],root/f'{site}_{treatment}_Annual.csv')
            half=crossing(df.Year.to_numpy(),df.SOC.to_numpy(),.5*label)
            loss90=crossing(df.Year.to_numpy(),df.SOC.to_numpy(),.1*label)
            summaries.append(dict(Site=site,Treatment=treatment,Label=label,Window_days=window_days,
                                  Cohorts=int(np.count_nonzero(dose)),Half_life=half,Loss90=loss90,Balance=error))
            for day in (0,365,5*365,10*365,25*365,50*365,100*365):
                if day<len(df): fates.append(dict(Site=site,Treatment=treatment,**df.iloc[day].to_dict()))
            for i,amt in enumerate(dose):
                if amt>0: labels.append(dict(Site=site,Treatment=treatment,Entry_hour=i,
                                             Entry_day=i/24,Label=amt))
            print(f'{site} {treatment}: {np.count_nonzero(dose)} cohorts, half-life {half:.3f} yr, balance {error:.2e}',flush=True)
    summary=pd.DataFrame(summaries)
    save_csv(summary,root/'Summary.csv')
    save_csv(pd.DataFrame(fates),root/'Fate.csv')
    save_csv(pd.DataFrame(labels),root/'Entry_Cohorts.csv')
    return summary

if __name__=='__main__': run_source()
