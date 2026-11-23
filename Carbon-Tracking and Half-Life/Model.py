"""Non-invasive tracer of existing POC under CT and NT. No physical C addition."""
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from numba import njit

SITES = ['POI','ZAN','PGL','BC','DOU','JIN','LOU','AF']
POOLS = ['A','POM','MAOM','DOC','B','L']

@njit(cache=False)
def year_run(x, z, rain, npp, q, ct, TI, track):
    dt,t1,t2,t3,t4,tref,temp,w1,w2,sfc,sp,klit,pdec,pnpp,pi,kb,vpa,kpa,amax,vpl,kpl,kpe,val,kal,kae,vlm,klm,Qmax,astar,ms,ns,mg,ng,KDOC,KO,kmm,CUE,sh,sw,sstar,emax,ew,zr,mp,m,ktex,ktot,ptex,ptot,alpha1,beta1,alpha2,beta2,rstar=q
    st=(t2+t3/3.14*np.arctan(3.14*t4*(temp-t1)))/(t2+t3/3.14*np.arctan(3.14*t4*(tref-t1)))
    n=len(rain); perday=int(round(1/dt)); background=np.zeros((366,8)); tracer=np.zeros((366,8))
    background[0]=x; tracer[0]=z
    for i in range(n):
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
        inc=.5*Ln if event and ct else 0.; disruption=TI*A if event and ct else 0.
        if track:
            f=z[:6]/x[:6]; zz=z.copy()
            za=Fa*f[0]*dt; zp=Fpa*f[1]*dt; zd=Fpl*f[1]*dt; zal=Fal*f[0]*dt
            zm=Flm*(f[3] if Flm>=0 else f[2])*dt; zmd=Fld*f[2]*dt; zb=Flb*f[3]*dt; zbm=Fbm*f[4]*dt; zl=decomp*f[5]*dt
            zz[0]+=zp-za-zal; zz[1]+=za-zp-zd+pi*pdec*zl
            zz[2]+=zm-zmd; zz[3]+=zd+zal+zbm-zb-Fl*f[3]*dt-zm+zmd+(1-pi)*pdec*zl
            zz[4]+=CUE*zb-zbm; zz[5]-=zl
            zz[6]+=(1-CUE)*zb+(1-pdec)*zl; zz[7]+=Fl*f[3]*dt
            if event and ct:
                za=TI*z[0]; zi=.5*zz[5]; zz[0]-=za; zz[1]+=za+zi; zz[5]-=zi
            z=zz
        x=np.array([A+(-Fa+Fpa-Fal)*dt-disruption,P+(pi*Fi+Fa-Fpa-Fpl)*dt+inc+disruption,M+(Flm-Fld)*dt,D+((1-pi)*Fi+Fpl+Fal+Fbm-Flb-Fl-Flm+Fld+pnpp*npp[i])*dt,B+(CUE*Flb-Fbm)*dt,Ln-inc,s,N])
        if np.any(x[:6]<=0) or not np.all(np.isfinite(x)): raise ValueError('Invalid carbon stock')
        if track and (np.any(z < -1e-10) or np.any(z[:6]>x[:6]+1e-8)): raise ValueError('Invalid tracer stock')
        if (i+1)%perday==0:
            day=(i+1)//perday; background[day]=x; tracer[day]=z
    return x,z,background,tracer

def setup(info, seed):
    p=info['params']; site=info['site']; l,w,u,v,h,c=[p[k] for k in ['l','w','u','v','h','c']]; soil=p['soilp']
    # Shared parameters must agree so the source and destination use the same flux.
    for key in ['t1','t2','t3','t4','t_ref','t']:
        if not all(np.isclose(d[key],l[key]) for d in [w,u,v,h,c]): raise ValueError(f'Inconsistent {key}')
    for key in ['w1','w2','sfc','sp']:
        if not all(np.isclose(d[key],l[key]) for d in [w,u,v,h]): raise ValueError(f'Inconsistent {key}')
    groups=[(w,u,['vpl','kpl','kpe','pi']),(w,v,['kb','vpa','kpa','amax']),(u,v,['val','kal','kae']),(u,h,['klm','Qmax']),(u,c,['vlm','a_star','ms_star','ns_star','mg_star','ng_star','KDOC','KO','kmm']),(u,l,['pnpp'])]
    for a,b,keys in groups:
        for key in keys:
            if not np.isclose(a[key],b[key]): raise ValueError(f'Inconsistent {key}')
    dt=p['dt']; perday=int(round(1/dt))
    if not np.isclose(dt*perday,1): raise ValueError('dt must divide one day exactly')
    rng=np.random.default_rng(seed); n=365*perday
    rain=np.where(rng.random(n)<info['rain']['lbd'],rng.exponential(info['rain']['alpha'],n),0.)
    if rain.sum()==0: raise ValueError('No rainfall events; use another seed')
    rain*=n*info['rain']['lbd']*info['rain']['alpha']/rain.sum()
    shape=np.sin(np.pi*np.arange(365)/365); npp=np.repeat(site['annual_avg_npp']*shape/shape.sum(),perday)
    ai=site['PET']/(rain.sum()*1000); power=site['n_final']; ka=site['KA_final']; maximum=site['R_max']
    ti_ai=min(1,(ai**power/(ka**power+ai**power))/(maximum**power/(ka**power+maximum**power)))
    TI=(ti_ai+1/(1+np.exp(-site['k']*(site['sand']-site['s50']))))/2
    q=np.array([dt,l['t1'],l['t2'],l['t3'],l['t4'],l['t_ref'],l['t'],l['w1'],l['w2'],l['sfc'],l['sp'],l['klit'],l['p_dec'],l['pnpp'],w['pi'],w['kb'],w['vpa'],w['kpa'],w['amax'],w['vpl'],w['kpl'],w['kpe'],u['val'],u['kal'],u['kae'],u['vlm'],u['klm'],u['Qmax'],u['a_star'],u['ms_star'],u['ns_star'],u['mg_star'],u['ng_star'],u['KDOC'],u['KO'],c['kmm'],c['CUE'],soil['sh'],soil['sw'],soil['sstar'],soil['emax'],soil['ew'],soil['zr'],soil['m_prime'],soil['m'],soil['kstex'],soil['kstot'],soil['phitex'],soil['phitot'],site['alpha1'],site['beta1'],site['alpha2'],site['beta2'],soil['rstar']],dtype=float)
    x=np.array([p['A0'],p['P0'],p['M0'],p['DOC0'],p['B0'],p['L0'],p['s'],p['NPP_c']],dtype=float)
    return x,rain,npp,q,TI

def initialize_tracer(x, label=1.0, pool='POM'):
    """Identify existing carbon without altering background stocks.

    x: existing [A, POM, MAOM, DOC, B, L, water, NPP storage] state.
    z: tracer bookkeeping [A, POM, MAOM, DOC, B, L, respired, leached].
    """
    if pool not in POOLS[:5]:
        raise ValueError('The tracer must start in an SOC pool, not surface residue')
    if label <= 0:
        raise ValueError('label must be positive')
    i = POOLS.index(pool)
    if label > x[i]:
        raise ValueError(f'Cannot label {label:g} g C: {pool} contains {x[i]:g} g C')
    z = np.zeros(8)
    z[i] = label
    return x.copy(), z

def crossing(t,s,threshold):
    hit=np.flatnonzero(s<=threshold)
    if not len(hit): return np.nan
    i=hit[0]
    return t[0] if i==0 else t[i-1]+(threshold-s[i-1])*(t[i]-t[i-1])/(s[i]-s[i-1])

def save_csv(df, path, compact=False):
    path=Path(path); temporary=path.with_name('_'+path.name+'.tmp')
    compression={'method':'gzip','compresslevel':1} if path.suffix=='.gz' else None
    df.to_csv(temporary,index=False,float_format='%.12g' if compact else None,compression=compression)
    os.replace(temporary,path)

def run_model(config='sites.json', spinup=1000, years=100, label=1., seed=42, folder='Results', sites=None):
    if spinup<30 or years<1 or label<=0: raise ValueError('Use spinup>=30, years>=1 and label>0')
    data=json.loads(Path(config).read_text()); sites=SITES if sites is None else sites
    
    
    paths={'p_dec':('l','p_dec'),'vlm':('u','vlm'),'kmm':('u','kmm'),
           'amax':('w','amax'),'CUE':('c','CUE'),'vpa':('w','vpa'),
           'vpl':('w','vpl'),'kpa':('w','kpa'),'kpl':('w','kpl'),
           'kal':('u','kal'),'kpe':('w','kpe')}
    for key,(group,name) in paths.items():
        values=[data[site]['params'][group][name] for site in sites]
        if not np.allclose(values,values[0]):
            raise ValueError(f'Parameter {key} differs among sites: {dict(zip(sites,values))}')
    root=Path(folder); root.mkdir(parents=True,exist_ok=True); background=[]; summaries=[]; snapshots=[]; label_checks=[]
    for site in sites:
        initial,rain,npp,q,TI=setup(data[site],seed+SITES.index(site))
        for ct in [True,False]:
            treatment='CT' if ct else 'NT'; x=initial.copy(); annual=[]
            background.append(dict(Site=site,Treatment=treatment,Year=0,**dict(zip(POOLS,x[:6])),SOC=x[:5].sum(),Total=x[:6].sum()))
            for y in range(1,spinup+1):
                x,_,stocks,_=year_run(x,np.zeros(8),rain,npp,q,ct,TI,False); avg=stocks[1:,:6].mean(axis=0); annual.append(avg)
                background.append(dict(Site=site,Treatment=treatment,Year=y,**dict(zip(POOLS,avg)),SOC=avg[:5].sum(),Total=avg.sum()))
            drift=100*(np.mean(np.array(annual)[-10:],axis=0)-np.mean(np.array(annual)[-30:-20],axis=0))/np.mean(np.array(annual)[-30:-20],axis=0)
           
            np.savez_compressed(root/f'{site}_{treatment}_End.npz',x=x,rain=rain,npp=npp,q=q,TI=TI)
            
            before=x.copy()
            x,z=initialize_tracer(x,label=label,pool='POM')
            if not np.array_equal(x,before):
                raise AssertionError('Tracer initialization modified the actual SOC stocks')
            for i,pool in enumerate(POOLS):
                label_checks.append(dict(Site=site,Treatment=treatment,Pool=pool,
                                         Background=before[i],Labeled=z[i],StockChange=x[i]-before[i]))
            chunks=[]; stock_chunks=[]
            for y in range(years):
                x,z,stocks,tracer=year_run(x,z,rain,npp,q,ct,TI,True); times=y+np.arange(366)/365
                if y: stocks=stocks[1:]; tracer=tracer[1:]; times=times[1:]
                chunks.append(np.column_stack((times,tracer))); stock_chunks.append(np.column_stack((times,stocks[:,:6])))
            df=pd.DataFrame(np.vstack(chunks),columns=['Year']+POOLS+['Respired','Leached'])
            df['SOC']=df[POOLS[:5]].sum(axis=1); df['Total']=df.SOC+df.L; df['Balance']=df.Total+df.Respired+df.Leached-label
            if df.Balance.abs().max()>1e-8*max(1,label): raise ValueError('Tracer carbon balance failed')
            save_csv(df,root/f'{site}_{treatment}.csv.gz',compact=True)
            save_csv(df.iloc[::365],root/f'{site}_{treatment}_Annual.csv')
            total=pd.DataFrame(np.vstack(stock_chunks),columns=['Year']+POOLS); total['SOC']=total[POOLS[:5]].sum(axis=1); total['Total']=total.SOC+total.L
            save_csv(total.iloc[::365],root/f'{site}_{treatment}_Stocks.csv')
            summaries.append(dict(Site=site,Treatment=treatment,Spinup=spinup,Track=years,Label=label,Change=np.max(np.abs(drift)),Half_life=crossing(df.Year.to_numpy(),df.Total.to_numpy(),.5*label),Loss90=crossing(df.Year.to_numpy(),df.Total.to_numpy(),.1*label),Balance=df.Balance.abs().max()))
            for t in sorted(set([0,1,5,10,25,50,100,years])):
                if t<=years: snapshots.append(dict(Site=site,Treatment=treatment,**df.iloc[int(t*365)].to_dict()))
            print(f'{site} {treatment}: labeled {label:g} g existing POC; half-life {summaries[-1]["Half_life"]:.3f} yr; balance {summaries[-1]["Balance"]:.2e}',flush=True)
    save_csv(pd.DataFrame(background),root/'Background.csv')
    save_csv(pd.DataFrame(label_checks),root/'Label_Check.csv')
    save_csv(pd.DataFrame(snapshots),root/'Fate.csv')
    summary=pd.DataFrame(summaries); save_csv(summary,root/'Summary.csv')
    return summary

if __name__=='__main__': run_model(folder='Results_POC_Tracer')
