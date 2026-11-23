# Execute all eight Millennial-Till sites with 500-year setup.
# Run this file from the same folder as Soil_Carbon_Model.py.
from pathlib import Path
import numpy as np
import Soil_Carbon_Model as md

# Site-specific inputs, taken from the original notebooks and Python models.
SITES = {
    'AF': {
        'alpha': 0.0056,
        'lbd': 0.19,
        'annual_avg_npp': 485.5,
        'PET': 1124.9,
        'sand': 38.0,
        'kstex': 0.08,
        'kstot': 0.1,
        'phitex': 0.45,
        't': 13.8,
        'sfc': 0.65,
        'Qmax': 1860,
        'sh': 0.19,
        'sw': 0.24,
        'sstar': 0.57,
        'emax': 0.00389,
        'ew': 0.000389,
    },
    'BC': {
        'alpha': 0.01512,
        'lbd': 0.17,
        'annual_avg_npp': 492.8,
        'PET': 1260.9,
        'sand': 30.0,
        'kstex': 0.34,
        'kstot': 0.42,
        'phitex': 0.54,
        't': 17.8,
        'sfc': 1.0,
        'Qmax': 1583.6,
        'sh': 0.47,
        'sw': 0.52,
        'sstar': 0.78,
        'emax': 0.0034,
        'ew': 0.00034,
    },
    'DOU': {
        'alpha': 0.0122,
        'lbd': 0.22,
        'annual_avg_npp': 737.3,
        'PET': 1303.6,
        'sand': 47.0,
        'kstex': 0.36,
        'kstot': 0.42,
        'phitex': 0.42,
        't': 20,
        'sfc': 0.8,
        'Qmax': 1811,
        'sh': 0.36,
        'sw': 0.42,
        'sstar': 0.76,
        'emax': 0.0035,
        'ew': 0.00035,
    },
    'JIN': {
        'alpha': 0.01331,
        'lbd': 0.13,
        'annual_avg_npp': 419.8,
        'PET': 850.7,
        'sand': 30.0,
        'kstex': 0.28,
        'kstot': 0.35,
        'phitex': 0.53,
        't': 10.1,
        'sfc': 0.65,
        'Qmax': 1422.3,
        'sh': 0.19,
        'sw': 0.24,
        'sstar': 0.57,
        'emax': 0.0031,
        'ew': 0.00031,
    },
    'LOU': {
        'alpha': 0.00915,
        'lbd': 0.147,
        'annual_avg_npp': 306.6,
        'PET': 972.6,
        'sand': 36.0,
        'kstex': 0.17,
        'kstot': 0.21,
        'phitex': 0.47,
        't': 6.5,
        'sfc': 0.65,
        'Qmax': 2432,
        'sh': 0.19,
        'sw': 0.24,
        'sstar': 0.57,
        'emax': 0.0027,
        'ew': 0.00027,
    },
    'PGL': {
        'alpha': 0.01188,
        'lbd': 0.22,
        'annual_avg_npp': 560,
        'PET': 884.04,
        'sand': 25.0,
        'kstex': 0.79,
        'kstot': 1.01,
        'phitex': 0.43,
        't': 10.5,
        'sfc': 0.8,
        'Qmax': 4590,
        'sh': 0.36,
        'sw': 0.42,
        'sstar': 0.76,
        'emax': 0.0025,
        'ew': 0.00025,
    },
    'POI': {
        'alpha': 0.01442,
        'lbd': 0.19,
        'annual_avg_npp': 456.3,
        'PET': 689.1,
        'sand': 21.0,
        'kstex': 0.16,
        'kstot': 0.21,
        'phitex': 0.5,
        't': 6.3,
        'sfc': 0.8,
        'Qmax': 3648,
        'sh': 0.36,
        'sw': 0.42,
        'sstar': 0.76,
        'emax': 0.0019,
        'ew': 0.00019,
    },
    'ZAN': {
        'alpha': 0.0166,
        'lbd': 0.24,
        'annual_avg_npp': 996.5,
        'PET': 1119.8,
        'sand': 54.0,
        'kstex': 0.88,
        'kstot': 0.99,
        'phitex': 0.55,
        't': 19.4,
        'sfc': 0.65,
        'Qmax': 1073.6,
        'sh': 0.19,
        'sw': 0.24,
        'sstar': 0.57,
        'emax': 0.0031,
        'ew': 0.00031,
    },
}

# Original, common simulation and tillage settings.
dt = 1 / 24
nyears = 700
alpha1 = alpha2 = 580
beta1 = beta2 = 6
KA_final = 9.0132
n_final = 1.7444
R_max = 3.8
s50 = 25.0
k = 0.08


def compute_TI(AI):
    numerator = AI ** n_final
    denominator = KA_final ** n_final + AI ** n_final
    norm_factor = R_max ** n_final / (KA_final ** n_final + R_max ** n_final)
    return np.minimum((numerator / denominator) / norm_factor, 1.0)


def TI_from_sand(sand):
    return 1.0 / (1.0 + np.exp(-k * (sand - s50)))


def run_site(site, sitep):
    # Rainfall and seasonal NPP: same hourly construction as the notebooks.
    rain = md.poisson_process(n=int(nyears * 365 * 24),
                              alpha=sitep['alpha'], lbd=sitep['lbd'] * dt)
    days = np.arange(365)
    shape = np.sin((np.pi * days) / 365)
    daily_npp = sitep['annual_avg_npp'] * (shape / np.sum(shape))
    hourly_npp = np.repeat(daily_npp, 24)
    npp = np.tile(hourly_npp, nyears)

    # Same AI-based and sand-based TI equations.
    total_rainfall = rain.sum()
    RPY = (total_rainfall / nyears) * 1000
    AI = sitep['PET'] / RPY
    TI_AI = compute_TI(AI)
    TI_sand = TI_from_sand(sitep['sand'])
    TI = (TI_sand + TI_AI) / 2

    phitot = md.phitot(sitep['sand'], sitep['phitex'])

    print(f"{site}: AI={AI:.4f}, TI={TI:.4f}, phitot={phitot:.4f}", flush=True)
    sim = md.run_julia(rain, npp, sitep['kstex'], sitep['kstot'], sitep['phitex'], phitot, alpha1, beta1, alpha2, beta2, TI, {'site': site, **sitep})
    sim_year = sim.pivot_table(values=['s', 'phi', 'ksstr', 'P', 'DOC', 'A', 'M', 'B', 'Fa', 'Fpa', 'Flb', 'Fl', 'Flm', 'Fpl', 'Fbm', 'Fld', 'Fal',
                'lk', 'npp_', 'L', 'Fi', 'Flpom', 'decomp'], index='Years')
    from IPython.display import display
    display(sim_year)
    
    output_dirs = [r"C:\Users\rishabh.singh\OneDrive - Texas A&M AgriLife\Desktop\Final Testing (Revision)\Millennial-Till Master File\Results",
                   r"C:\Users\rishabh.singh\OneDrive - Texas A&M AgriLife\Desktop\Final Testing (Revision)\Results",]

    for directory in output_dirs:
        output = Path(directory)
        output.mkdir(parents=True, exist_ok=True)
        sim_year.to_csv(output / f'{site}_Test_Zr.csv', index=True, header=True)
        print(f"Saved: {output / f'{site}_Test_Zr.csv'}", flush=True)

if __name__ == '__main__':
    for site, sitep in SITES.items():
        run_site(site, sitep)
