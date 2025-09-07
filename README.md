This repository hosts the Millennial-Till model, which explicitly represents the impacts of conventional tillage (CT) and no-tillage (NT) on soil organic carbon (SOC) pools and dynamics.

The repository includes site-specific implementations representing median study sites across diverse climatic conditions and crop productivity gradients. Each study is given a study name identified by a code — **POI, ZAN, PGL, BC, JIN, DOU, LOU, and AF**.
  1. For each site, the main simulation code is provided in **Soil Carbon Model.py**, which integrates soil water balance processes with C cycling dynamics. This script also specifies the parameter values under different soil textures and climatic conditions.
  2. Model runs are executed through the Jupyter notebook **Execution File.ipynb**.
  3. Simulation results are exported as **StudyName.csv** files (example **POI.csv**), containing the full temporal variations of SOC pools and fluxes across the modeled period.

Contact Rishabh Singh (irishabh1996@tamu.edu) for any model development-related questions.
