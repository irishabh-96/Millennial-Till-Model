This repository hosts the **Millennial-Till** model, designed to explicitly capture the effects of conventional tillage (CT) and no-tillage (NT) on soil organic carbon (SOC) pools and dynamics.

Repository Structure

**Studies**/
Contains site-specific parameterizations for eight representative sites across climatic and crop productivity gradients: **POI, ZAN, PGL, BC, JIN, DOU, LOU, and AF**. Each site folder includes:

**Soil Carbon Model.py** – Core simulation script integrating soil water balance processes with carbon cycling. Includes parameter sets for different soil textures and climatic conditions.

**Execution File.ipynb** – Jupyter notebook used to run model simulations.

**StudyName.csv** – Output file (e.g., POI.csv) containing temporal dynamics of SOC pools and fluxes for the modeled period.

**Carbon Pools and Fluxes PDF.ipynb**
Notebook visualizing time-series of aggregate carbon, particulate organic matter (POM), and microbial biomass (MB), along with fluxes such as POM aggregation (Fpa), aggregate breakdown (Fa), and microbial DOC uptake (Flb).

**Comparison between CT and NT.ipynb**
Notebook comparing SOC pool and flux responses during transitions from CT to NT.

**Observed vs Simulated change in SOC and the impact on C pools.ipynb**
Notebook evaluating observed vs simulated SOC changes (SOC_NT – SOC_CT) across all median sites, with a breakdown of contributions from individual carbon pools.

Contact Rishabh Singh (irishabh1996@tamu.edu) for any model development-related questions.
