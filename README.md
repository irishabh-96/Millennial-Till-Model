[![DOI](https://zenodo.org/badge/1052224089.svg)](https://doi.org/10.5281/zenodo.20207401)

This repository hosts the **Millennial-Till** model, designed to explicitly capture the effects of conventional tillage (CT) and no-tillage (NT) on soil organic carbon (SOC) pools and dynamics. The model integrates soil carbon cycling with soil water balance processes to simulate SOC dynamics under different climatic conditions, soil properties, and crop productivity levels.

Repository Structure

1. Millennial-Till/

Contains the Millennial-Till model implementation for eight representative sites across climatic and crop productivity gradients: POI, ZAN, PGL, BC, JIN, DOU, LOU, and AF. 
Soil_Carbon_Model.py – Model script containing soil carbon cycling equations, soil water balance processes, and their interactions.
Execute_All_Sites.py – Defines site-specific inputs, calibrated parameters, and simulation settings for all eight representative sites.
Execution File.ipynb – Jupyter notebook used to execute the unified model simulations.

2. CENTURY/

Contains the CENTURY model implementation used in EPIC to evaluate SOC dynamics across the same eight representative sites.
Century_APEX.py – Implements the CENTURY soil carbon model, including active, slow, and passive SOC pools, litter decomposition, soil water balance, and tillage-related processes.
Execution File.ipynb – Jupyter notebook used to execute CENTURY model simulations.

3. Sensitivity-Analysis/

Contains the Millennial-Till sensitivity analysis framework used to examine how changes in climatic conditions, net primary productivity (NPP), and soil properties influence simulated SOC dynamics. 

Soil_Carbon_Model.py – Contains the coupled soil carbon and water balance equations used in the sensitivity simulations.
Execute_All_Sites.py – Defines site-specific parameters and executes sensitivity analysis scenarios across the eight representative sites.
Execution File.ipynb – Jupyter notebook used to execute sensitivity simulations.

4. Carbon-Tracking-and-Half-Life/

Contains the framework for tracking the fate of carbon entering the soil under CT and NT practices by estimating carbon retention and half-life.

Model.py – Contains the model equations used for carbon tracking simulations.
Execution_Source.ipynb – Executes simulations used to establish the initial model conditions for carbon tracking.
Carbon tracking and Half-life.ipynb – Analyzes the redistribution and retention of newly introduced carbon across SOC pools and its losses through respiration and leaching.
sites.json – Stores site-specific parameters used in the simulations.
End_States/ – Contains saved model states used to initialize carbon tracking simulations.
Results_SourceFlux/ – Contains simulation outputs from the carbon tracking analysis.

5. Analysis and Visualization Notebooks

Carbon Pools and fluxes PDF.ipynb – Visualizes temporal changes in aggregate carbon (AAC), particulate organic carbon (POC), microbial biomass carbon (MBC) under CT and NT practices.
Observed vs Simulated change in SOC (Millennial-Till).ipynb – Evaluates observed and simulated changes in SOC (SOC_NT − SOC_CT) across the eight representative sites.
Porosity, Saturated Hydraulic Conductivity - Millennial Till.ipynb – Analyzes simulated soil porosity and saturated hydraulic conductivity (Ksat) within the Millennial-Till framework.

6. Contact

For questions related to model development, implementation, or simulations, contact:

Rishabh Singh
irishabh1996@tamu.edu
