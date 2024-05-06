## Overview

This repo examines the performance of **TSK fuzzy models** in regression and classification tasks.

To explore the behaviours of fuzzy regression & fuzzy classification in various conditions, both are applied on two types of datasets:
- a simple dataset &
- a high-dimensional dataset

The project is structured following this rationale.

The implementation is in **Matlab** (R2019a) using the **Fuzzy Logic Toolbox**.

## Data

Datasets for Regression:
- **Simple:** Airfoil Self-Noise Dataset - https://archive.ics.uci.edu/dataset/291/airfoil+self+noise
- **High-dimensional:** Superconductivty Dataset - https://archive.ics.uci.edu/dataset/464/superconductivty+data

Datasets for Classification:
- **Simple:** Avila Dataset - https://archive.ics.uci.edu/dataset/459/avila
- **High-dimensional:** Isolet Dataset - https://archive.ics.uci.edu/dataset/54/isolet 

All the above datasets are available through the UCI repository (https://archive.ics.uci.edu/)

## Project Structure
```
| - 1.fuzzy_regression/
| - - airfoil_self_noise_dataset/
| - - - - airfoil_self_noise.dat
| - - - - src/
| - - - - - - Regression_TSK_model_1.m
| - - - - - - Regression_TSK_model_2.m
| - - - - - - Regression_TSK_model_3.m
| - - - - - - Regression_TSK_model_4.m
| - - - - - - split_scale.m
| - - - - - - plotMFs.m
| - - - - - - fis.fis
| - - - - evaluation_metrics_plots/
| - - superconduct_dataset/
| - - - - superconduct.csv
| - - - - src/
| - - - - - - plotErrorVsNumOfRules.m
| - - - - - - plotMFs.m
| - - - - - - Regression_Grid_Search.m
| - - - - - - split_80_20.m
| - - - - evaluation_metrics_plots/
| - - - - ranks/
| - 2.fuzzy_classification/
| - - avila_dataset/
| - - - - avila.txt
| - - - - src/
| - - - - - - Classification_TSK_classDependent_bigRadius.m
| - - - - - - Classification_TSK_classDependent_smallRadius.m
| - - - - - - Classification_TSK_classIndependent_bigRadius.m
| - - - - - - Classification_TSK_classIndependent_smallRadius.m
| - - - - - - plotMFs.m
| - - - - - - sug101.fis 
| - - - - evaluation_metrics_plots/
| - - isolet_dataset/
| - - - - src/
| - - - - - - Classification_Grid_Search.m
| - - - - - - plotMFs.m
| - - - - - - split_80_20_Stratified.m
| - - - - - - sug191.fis
| - - - - evaluation_metrics_plots/
```
