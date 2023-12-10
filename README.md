# Approximate Kernel Density Estimation under Metric-based Local Differential Privacy

This repository contains the source code of our paper titled "Approximate Kernel Density Estimation under Metric-based Local Differential Privacy". All algorithms are implemented in Python 3.

## Datasets

-  `small_datasets/`: Provides sampled example datasets for the CodRNA, CovType, RCV1, Yelp, and SYN (default) datasets mentioned in the paper. 

-  `generate_syn_datasets.py`: Contains the code to generate a series of SYN datasets. 

## Algorithms

- `RACE.py` , `DM.py`, `PM.py`, `SW.py`, `GI.py`, `FKM_LL_RACE.py`, `FKM_LR_RACE.py` and `mLDP_KDE.py`: Store the codes for algorithms RACE, DM, PM, SW, GI, FKM-LL-RACE, FKM-LR-RACE and mLDP-KDE, respectively.
- `examples.py`: Provides an example of the mLDP-KDE algorithm with 1000 construction points and 1 query point.

## Experiments

We have provided python scripts to run all experiments.

- `test_epsilon_utility_efficiency.py`: Contains the code for **Expt1 - Utility vs. Privacy** and **Expt2 -  Time Efficiency**.
- `test_epsilon_sketchsize_communication.py` and `test_sketchsize_MSE.py`: Contain the codes for **Expt3 - Sketch Size and Communication Cost**.
- `test_m_utility_efficiency.py` and `test_n_utility_efficiency.py`: Contain the codes for **Expt4 - Scalability**.
- `test_epsilon_utility_L1.py` and `test_epsilon_utility_Angular.py`: Contain the codes for **Expt5 - Performance on Other LSH Kernels**.
- Reproduction of results:
  1. Run any of the experimental scipts simply by using `python scripts_name.py`.
  2. In scripts requiring datasets selection, modify the value of `selected_flag` to generate results for the chosen dataset.

## Visualization

-  `draw_plot.py`: Provided for visualization. It requires Python 3.7 (or higher versions) and Matplotlib.
