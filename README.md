# Approximate Kernel Density Estimation under Metric-based Local Differential Privacy

This repository contains the source code corresponding to our paper titled "Approximate Kernel Density Estimation under Metric-based Local Differential Privacy". All algorithms are implemented in Python 3.

## Datasets

-  `small_datasets/`: Provides sampled example datasets for the CodRNA, CovType, RCV1, Yelp, and SYN (default) datasets mentioned in the paper. 

-  `generate_syn_datasets.py`: Contains code to generate a series of SYN datasets. 

## Algorithms

- `RACE.py`, `DM.py`, `PM.py`, `SW.py`, `GI.py`, `FKM_LL_RACE.py`, `FKM_LR_RACE.py` and `mLDP_KDE.py`: Store the code for algorithms RACE, DM, PM, SW, GI, FKM-LL-RACE, FKM-LR-RACE and mLDP-KDE, respectively.
- `examples.py`: Provides an example of the mLDP-KDE algorithm with 1000 construction points and 1 query point.

## Experiments

Python scripts are provided to run all experiments.

- `test_epsilon_utility_efficiency.py`: Contains code for **Expt1 - Utility vs. Privacy** and **Expt2 -  Time Efficiency**.
- `test_epsilon_sketchsize_communication.py` and `test_sketchsize_MSE.py`: Contain code for **Expt3 - Sketch Size and Communication Cost**.
- `test_m_utility_efficiency.py` and `test_n_utility_efficiency.py`: Contain code for **Expt4 - Scalability**.
- `test_epsilon_utility_L1.py` and `test_epsilon_utility_Angular.py`: Contain code for **Expt5 - Performance on Other LSH Kernels**.
- To reproduce results:
  1. Run any of the experimental scripts simply by using `python scripts_name.py`.
  2. In scripts requiring dataset selection, modify the value of `selected_flag` to generate results for the chosen dataset.
  3. In scripts requiring privacy radius $r$ selection, modify the value of `nearest_flag` and `L_R_set` to generate results for the chosen $r$.

## Visualization

-  `plotting_tools.py`: Provided for visualization. It requires Python 3.7 (or higher versions) and Matplotlib.





## Updates

### [Apr 2024]

- **Additional Experiments**: 
  - `test_small_range_epsilon_utility.py`: Contains code to test the utility of the RACE, GI and mLDP-KDE algorithms using a smaller $\varepsilon$ range: [1, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20].
  - `test_r_effect.py`: Contains code to evaluate the impact of a range of privacy radius $r$ on the mLDP-KDE algorithm. We tested two classes of $r$:
    - average distance from a point to its t-nearest neighbors for t ∈ {1, 10, 100, 1000, 10000};
    - maximum distance from a point to its t-nearest neighbors for t ∈ {1, 10, 100, 1000, 10000};
  - `heatmap_visualization.py`: Contains code to visualizing 2D Heatmaps for KDE on each dataset.
- **Result Presentation**: The results of additional experiments are presented in `exp-rebuttal.pdf`.
- Update `mLDP_KDE.py`: Added comments to enhance code clarity.
- Update `parameters.py`: Added a set of parameters for new experiments.
- Update `plotting_tools.py`: Added functions `draw_heatmap` and  `draw_small_range_epsilon_MSE`  for new experiments.