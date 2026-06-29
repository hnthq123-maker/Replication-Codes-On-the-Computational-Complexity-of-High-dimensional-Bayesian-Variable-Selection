# Replication Codes for *On the Computational Complexity of High-Dimensional Bayesian Variable Selection*

This repository provides an independent Python reproduction of selected numerical experiments from the paper:

> Yang, Y., Wainwright, M. J., & Jordan, M. I. (2016).  
> *On the Computational Complexity of High-Dimensional Bayesian Variable Selection*.  
> **The Annals of Statistics**, 44(6), 2497--2532.

The paper studies an important question in high-dimensional Bayesian variable selection: even when the posterior distribution has good statistical properties, does a local MCMC algorithm necessarily mix rapidly? This repository aims to reproduce the main computational phenomena discussed in the paper, including MCMC trajectories, mixing behavior under different signal-to-noise regimes, and the comparison between Bayesian variable selection and Lasso.

## Project Overview

The reproduction focuses on three groups of experiments:

1. **Figure 1-style MCMC trajectories**  
   The code simulates high-dimensional Bayesian variable selection under independent Gaussian designs and plots the evolution of the unnormalized log posterior along multiple Metropolis--Hastings chains.

2. **Table 1--2-style mixing experiments**  
   The code compares independent and correlated Gaussian designs under several signal-to-noise ratio settings. Mixing is evaluated using the Gelman--Rubin diagnostic.

3. **Figure 2-style BVS versus Lasso comparison**  
   The code compares Bayesian variable selection with cross-validated Lasso in a correlated-design setting where Lasso may fail due to violation of the irrepresentable condition.

## Important Note

This is an independent approximate reproduction, not the original code of Yang, Wainwright, and Jordan. Some implementation details in the paper, such as exact random seeds, certain MCMC settings, and the sparsity-prior exponent used in every experiment, are not fully specified. Therefore, the numerical values produced by this repository may not exactly match the paper. The main goal is to reproduce the qualitative conclusions and computational patterns.

## Repository Structure

```text
.
├── reproduce3.py
├── reproduce3_output/
│   ├── figure1_snr1.png
│   ├── figure1_snr3.png
│   ├── figure2_bvs_vs_lasso.png
│   ├── figure2_log_posterior_differences.csv
│   ├── paper_mixing_raw.csv
│   ├── paper_mixing_summary.csv
│   ├── paper_mixing_summary.xlsx
│   ├── quick_test.csv
│   └── run_summary.json
└── README.md
```

The main implementation is contained in `reproduce3.py`. The folder `reproduce3_output/` stores generated figures, tables, and summary files.

## Requirements

The code was written in Python and uses the following packages:

```text
numpy
pandas
matplotlib
scikit-learn
openpyxl
```

You can install the required packages using:

```bash
pip install numpy pandas matplotlib scikit-learn openpyxl
```

## Usage

All experiments can be run from the command line.

### Quick test

```bash
python reproduce3.py --mode quick
```

This runs a small test experiment and generates:

```text
reproduce3_output/quick_test.csv
```

### Reproduce Figure 1-style MCMC trajectories

```bash
python reproduce3.py --mode figure1
```

This generates trajectory plots for independent designs under SNR = 1 and SNR = 3:

```text
reproduce3_output/figure1_snr1.png
reproduce3_output/figure1_snr3.png
```

### Reproduce Table 1--2-style mixing experiments

```bash
python reproduce3.py --mode paper-mixing
```

This generates raw and summarized mixing results:

```text
reproduce3_output/paper_mixing_raw.csv
reproduce3_output/paper_mixing_summary.csv
reproduce3_output/paper_mixing_summary.xlsx
```

The paper-scale mixing experiment can be computationally expensive because it involves multiple combinations of sample size, dimension, design type, signal strength, datasets, and MCMC chains.

### Reproduce Figure 2-style BVS versus Lasso comparison

```bash
python reproduce3.py --mode figure2
```

This generates:

```text
reproduce3_output/figure2_bvs_vs_lasso.png
reproduce3_output/figure2_log_posterior_differences.csv
```

### Run all main experiments

```bash
python reproduce3.py --mode paper-all
```

This runs the Figure 1-style trajectories, the Table 1--2-style mixing experiments, and the Figure 2-style BVS versus Lasso comparison.

## Optional Arguments

The script supports several command-line arguments. Some useful examples are:

```bash
python reproduce3.py --mode figure1 --seed 2026
python reproduce3.py --mode paper-mixing --mixing-replicates 20 --steps-per-p 20
python reproduce3.py --mode figure2 --figure2-replicates 100
python reproduce3.py --mode paper-all --kappa 1.0
```

Main options include:

- `--mode`: choose the experiment to run. Options are `quick`, `figure1`, `paper-mixing`, `figure2`, and `paper-all`.
- `--output-dir`: specify the output directory. The default is `reproduce3_output`.
- `--seed`: set the random seed.
- `--kappa`: set the sparsity-prior exponent.
- `--mixing-replicates`: set the number of replicated datasets in the mixing experiment.
- `--steps-per-p`: set the number of MCMC iterations as a multiple of `p`.
- `--figure1-chains`: set the number of chains for the Figure 1-style experiment.
- `--figure2-replicates`: set the number of replications for the BVS versus Lasso experiment.
- `--figure2-bvs-steps`: set the number of BVS MCMC steps in the Figure 2-style experiment.

## Main Implementation Details

The script `reproduce3.py` contains the following main components:

- `generate_design`: generates independent or correlated Gaussian design matrices.
- `paper_beta`: constructs the sparse coefficient vector used in the paper's simulations.
- `PosteriorScorer`: evaluates the unnormalized log posterior for a candidate model.
- `mh_chain`: implements the Metropolis--Hastings sampler with single-flip and double-flip moves.
- `gelman_rubin`: computes the Gelman--Rubin / R-hat convergence diagnostic.
- `make_figure1`: produces Figure 1-style MCMC trajectory plots.
- `paper_mixing_experiment`: produces Table 1--2-style mixing summaries.
- `make_figure2`: produces the BVS versus Lasso comparison.

## Statistical Background

The simulation is based on the high-dimensional linear model

```text
Y = X beta* + w,
```

where `p` can be much larger than `n`, and the goal is to recover the important covariates. In Bayesian variable selection, each candidate model is represented by a binary inclusion vector. Since there are `2^p` possible models, direct enumeration is infeasible in high dimensions, so MCMC is used to explore the posterior distribution over models.

The paper emphasizes that statistical correctness and computational efficiency are different issues. A posterior distribution may concentrate on the correct model, but a local MCMC chain may still mix slowly if the posterior geometry contains isolated modes or low-probability valleys.

## Interpretation of the Reproduction

The reproduced experiments illustrate the following main patterns:

1. **Strong signal regime**  
   When the signal is strong, the true model tends to receive high posterior probability, and the MCMC chain often moves toward high-posterior regions.

2. **Weak signal regime**  
   When the signal is weak, the sparsity penalty can dominate the likelihood contribution, and the null model may become highly competitive.

3. **Intermediate signal regime**  
   This is often the most delicate regime. The posterior may contain several competing modes, and mixing can become unstable, especially under correlated designs.

4. **Effect of the design matrix**  
   Independent designs usually lead to more stable behavior. Correlated designs can make the posterior geometry more complicated and may create harder computational problems.

5. **BVS versus Lasso**  
   In some correlated-design settings, Bayesian variable selection can perform better than cross-validated Lasso, especially when the irrepresentable condition required by Lasso is violated.

## Output Files

The main output files are:

- `figure1_snr1.png`: MCMC trajectory plot under SNR = 1.
- `figure1_snr3.png`: MCMC trajectory plot under SNR = 3.
- `paper_mixing_raw.csv`: raw results from the mixing experiment.
- `paper_mixing_summary.csv`: summarized results from the mixing experiment.
- `paper_mixing_summary.xlsx`: Excel version of the summarized mixing results.
- `figure2_bvs_vs_lasso.png`: boxplot comparing BVS and Lasso.
- `figure2_log_posterior_differences.csv`: numerical values used in the Figure 2-style comparison.
- `run_summary.json`: summary of the most recent run.

## Limitations

This repository is intended for educational and research demonstration purposes. It does not claim to be an exact reproduction of every numerical value in the original paper. Differences may arise from random seeds, implementation choices, and unspecified details in the original simulation setup.

In addition, full paper-scale experiments may require substantial computational time, especially when `p = 5000` and multiple MCMC chains are run for many replicated datasets.

## Citation

```bibtex
@article{yang2016computational,
  title={On the computational complexity of high-dimensional Bayesian variable selection},
  author={Yang, Yun and Wainwright, Martin J. and Jordan, Michael I.},
  journal={The Annals of Statistics},
  volume={44},
  number={6},
  pages={2497--2532},
  year={2016}
}
```

## Author

This reproduction was prepared by Huiqun Tan as a reading and coding project on high-dimensional Bayesian variable selection and MCMC computation.
