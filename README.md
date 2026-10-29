# BaS-TCP

Code and experiment inputs for **Leveraging Risk Landscape for Balanced Test Prioritization in Automated Driving Systems**.

## Requirements

- MATLAB R2021b with Statistics and Machine Learning Toolbox, Global Optimization Toolbox, and Deep Learning Toolbox (`ind2vec`).
- Python 3.12. Install the Python dependencies with `python -m pip install -r requirements.txt`.

## Layout

- `matlab/algorithm`: RaDisP, balanced greedy prioritization, and the continuous group-sparse solver.
- `matlab/baselines`: prioritization baselines and permutation operators.
- `matlab/representations`: PCA, UMAP, VAE, LLE, and the supervised ISA adaptation.
- `python`: MO-TCP, score/statistics summaries, and curve-data export.
- `data`: the three numerical experiment inputs.
- `results`: saved orderings, representations, continuous solutions, and timing measurements for the reported comparisons.
- `outputs`: generated runs and summaries (not tracked).

## Data

Each `data/Dataset*.mat` contains `X` (scenarios by features), `y` (unsafe indicator), `costs` (evaluation costs), and `ranking_costs` (prioritization inputs). Dataset1 contains 10,000 rows and 63 features, Dataset2 contains 1,178 rows and 16 features, and Dataset3 contains 110 rows and 28 features. Dataset1 uses rows 3001–13000 of the released filtered feature table. The ten-feature Dataset1 view retains columns 15–24.

Dataset1 and Dataset2 reuse public data. Dataset3 is author generated. The datasets are also available at [IEEE DataPort](https://doi.org/10.21227/0z2j-7d95).

Dataset1 and Dataset2 use trapezoidal APFDc. Dataset3 uses APFD with unit evaluation costs. Dataset3's supplied ranking-cost vector is retained for the methods that consume it, while Greedy-TCP and GA-TCP use unit costs in this dataset.

## Configuration

BaS-TCP uses `k = ceil(d/2)`, `l = min(10,k,rank(X-mean(X)))`, `alpha = 0.75`, and `eta = beta = 1`. `default_config.m` defines this rule. `baseline_config.m` defines the baseline settings, including RL-static's `H = 2d` and five-fold cross-fitting. GA-TCP and MO-TCP use a population of 100 and 4,000 generations. SPECTRE*, RL-static*, and LEAST-static* denote the implemented adaptations.

The representation settings are PCA with five components, UMAP with 10 neighbors, 50 epochs, minimum distance 0.1 and four output dimensions, LLE with 12 neighbors and regularization 1000, and the VAE and ISA settings in `run_components.m`.

## Summarize the saved experiments

```bash
python python/summarize.py
python python/export_curves.py
```

The CSV summaries are written to `outputs/summary`. Statistical comparisons use two-sided one-sample t tests against the fixed BaS-TCP score, sample standard deviations, 95% difference intervals, and Holm adjustment within each dataset and feature view.

To draw the APFD(c) curves in MATLAB:

```matlab
setup;
plot_curves;
```

## Run experiments

Run commands from the repository root. Existing files in `outputs` are reused. Choose methods, datasets, and seeds explicitly to run a subset.

```matlab
setup;
run_effectiveness;                        % MATLAB main comparison, seeds 1–30
run_components;                           % representation and ordering comparisons
run_timing;                               % four-thread phase and Greedy-TCP timing
run_sparsity;                             % continuous group sparsity and hard-budget comparison
```

MO-TCP uses the Python NSGA-II compatibility implementation:

```bash
python python/run_mo.py --datasets 1 2 3
```

Subset examples:

```matlab
setup;
run_effectiveness(3, {'bas','greedy','rl'}, 1:30);
run_components(3, {'pca','lle'}, 1:30);
run_effectiveness(1, {'bas','greedy','art','cluster','spectre','rl','least','ga','random','strap','original'}, 1:30, 'feature10');
```

```bash
python python/run_mo.py --datasets 1 --view feature10
python python/summarize.py --source outputs
python python/export_curves.py --source outputs
```

`run_sparsity.m` selects the continuous penalty using support counts and numerical feasibility. It scores the selected native solution and a separate fixed-support refit. Labels are not used to select the penalty.

Recorded timing depends on hardware and numerical libraries. Rerun `run_timing` to measure the local environment.

PCA and mySVD include the implementation by Deng Cai.
