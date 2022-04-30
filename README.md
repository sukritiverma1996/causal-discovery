# Causal Discovery from Observable Data (CMU, 10-708 course project)
*Sukriti Verma, Priyam Tejaswin*

## Required packages:
```
1. numpy
2. cdt https://fentechsolutions.github.io/CausalDiscoveryToolbox
3. deap
4. rpy2
5. notears https://github.com/xunzheng/notears/releases/tag/v2.3
6. ges https://github.com/juangamella/ges
```

**Notes**:
1. Some algorithms for `cdt` will require R-libraries as well as PyTorch for continuous optimization.
2. Iterative search algorithms like `GES` and `GA` will take about 12 hours to complete.
3. `NOTEARS` and `SAM` will require GPU hardware. They have not been tested on CPU hardware.

## Motivation analysis
**GES**
* Install `ges` and required packages.
* Run `python runges.py generated_graph_scorebased_noheader.csv`
* Result will be saved under `est_ges.npy`

**NOTEARS**
* Install `NOTEARS` using the Github release.
* Ensure you have the cli binaries installed. Check `which notears_linear`.
* Run `notears_linear generated_graph_scorebased_noheader.csv`.
* Result will be saved under `W_est.csv`. Rename this to `est_notears_linear.csv`.

**NOTEARS-MLP**
* Install `NOTEARS` using the Github release.
* Ensure you have the cli binaries installed. Check `which notears_nonlinear`.
* Run `notears_nonlinear generated_graph_scorebased_noheader.csv`.
* Result will be saved under `W_est.csv`. Rename this to `est_notears_nonlinear.csv`.

**SAM v1, v2**
* Install `cdt` and all dependencies (inlcuding PyTorch with CUDA support).
* Run `python runsam.py generated_graph_scorebased_data.csv`
* Results will be saved under `est_sam_endtoend.pkl` and `est_sammod_endtoend.pkl`.
* `v1` had some bugs in the implementation which caused it to return a uniform adjacency distribution. `v2` is the improved algorithm.

**Metrics**
* Ensure you have `sklearn` installed for metric computation.
* Run `python metrics_scorebased.py` to see metrics for all score-based algorithms.
* Code used is from the `cdt` source: <https://fentechsolutions.github.io/CausalDiscoveryToolbox/html/_modules/cdt/metrics.html#precision_recall>

## Proposed (GA) model, evaluation and results
Run the cells in .ipynb in order. Data has been provided in csv files.
