# Battery System Identification

This is a project that I've been researching for some time now. The motivation is to understand the phenomenon of lithium-ion battery (LiB) degradation through the lens of nonlinear parameter estimation. Everyone knows that LiBs degrade over time from their experience with cellphones and research throughout the past decades has revealed many insights into the various mechanisms involved (see [^1] for a recent review). While many papers propose a model and estimate degradation parameters based on experimental data, the literature is missing a systematic validation of these proposed models across large-scale datasets.

The goal of this repository is to address that gap by:
1. Implementing a numerically-stable ODE-constrained optimization algorithm that scales to multi-GB datasets
2. Fitting a variety of (both equivalent circuit-based and physics-based) models to a large experimental dataset to validate predictive capability

Ultimately, I am motivated by power system modeling objectives: as the electrical grid adds substantial capacity in the form of battery energy storage systems (BESSs), accurate battery submodels (including the affects of degradation) will become increasingly important to inform capacity expansion and (security-contrained) production-cost models.

## Algorithm Selection

I will begin with a custom implementation of the multiple-shooting method proposed by Bock [^2]. This algorithm was selected because it solves a quite general formulation of the parameter estimation problem and provides a rigorous treatment of the solution method's numerical stability. More detailed notes on the mathematical formulation can be found in [`docs/math_notes.md`](docs/math_notes.md).

## Dataset Selection

For initial results, I am using the dataset shared by Stroebl, et al. [^3]. This dataset was selected because of its BESS application focus and advanced experimental design techniques (although model-based online experimentation techniques like those employed by Attia et al. [^4] remain of interest to me).  After fitting to this initial dataset, I plan to collate other high-quality datasets to build a model which is validated across chemistries and a range of operating regimes.

## Past work

I have previously analyzed this dataset using the Python scripts provided alongside [^3] to extract the battery capacity as a function of full equivalent cycles (FEC) and operational parameters (like depth of discharge and temperature). I created some Tableau [visualizations](https://public.tableau.com/app/profile/sean.jennings1617/viz/LiDeg--Story/Story1) to understand degradation trends, but also found some limitations of this "feature extraction" approach.

The simple [scripts](https://github.com/fst2112/Multi-Stage-Lithium-Ion-Battery-Aging-Dataset-Analysis/blob/main/src/feature_extraction.py) estimate capacity taking the integral of current (dis)charged between a minimum and maximum voltage. But this method of estimating capacity does not account for the slow timescale effects of diffusion within electrode particles. This is evidenced by capacity measurements which defy expectations by as much as 20%. This motivates the current approach of fitting a single multi-timescale model across the entire dataset.

# References
[^1]: Edge, J. S. et al. Lithium ion battery degradation: what you need to know. Phys. Chem. Chem. Phys. 23, 8200–8221 (2021).
[^2]: Bock, H. G., Kostina, E. & Schlöder, J. P. Direct Multiple Shooting and Generalized Gauss-Newton Method for Parameter Estimation Problems in ODE Models. in Multiple Shooting and Time Domain Decomposition Methods (eds Carraro, T., Geiger, M., Körkel, S. & Rannacher, R.) 1–34 (Springer International Publishing, Cham, 2015). doi:10.1007/978-3-319-23321-5_1.
[^3]: Stroebl, F. et al. A multi-stage lithium-ion battery aging dataset using various experimental design methodologies. Sci Data 11, 1020 (2024).
[^4]: Attia, P. M. et al. Closed-loop optimization of fast-charging protocols for batteries with machine learning. Nature 578, 397–402 (2020).
