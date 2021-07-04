# Indirect Cross Validation for KDE Bandwidth Optimization
This is a Python implementation of the Indirect Cross Validation (ICV) method of [(Savchuk2010)](https://www.tandfonline.com/doi/abs/10.1198/jasa.2010.tm08532) for bandwidth selection in kernel density estimation problems using a Gaussian kernel. Currently only 2-D KDE problems are supported with bivariate Gaussian kernels.

The idea behind ICV is to identify the bandwidth which minimizes the least-squares cross validation error of a specialized kernel L, and then re-scale this value to obtain an estimate of the optimal Gaussian kernel bandwidth. See [(Savchuk2010)](https://www.tandfonline.com/doi/abs/10.1198/jasa.2010.tm08532) for details. The L-kernel depends on two parameters (α, σ), and the script provides two automatic methods for setting these based on the sample size. The default is to use the “reference rule” proposed in (Savchuk2010), and an alternative is the adaptive rule defined in [(Savchuk2008)](http://arxiv.org/abs/0812.0052). The user may also specify (α, σ) directly, which may be useful for comparing results between different problems.

Current goals are to (1) expand this to arbitrary N-dimensional problems and (2) develop optimizations/approximations for handling larger data sets.

# References
[1] Savchuk, O. Y., Hart, J. D., & Sheather, S. J. (2010).
    Indirect cross-validation for density estimation. 
    Journal of the American Statistical Association,
    105(489), 415–423.
    https://doi.org/10.1198/jasa.2010.tm08532

[2] Savchuk, O. Y., Hart, J. D., & Sheather, S. J. (2008).
    An Empirical Study of Indirect. 1, 1–22.
    http://arxiv.org/abs/0812.0052
