#Script: bivariateICV.py
#Author: Grant Shoffner
#Date: 2021-07-02
#Contact: gshoffner AT mbi DOT ucla DOT edu
#Description: This is a Python implementation of the 
#Indirect Cross Validation (ICV) method of [1] for bandwidth
#selection in 2-dimensional kernel density estimation 
#problems using a bivariate Gaussian kernel.
#
#References:
#
#   [1] Savchuk, O. Y., Hart, J. D., & Sheather, S. J. (2010).
#       Indirect cross-validation for density estimation. 
#       Journal of the American Statistical Association,
#       105(489), 415–423.
#       https://doi.org/10.1198/jasa.2010.tm08532
#
#   [2] Savchuk, O. Y., Hart, J. D., & Sheather, S. J. (2008).
#       An Empirical Study of Indirect. 1, 1–22.
#       http://arxiv.org/abs/0812.0052

import sys
import numpy as np
from scipy.stats import multivariate_normal


#Shorthand for the normal PDF since this gets called frequently
N = lambda *args, **kwargs : multivariate_normal.pdf(*args, **kwargs)

ICV_PARAM_SELECTION_METHODS = ['auto' , 'Savchuk2008']


class BivariateICV:
    """Base class for bivariate indirect cross validation.

    Don't use directly, use IndirectCrossValidation or 
    IndirectKFoldCrossValidation instead.
    """
    
    def __init__(self, data, params = 'auto'):

        #Verify data shape
        try:
            assert type(data) == np.ndarray
            assert len(data.shape) == 2
            assert data.shape[1] == 2
        except AssertionError:
            sys.exit(
                "Error: data must be an ndarray with shape (n_samples, 2)."
                )

        self.data = data
        
        #Verify params
        self.set_params(params)


        #Cache the RL value
        self._RL = self.RL(self.params)

    def set_params(self, params):
        """Verifies selection method and stores (alpha,sigma)

        Can be used to reset (alpha, sigma) to custom values
        after instantiation of an ICV class.
        """

        param_error_msg = "Error: supplied params must be a 2-tuple (alpha, sigma)."
        if type(params) == str:

            if params not in ICV_PARAM_SELECTION_METHODS:
                sys.exit("Error: unrecognized param selection sting %s given." \
                        % params)

            elif params == 'auto':
                self.params = \
                    (2.42, max(5.06, 0.149 * self.data.shape[0]**(3./8.)))

            elif params == 'Savchuk2008':
                    self.params = self._Savchuk2008(self.data.shape[0])

        elif type(params) == tuple:
            if len(params) != 2:
                sys.exit(param_error_msg)
            else:
                self.params = np.array(params)

        elif type(params) == np.ndarray:
            if params.flatten().shape != (2,):
                sys.exit(param_error_msg)
            else:
                self.params = params.flatten()

        else:
            sys.exit("Error: `params` value not recognized. Either specify a method string or a 2-tuple (alpha, sigma)")
        
    @staticmethod
    def _Savchuk2008(N_samples):
        """Calculates the alpha,sigma params based on the
        formulas giving in [Savchuk2008]."""
        
        lNsamp = np.log10(N_samples)

        alpha_mod = 10.0**(
                3.39-1.093*lNsamp + 0.025*lNsamp**3 - 0.00004*lNsamp**6.0
            )

        sigma_mod = 10.0**(
                -0.58 + 0.386*lNsamp - 0.012*lNsamp**2.0
            )

        return alpha_mod, sigma_mod

    @staticmethod
    def RL(params):
        """Operator `R(g)` from [Savchuk2010] applied to the L
        kernel and evaluated at params = (alpha,sigma)."""
        
        α, σ = params

        value = (1.0/np.pi)*(((1.0+α)**2.0)/4.0 \
                - α*(1.0+α)*σ/(1.0+σ**2.0)\
                + (α**2.0)/4.0)

        return value

    def _L_Kernel(self, points):
        """Evaluates the 2-dimensional L-kernel."""
        α, σ = self.params
        L = (1.0 + α) * N(points, mean = np.zeros(2))
        L -= (α/σ) * N(points/σ, mean = np.zeros(2))
        return L
    
    def _LSCV(self, b, kde_pts, trial_pts):
        """Evaluates a single ICV step on the L-kernel bandwidth `b`.

        This function implements the sumations shown in 
        equation (2) in [Savchuk2010] for the L-kernel. Note
        that the first term in the equation (1/nh)R(L) is 
        left out so that iterative calls to _LSCV over
        successive values of `j` (the index left out data
        point) do not repeatedly add this first term. This 
        means _LSCV should be invoked as:

        SUM = (1/nh)R(L)
        for OUT_PT in PTS:
            SUM += _LSCV(b, IN_PTS, OUT_PT)

        The awkward outer-product call to substract below is
        for compatibility with multiple left-out data, ie in
        K-fold cross validation.     
        """

        n = float(self.data.shape[0])

        #Not sure if using Unicode symbols in code is exactly
        #kosher, but it greatly simplifies and clarifies the
        #equations below.
        α, σ = self.params
        
        #Take the difference between the kernel means `kde_pts`
        #the left-out points `trial_pts`
        diff = np.concatenate(
            [np.subtract.outer(kde_pts[:,i], trial_pts[:,i]).reshape(-1,1)\
                for i in range(kde_pts.shape[1])],
            axis = 1
        )

        #Rescale by the bandwidth `b`
        diff /= b

        I = np.eye(2)
        mu = np.zeros(2)

        #Second term in Equation (2)
        SUM = (1.0/(b * n**2.0)) * np.sum(
            ((1.0+α)**2.0) * N(diff, mean=mu, cov = 2.0*I) \
            - 2.0*σ*α*(1.0+α) * N(diff, mean=mu, cov = (1.0+σ**2.0)*I) \
            + (σ**2.0)*(α**2.0) * N(diff, mean=mu, cov = 2.0*(σ**2.0)*I)
        )

        #Third term in Equation (2)
        SUM -= (2.0/(n*(n-1.0)*b)) * np.sum(self._L_Kernel(diff))

        return SUM

    def bUCV_to_hUCV_factor(self):
        """Calculates the multipicative factor required for
        converting the L-kernel bandwidth bUCV to the
        Gaussian kernel bandwidth hUCV."""

        α, σ = self.params

        C = (((4.0*(1.0 + α) - 2.0*α*(σ**3.0))**2.0)/(64.0*np.pi*self._RL))**0.2

        return C

    def evaluate(self, b):
        """Evaluates Full ICV score on the L-kernel bandwidth `b`."""
        return self.__call__(b)


class IndirectCrossValidation(BivariateICV):
    """Evaluate 2-D KDE bandwidths by Indirect Cross Validation.

    Performs the ICV protocol of [Savchuk2010] to score
    a proposed bandwidth `b` using the L-kernel
    over leave-one-out cross validation. Currently only set
    up for 2-D bivariate data.

    This scales as O(N_samples**2) so it will struggle on
    large datasets.

    Parameters
    ----------
    data : np.ndarray
        Array of shape (N_samples, 2)

    params : {'auto' , 'Savchuk2008'} or tuple(alpha, sigma)
                default = 'auto'
        `params` is either a string specifying the automatic
        selection method to use for setting the parameters
        ALPHA and SIGMA, or a tuple (alpha, sigma) containing
        user defined values for these parameters.

        `auto` uses the reference rule from [Savchuk2010]
        (alpha, sigma) = (2.42, max(5.06, 0.149*N_samples**(3/8)))

        `Savchuk2008` uses the adaptive rule proposed in
        [Savchuk2008] for (alpha_mod, sigma_mod).

        Note that these two methods can produce very different
        values for alpha, sigma.

    Examples
    --------
    This example uses multiprocessing to speed up ICV over
    a range of possible bandwidths.
    >>> import numpy as np
    >>> import multiprocessing as mp
    >>> from bivariateICV import IndirectCrossValidation
    >>> from scipy.stats import multivariate_normal as MVN
    >>> from matplotlib import pyplot
    >>> sample = MVN.rvs(mean = np.zeros(2), size = 300)
    >>> ICV = IndirectCrossValidation(sample)
    >>> pool = mp.Pool(None)
    >>> b_grid = np.linspace(0.002,0.15,100)
    >>> scores = pool.map(ICV, b_grid)
    >>> b_hat_UCV = b_grid[np.argmin(scores)]
    >>> print(f"Optimal L-kernel bandwidth b_hat_UCV = {b_hat_UCV}")
    
    The value b_hat_UCV is the optimal LSCV bandwidth for
    the L-kernel. We need to convert this to the optimal
    bandwidth h_hat_UCV for the Gaussian kernel.
    >>> C = ICV.bUCV_to_hUCV_factor()
    >>> h_hat_UCV = C * b_hat_UCV
    >>> print(f"Optimal Gaussian bandwidth h_hat_UCV = {h_hat_UCV}")

    Here we plot the scores to visually check that we are in
    the range of the smallest minimum. If the minimum occurs
    in the tail of the plot, expand the `b_grid` bounds above
    >>> pyplot.plot(b_grid, scores, '-')
    >>> pyplot.suptitle("LSCV score vs L-kernel bandwidth")
    >>> pyplot.title(
    ...  f"Parameters: n={ICV.data.shape[0]}, α={ICV.params[0]:.3f}, σ={ICV.params[1]:.3f}")
    >>> pyplot.xlabel("Bandwidth (b)")
    >>> pyplot.ylabel("LSCV score")
    >>> pyplot.plot(np.tile(b_hat_UCV, 10), np.linspace(min(scores), 
    ...     max(scores)-0.3, 10), 'k--', linewidth = 1)
    >>> pyplot.annotate(f"b̂UCV = {b_hat_UCV:0.3f}\nĥUCV = {C:0.3f}⋅b̂UCV = {h_hat_UCV:0.3f}",
    ...     (b_hat_UCV, max(scores)-0.3))
    >>> pyplot.grid(linestyle = '-.', linewidth = 0.5)
    >>> pyplot.show()

    We can also plot the L-kernel for the current values
    of (alpha, sigma). Since the 2-D kernel is symmetric
    about the origin, we just plot a slice through the
    y = 0 plane.
    >>> support = np.linspace(-10,10,200)
    >>> support2d = np.concatenate(
    ...     (support.reshape(-1,1), np.zeros(200).reshape(-1,1)),
    ...     axis = 1)
    >>> kernel = ICV._L_Kernel(support2d)
    >>> pyplot.plot(support, kernel, '-')
    >>> pyplot.title(
    ...  f"L(u, α={ICV.params[0]:.3f}, σ={ICV.params[1]:.3f})")
    >>> pyplot.xlabel('u')
    >>> pyplot.ylabel('L(u,α,σ)')
    >>> pyplot.grid(linestyle = '-.', linewidth = 0.5)
    >>> pyplot.tight_layout()
    >>> pyplot.show()
    """

    def __init__(self, data, params = 'auto'):
        super().__init__(data, params)

    def __call__(self, b):
        """Evaluates Full ICV score on the L-kernel bandwidth `b`."""
        
        n = self.data.shape[0]

        #Initiallize the LSCV score with the first term in
        #equation (2) in [Savchuk2010]
        LOO_CV_score = (1.0/(n*b)) * self._RL
        
        for missing_idx in np.arange(n):
        
            LOOarray = np.concatenate(
                        (self.data[:missing_idx],
                         self.data[missing_idx+1:]),
                        axis = 0,
                    )
            
            trial_pt = self.data[missing_idx].reshape(1,2)
            
            LOO_CV_score += self._LSCV(b, LOOarray, trial_pt)

        return LOO_CV_score


class IndirectKFoldCrossValidation(BivariateICV):
    """Evaluate 2-D KDE bandwidths by K-fold Indirect CV.

    Performs the ICV protocol of [Savchuk2010] to score
    a proposed bandwidth `b` using the L-kernel over K-fold
    cross validation. Currently only set up for 2-D 
    bivariate data.

    K-fold CV scales as O(((K-1)/K)N**2) so it offers a
    small performance advantage over Leave-One-Out CV for
    large data sets when K is small.

    Note that K-fold CV is not addressed in [Savchuk2010]
    so its use here may not strictly follow their theory.
    Testing on example data has shown that bandwidths
    estimated by K-fold CV always overestimate the h_hat_UCV
    obtained from LOO-CV by a relatively small margin. So 
    the error from K-fold estimation is on the conservative 
    side.
    

    Parameters
    ----------
    data : np.ndarray
        Array of shape (N_samples, 2)

    K : int, default = 5
        Subdivide the data into K portions for K-fold CV.
        If K is not a divisor of N_samples the final portion
        will be truncated. For large data sets this will not
        have a material impact on the result. For small data
        use IndirectCrossValidation instead.

    params : {'auto' , 'Savchuk2008'} or tuple(alpha, sigma)
                default = 'auto'
        `params` is either a string specifying the automatic
        selection method to use for setting the parameters
        ALPHA and SIGMA, or a tuple (alpha, sigma) containing
        user defined values for these parameters.

        `auto` uses the reference rule from [Savchuk2010]
        (alpha, sigma) = (2.42, max(5.06, 0.149*N_samples**(3/8)))

        `Savchuk2008` uses the adaptive rule proposed in
        [Savchuk2008] for (alpha_mod, sigma_mod).

        Note that these two methods can produce very different
        values for alpha, sigma.

    seed : int, default = 0
        Seed for numpy.random.default_rng for use in data
        shuffling prior to splitting. A default value is
        given here so that multiple calls to 
        IndirectKFoldCrossValidation on the same data will
        produce the same results. Set seed = None for no
        seed in the rng.

    Examples
    --------
    This example uses multiprocessing to speed up ICV over
    a range of possible bandwidths. Results from LOO and 
    K-fold are compared.
    >>> import numpy as np
    >>> import multiprocessing as mp
    >>> from bivariateICV import IndirectCrossValidation,\
    >>>        IndirectKFoldCrossValidation
    >>> from scipy.stats import multivariate_normal as MVN
    >>> from matplotlib import pyplot
    >>> sample = MVN.rvs(mean = np.zeros(2), size = 300)
    >>> ICV = IndirectCrossValidation(sample)
    >>> KICV = IndirectKFoldCrossValidation(sample)
    >>> pool = mp.Pool(None)
    >>> b_grid = np.linspace(0.002,0.15,100)
    >>> loo_scores = pool.map(ICV, b_grid)
    >>> kfold_scores = pool.map(KICV, b_grid)
    
    Plot the scores vs the L-kernel bandwidth
    >>> pyplot.plot(b_grid, loo_scores, '-')
    >>> pyplot.plot(b_grid, kfold_scores, '-')
    >>> pyplot.show()
    """
    
    def __init__(self, data, K = 5, params = 'auto', seed = 0):
        super().__init__(data.copy(), params)

        self.rng = np.random.default_rng(seed = seed)
        self.split_data = None
        self._generate_kfold_splits(K)

    def _generate_kfold_splits(self, K):
        #Shuffle the samples. Since we called data.copy()
        #in __init__ this doesn't change the original data
        self.rng.shuffle(self.data, axis = 0)
        self.split_data = np.array_split(self.data, K, axis = 0)

    def __call__(self, b):
        
        K = len(self.split_data)
        n = self.data.shape[0]

        #Initiallize the LSCV score with the first term in
        #equation (2) in [Savchuk2010]
        KFold_CV_score = (1.0/(n*b)) * self._RL

        for missing_idx in np.arange(K):
            KFoldarray = np.concatenate(
                    (*self.split_data[:missing_idx], 
                        *self.split_data[missing_idx+1:]),
                    axis = 0
                )
            KFoldarray = KFoldarray.reshape(-1,2)
    
            trial_pts = self.split_data[missing_idx]

            KFold_CV_score += self._LSCV(b, KFoldarray, trial_pts)

        return KFold_CV_score
