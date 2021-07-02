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
#
#Notes:

import sys
import numpy as np
from scipy.stats import multivariate_normal as MVN
from timeit import timeit

#Import tqdm for a progress bar if available
try:
    from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = lambda iterator, total = None : iterator


ICV_PARAM_SELECTION_METHODS = ['auto' , 'Savchuk2008']


class IndirectCrossValidation:
    """Evaluate 2-D KDE bandwidths by Indirect Cross Validation.

    Performs the ICV protocol of [Savchuk2010] to score
    a proposed bandwidth `b_trial` using the L-kernel
    over leave-one-out cross validation. Currently only set
    up for 2-D bivariate data.

    This scales as O(N_samples**2) so it will struggle on
    large datasets.

    Parameters
    ----------
    data : np.ndarray
        Array of shape (N_samples, 2)

    params : {'auto' , 'Savchuk2008'} or tuple(alpha, sigma)
                defaul = 'auto'
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

    cache_LOO_data : bool, default = True
        Set this flag to False to disable caching of the
        leave-one-out data array for cross validation.
        Caching currently appears to provide a very small
        performance advantage. For large data sets caching
        may exceed memory limits.

    Examples
    --------
    This example uses multiprocessing to speed up ICV over
    a range of possible bandwidth
    >>> import numpy as np
    >>> import multiprocessing as mp
    >>> from bivariateICV import IndirectCrossValidation
    >>> from scipy.stats import multivariate_normal as MVN
    >>> from matplotlib import pyplot
    >>> sample = MVN.rvs(mean = np.zeros(2), size = 300)
    >>> ICV = IndirectCrossValidation(sample)
    >>> pool = mp.Pool(None)
    >>> b_trial = np.linspace(0.002,0.15,100)
    >>> scores = pool.map(ICV, b_trial)
    >>> b_hat_UCV = b_trial[np.argmin(scores)]
    >>> print(f"Optimal L-kernel bandwidth b_hat_UCV = {b_hat_UCV}")
    
    Here we plot the scores to visually check that we are in
    the range of the smallest minima. If the minimum occurs
    in the tail of the plot, expand the `b_trial` bounds above
    >>> pyplot.plot(b_trial, scores, '.')
    >>> pyplot.show()

    The value b_hat_UCV is the optimal LSCV bandwidth for
    the L-kernel. We need to convert this to the optimal
    bandwidth h_hat_UCV for the Gaussian kernel.
    >>> C = ICV.bUCV_to_hUCV_factor(b_hat_UCV)
    >>> h_hat_UCV = C * b_hat_UCV
    >>> print(f"Optimal Gaussian bandwidth h_hat_UCV = {h_hat_UCV}")
    """

    def __init__(self, data, params = 'auto', cache_LOO_data = True):

        #Verify data shape
        try:
            assert type(data) == np.ndarray
            assert len(data.shape) == 2
            assert data.shape[1] == 2
        except AssertionError:
            sys.exit(
                "Error: data must have shape = (n_samples, 2)."
                )

        #Verify params
        param_error_msg = "Error: user supplied parameters must be a 2-tuple (alpha, sigma)."
        if type(params) == str:

            if params not in ICV_PARAM_SELECTION_METHODS:
                sys.exit("Error: unrecognized parameter selection sting %s specified." \
                        % params)

            elif params == 'auto':
                self.params = \
                    (2.42, max(5.06, 0.149 * data.shape[0]**(3./8.)))

            elif params == 'Savchuk2008':
                    self.params = self._Savchuk2008(data.shape[0])

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

        self.data = data

        #Cache the RL value
        self._RL = self.RL(self.params)
        
        #Flag `cache_LOO_data` can be set to false to prevent
        #caching a large array.
        assert type(cache_LOO_data) == bool
        self._LOO_data_stack = None
        self.cache_LOO_data = cache_LOO_data
            
    @staticmethod
    def _Savchuk2008(N_samples):
        """Calculates the alpha,sigma params based on the
        formulas giving in [Savchuk2008]."""
        
        lN = np.log10(N_samples)

        alpha_mod = 10.0**(
                3.39-1.093*lN + 0.025*lN**3 - 0.00004*lN**6.0
            )

        sigma_mod = 10.0**(
                -0.58 + 0.386*lN - 0.012*lN**2.0
            )

        return alpha_mod, sigma_mod

    @staticmethod
    def RL(params):
        """Operator `R` from [Savchuk2010] applied to the L
        kernel and evaluated at `params`."""
        
        alpha, sigma = params

        value = (1.0/np.pi)*(((1.0+alpha)**2.0)/4.0 \
                - alpha*(1.0+alpha)*sigma/(1.0+sigma**2.0)\
                + (alpha**2.0)/4.0)

        return value


    def _L_Kernel(self, points):
        """Evaluates the 2-dimensional L-kernel."""
        alpha, sigma = self.params
        L = (1.0 + alpha) * MVN.pdf(points, mean = np.zeros(2))
        L -= (alpha/sigma)*MVN.pdf(points/sigma, mean = np.zeros(2))
        return L

    def __call__(self, b_trial):
        """Evaluates ICV score on the L-kernel bandwidth `b_trial`."""

        #Check for cached Leave One Out data array
        has_cached_LOO_data = True if self._LOO_data_stack \
                is not None and self.cache_LOO_data else False

        if not has_cached_LOO_data:
            data_stack = []

        n = float(self.data.shape[0])

        alpha, sigma = self.params
        
        SUM = (1/(n*b_trial)) * self._RL
        for missing_idx in np.arange(self.data.shape[0]):
            if not has_cached_LOO_data:
                LOOarray = np.concatenate(
                        (self.data[:missing_idx],
                         self.data[missing_idx+1:]),
                        axis = 0,
                    )
                if self.cache_LOO_data:
                    data_stack.append(LOOarray)
            else:
                LOOarray = self._LOO_data_stack[missing_idx]

            diff = (LOOarray - self.data[missing_idx]) / b_trial

            SUM += (1.0/(b_trial * n**2.0)) * np.sum(
                ((1.0+alpha)**2.0)*MVN.pdf(diff, mean = np.zeros(2), cov = 2.0*np.eye(2)) \
                - 2.0*sigma*alpha*(1.0+alpha)*MVN.pdf(diff, mean = np.zeros(2), cov = (1.0+sigma**2.0)*np.eye(2)) \
                + (sigma**2.0)*(alpha**2.0)*MVN.pdf(diff, mean = np.zeros(2), cov = 2.0*(sigma**2.0)*np.eye(2))
            )

            SUM -= (2.0/(n*(n-1.0)*b_trial)) * np.sum(self._L_Kernel(diff))

        if not has_cached_LOO_data and self.cache_LOO_data:
            self._LOO_data_stack = np.stack(data_stack)

        return SUM

    def evaluate(self, b_trial):
        """Evaluates ICV score on the L-kernel bandwidth `b_trial`."""
        return self.__call__(b_trial)

    def bUCV_to_hUCV_factor(self, b_trial):
        """Calculates the multipicative factor required for
        converting the L-kernel bandwidth bUCV to the
        Gaussian kernel bandwidth hUCV."""

        alpha, sigma = self.params
        
        return (((4.0*(1.0 + alpha) - 2.0*alpha*(sigma**3.0))**2.0)/(64.0*np.pi*self._RL))**0.2
