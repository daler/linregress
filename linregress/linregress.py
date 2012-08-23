import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects.numpy2ri import numpy2ri
from rpy2.robjects import r
robjects.conversion.py2ri = numpy2ri

from rpy2.robjects.packages import importr
grdevices = importr('grDevices')


class LinearRegression(object):
    def __init__(self, formula, **kwargs):
        """
        Class for managing linear regression in R.

        Data are specified with the keyword arguments, which are passed to R's
        global environment.  They are first converted to NumPy arrays.

        For example, the kwarg `x=[1,2,3,4]` will add the list of four
        numbers to R's global env with the variable name `x`.  You can then
        access `x` from the formula.

        `formula` is a string passed verbatim to R's `lm()` function.

        Example usage::

            >>> x = [1, 2, 3, 4]
            >>> y = [1.2, 3, 7, 10]
            >>> m = LinearRegression(x=x, y=y, formula='y~x')
            >>> m.slope
            3.0399999999999996

            >>> m.intercept
            -2.299999999999998

            >>> m.adj_r_squared
            0.97221750212404412

            >>> m.slope_pval(0)
            0.0093041159117684229

            >>> m.intercept_pval(0)
            0.10459053583417365

            >>> # Variables accessible as NumPy arrays
            >>> m.x
            array([1, 2, 3, 4])

        Cross-check with scipy.stats.linregress::

            >>> from scipy.stats import linregress as scipy_linregress
            >>> results = scipy_linregress(x, y)
            >>> eps = 1e-15
            >>> assert abs(results[0] - m.slope) < eps
            >>> eps = 1e-10
            >>> assert abs(results[1] - m.intercept) < eps
            >>> eps = 1e-15
            >>> assert abs(results[2] ** 2 - m.r_squared) < eps
            >>> eps = 1e-15
            >>> assert abs(results[3] - m.slope_pval(0)) < eps


        TODO:
            - support for more complex models (requires examining the coeffs
              matrix to see what's included)

        """

        for k, v in kwargs.items():
            v = np.array(v)
            robjects.globalenv[k] = v
            setattr(self, k, v)

        self.lm = r.lm(formula)
        self.summary = r.summary(self.lm)
        coeffs = self.summary.rx2('coefficients')
        self._intercept_p, self._slope_p = coeffs[6], coeffs[7]

    def __str__(self):
        return "y=%.2fx%+.2f (R2=%.2f, p[slope]=%.2e, p[intercept]=%.2e)" \
                % (self.slope, self.intercept, self.r_squared,
                        self.slope_pval(), self.intercept_pval())

    @property
    def intercept(self):
        return self.summary.rx2('coefficients')[0]

    @property
    def slope(self):
        return self.summary.rx2('coefficients')[1]

    @property
    def slope_std(self):
        return self.summary.rx2('coefficients')[3]

    @property
    def intercept_std(self):
        return self.summary.rx2('coefficients')[2]

    def intercept_pval(self, compare_to=0):
        """
        Test if the intercept is different from some value (by default, zero)
        """
        tb1 = np.abs((compare_to - self.intercept) / self.intercept_std)
        one_tailed_p = r.pt(
                tb1, self.summary.rx2('df')[1], lower_tail=False)[0]
        two_tailed_p = one_tailed_p * 2
        if compare_to == 0:
            assert two_tailed_p == self._intercept_p
        return two_tailed_p

    def slope_pval(self, compare_to=0):
        """
        Test if the slope is different from some value (by default, zero)
        """
        tb0 = np.abs((compare_to - self.slope) / self.slope_std)
        one_tailed_p = r.pt(
                tb0, self.summary.rx2('df')[1], lower_tail=False)[0]
        two_tailed_p = one_tailed_p * 2
        if compare_to == 0:
            assert two_tailed_p == self._slope_p
        return two_tailed_p

    @property
    def adj_r_squared(self):
        return self.summary.rx2('adj.r.squared')[0]

    @property
    def r_squared(self):
        return self.summary.rx2('r.squared')[0]

    def diagnostics(self, fn=None):
        """
        Plot diagnostics for the regression.  If `fn` is provided, then save as
        a PDF.
        """
        r.layout(r.matrix([1, 2, 3, 4], 2, 2))
        if fn:
            grdevices.pdf(file=fn)
        r.plot(self.lm)
        if fn:
            rclose()
            rclose()
        return fn


def rclose():
    """
    closes the current graphics device (calls dev.off() once)
    """
    grdevices.dev_off()


if __name__ == "__main__":
    import doctest
    doctest.testmod()
