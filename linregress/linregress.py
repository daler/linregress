import os
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
        Plot diagnostics for the regression.  If `fn` is provided, then save
        the results to file.  The filetype to be saved is determined by the
        extension.

        Additional kwargs are passed to the saving function (e.g., width=10)
        """
        d = {
                '.pdf': grdevices.pdf,
                '.png': grdevices.png
            }

        if fn:
            ext = os.path.splitext(fn)[1]
            try:
                saver_func = d[ext]
            except KeyError:
                raise ValueError('extension "%s" not supported, '
                        'please use one of %s' % (ext, d.keys()))
            saver_func(file=fn)

        r.layout(r.matrix([1, 2, 3, 4], 2, 2))
        r.plot(self.lm)

        if fn:
            rclose()
        return


def rclose():
    """
    closes the current graphics device (calls dev.off() once)
    """
    grdevices.dev_off()


def ancova(lm1, lm2,  names=('lm1', 'lm2')):
    """
    Compares the slopes and intercepts of two linear models.  Currently this is
    quite limited in that it only compares single-variable linear models that
    have `x` and `y` attributes.

    Returns (pval of slope difference, pval of intercept difference).

    Recall that if the slope is significant, you can't really say anything
    about the intercept.

    """
    # R code, from the extremely useful blog:
    # http://r-eco-evo.blogspot.com/2011/08/
    #           comparing-two-regression-slopes-by.html
    #
    # model1 = aov(y~x*factor, data=df)
    # (interaction term on summary(model1)'s 3rd table line)
    #
    # model2 = aov(y~x+factor, data=df)
    # (2nd table line for "factor" in summary(model2) is the sig of intercept
    # diff)
    #
    # anova(model1, model2)
    #  does removing the interaction term affect the model fit?

    # Construct variables suitable for ANOVA/ANCOVA
    label1 = [names[0] for i in lm1.x]
    label2 = [names[1] for i in lm2.x]
    labels = r.factor(np.array(label1 + label2))
    xi = np.concatenate((lm1.x, lm2.x))
    yi = np.concatenate((lm1.y, lm2.y))

    # The workflow is to populate the formula as a separate environment.
    # This first formula includes the interaction term
    fmla1 = robjects.Formula('yi~xi*labels')
    fmla1.environment['xi'] = xi
    fmla1.environment['yi'] = yi
    fmla1.environment['labels'] = labels
    result1 = r('aov(%s)' % fmla1.r_repr())
    interaction_pval = r.summary(result1)[0].rx2('Pr(>F)')[2]

    # No interaction term
    fmla2 = robjects.Formula('yi~xi+labels')
    fmla2.environment['xi'] = xi
    fmla2.environment['yi'] = yi
    fmla2.environment['labels'] = labels
    result2 = r('aov(%s)' % fmla2.r_repr())
    intercept_pval = r.summary(result2)[0].rx2('Pr(>F)')[1]

    # TODO: anova(result1, result2)?

    return interaction_pval, intercept_pval

if __name__ == "__main__":
    import doctest
    doctest.testmod()

    x = [1, 2, 3, 4]
    y = [1.2, 3, 7, 10]
    m = LinearRegression(x=x, y=y, formula='y~x')

    x2 = [2, 4, 7, 9]
    y2 = [-1, -4, -6, -10]
    m2 = LinearRegression(x=x2, y=y2, formula='y~x')

    result = ancova(m, m2)
