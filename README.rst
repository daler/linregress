``linregress``
==============
Small wrapper class for using R's linear regression functionality in Python,
via `rpy2 <http://rpy.sourceforge.net/rpy2.html>`_.

Example usage
-------------
Create some data::

    >>> x = [1, 2, 3, 4]
    >>> y = [1.2, 3, 7, 10]

Create the model.  The keyword arguments you provide are used to name the
variables in R, so that's how you need to refer to them in the formula::

    >>> from linregress import LinearRegression
    >>> m = LinearRegression(x=x, y=y, formula='y~x')

As an example of this, ``m2`` will be the same model as ``m`` with the same
data -- it's just that the keyword argument names have been changed, and so the
formula is correspondingly changed::

    >>> m2 = LinearRegression(dependent=x, response=y, formula='response~dependent')

Slope, intercept, R-squared, and adjusted R-squared are available as
attributes::

    >>> m.slope
    3.0399999999999996

    >>> m.intercept
    -2.299999999999998

    >>> m.adj_r_squared
    0.97221750212404412

    >>> m.r_squared
    0.98147833474936275

For testing if the slope is different from some expected value (0 by default),
use the ``slope_pval()`` method::

    >>> m.slope_pval()
    0.0093041159117684229

    >>> m.slope_pval(compare_to=0)
    0.0093041159117684229

    >>> m.slope_pval(compare_to=3.0)
    0.90465374107544172

The ``intercept_pval()`` method works similarly::

    >>> m.intercept_pval()
    0.10459053583417365

    >>> m.intercept_pval(compare_to=0)
    0.10459053583417365

    >>> m.intercept_pval(compare_to=5)
    0.012051074953242348

The ``diagnostics()`` method will plot the the diagnostics.  Providing
a filename will save the plots there; with no filename an X11 window will
appear which you can close with ``linregress.rclose()``::

    >>> m.diagnostics('diagnostics.png')


There is also limited support for ANCOVA (analysis of covariance) to compare
slopes or intercepts of different models.

First create another model to compare::

    >>> x2 = [2, 4, 7, 9]
    >>> y2 = [1, 4, 6, 10]
    >>> m2 = LinearRegression(x=x2, y=y2, formula='y~x')

Then run the ANCOVA.  The return value is (pval of slope comparison, pval of
intercept comparison)::

    >>> from linregress import ancova
    >>> ancova(m, m2)
    (0.0089121329373064283, 0.040977724663148127)
