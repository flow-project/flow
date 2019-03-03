"""Warnings that may be printed by Flow (e.g. deprecation warnings)."""

import warnings


def deprecation_warning(obj, dep_from, dep_to):
    """Print a deprecation warning.

    Parameters
    ----------
    obj : class
        The class with the deprecated attribute
    dep_from : str
        old (deprecated) name of the attribute
    dep_to : str
        new name for the attribute
    """
    warnings.warn(
        "The attribute {} in {} is deprecated, use {} instead.".format(
            dep_from, obj.__class__.__name__, dep_to))
