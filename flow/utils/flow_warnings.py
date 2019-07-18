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


def deprecated(message):
    """Print a deprecation warning. Use as decorator.

    Parameters
    ----------
    message : str
        Description of new method to use
    """
    def deprecated_decorator(func):
        def deprecated_func(*args, **kwargs):
            warnings.warn(
                "{} is a deprecated function. {}".format(func.__name__,
                                                         message),
                category=DeprecationWarning, stacklevel=2)
            warnings.simplefilter('default', DeprecationWarning)
            return func(*args, **kwargs)

        return deprecated_func

    return deprecated_decorator
