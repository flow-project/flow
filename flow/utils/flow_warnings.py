"""Warnings that may be printed by Flow (e.g. deprecation warnings)."""

import functools
import inspect
import warnings

string_types = (type(b''), type(u''))


def deprecated_attribute(obj, dep_from, dep_to):
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
    warnings.simplefilter('always', PendingDeprecationWarning)
    warnings.warn(
        "The attribute {} in {} is deprecated, use {} instead.".format(
            dep_from, obj.__class__.__name__, dep_to),
        PendingDeprecationWarning
    )


def deprecated(base, new_path):
    """Print a deprecation warning.

    This is a decorator which can be used to mark functions as deprecated. It
    will result in a warning being emitted when the function is used.
    """
    # if isinstance(base, string_types):

    # The @deprecated is used with a 'reason'.
    #
    # .. code-block:: python
    #
    #    @deprecated("please, use another function")
    #    def old_function(x, y):
    #      pass

    def decorator(func1):

        if inspect.isclass(func1):
            fmt1 = "The class {base}.{name} is deprecated, use " \
                   "{new_path} instead."
        else:
            fmt1 = "The function {base}.{name} is deprecated, use " \
                   "{new_path} instead."

        @functools.wraps(func1)
        def new_func1(*args, **kwargs):
            warnings.simplefilter('always', PendingDeprecationWarning)
            warnings.warn(
                fmt1.format(
                    base=base,
                    name=func1.__name__,
                    new_path=new_path
                ),
                category=PendingDeprecationWarning,
                stacklevel=2
            )
            warnings.simplefilter('default', PendingDeprecationWarning)
            return func1(*args, **kwargs)

        return new_func1

    return decorator
    #
    # elif inspect.isclass(reason) or inspect.isfunction(reason):
    #
    #     # The @deprecated is used without any 'reason'.
    #     #
    #     # .. code-block:: python
    #     #
    #     #    @deprecated
    #     #    def old_function(x, y):
    #     #      pass
    #
    #     func2 = reason
    #
    #     if inspect.isclass(func2):
    #         fmt2 = "Call to deprecated class {name}."
    #     else:
    #         fmt2 = "Call to deprecated function {name}."
    #
    #     @functools.wraps(func2)
    #     def new_func2(*args, **kwargs):
    #         warnings.simplefilter('always', DeprecationWarning)
    #         warnings.warn(
    #             fmt2.format(name=func2.__name__),
    #             category=DeprecationWarning,
    #             stacklevel=2
    #         )
    #         warnings.simplefilter('default', DeprecationWarning)
    #         return func2(*args, **kwargs)
    #
    #     return new_func2
    #
    # else:
    #     raise TypeError(repr(type(reason)))
