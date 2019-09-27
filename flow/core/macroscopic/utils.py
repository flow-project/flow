"""Utility methods for the macroscopic models."""


class DictDescriptor(object):
    """Dictionary object with descriptor the the individual elements.

    TODO: describe
    """

    def __init__(self, *args):
        """Instantiate the object.

        Parameters
        ----------
        args : (Any, Any, str), iterable
            specifies the key, value, and description of each element in the
            dictionary
        """
        self._dict = {}
        self._descriptions = {}
        self._types = {}

        for arg in args:
            key, value, typ, description = arg

            # in case the same key was used twice, raise an AssertionError
            assert key not in self._dict.keys(), \
                "Key variable '{}' was used twice".format(key)

            # add the new values
            self._dict[key] = value
            self._descriptions[key] = description
            self._types[key] = typ

    def copy(self):
        """Return the dictionary object."""
        return self._dict.copy()

    def description(self, key):
        """Return the description of the specific element."""
        return self._descriptions.get(key, "")

    def type(self, key):
        """Return the description of the specific element."""
        return self._types.get(key, "")
