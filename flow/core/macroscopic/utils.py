"""TODO

"""


class DictDescriptor(object):
    """Dictionary object with descriptor the the individual elements.

    TODO: describe
    """

    def __init__(self, *args):
        """Instantiate the object.

        Parameters
        ----------
        args : (str, Any, Any), iterable
            specifies the description, key, and value of each element in the
            dictionary
        """
        self._dict = {}
        self._descriptions = {}

        for arg in args:
            description, key, value = arg

            # in case the same key was used twice, raise an AssertionError
            assert key in self._dict.keys(), \
                "Key variable '{}' was used twice".format(key)

            # add the new values
            self._dict[key] = value
            self._descriptions[key] = description

    def __get__(self, instance, owner):
        """Return the dictionary object."""
        return self._dict

    def description(self, key):
        """Return the description of the specific element."""
        return self._descriptions.get(key, "")
