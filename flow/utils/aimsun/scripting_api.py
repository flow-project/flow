import sys
import os

import flow.config as config

SITEPACKAGES = os.path.join(config.AIMSUN_SITEPACKAGES,
                            "lib/python2.7/site-packages")
sys.path.append(SITEPACKAGES)

sys.path.append(os.path.join(config.AIMSUN_NEXT_PATH,
                             'programming/Aimsun Next API/AAPIPython/Micro'))


class AimsunTemplate(object):
    """Interface to do scripting with Aimsun.

    This can be used to create Aimsun templates or to load and modify existing
    ones. It provides a pythonic interface to manipulate the different objects
    accessible via scripting.
    """
    def __init__(self, GKSystem, GKGUISystem):
        """Initialize the template.

        This assumes that Aimsun is open, as it will try to access the
        current active Aimsun window.

        After that, this class provides different methods to manipulate
        templates in this Aimsun window:
        (1) to load a template, see 'load_template'
        (2) to create a new template by duplicating an existing one, see
            'duplicate_template'
        (3) to create a new blank template, see 'new_template'
        (4) if the template you want to use is already open in the Aimsun
            window, then you don't have to do annything

        In cases (2) and (3), see 'save' to then save the template.

        This class takes as parameter the two high-level objects provided
        when interfacing with Aimsun: GKSystem and GKGUISystem. After having
        imported this class, you should be able to create an AimsunTemplate
        object as follows:

            model = AimsunTemplate(GKSystem, GKGUISystem)
        """
        self.GKSystem = GKSystem
        self.GKGUISystem = GKGUISystem

        self.gui = self.GKGUISystem.getGUISystem().getActiveGui()
        self.model = self.gui.getActiveModel()

    def load(self, path):
        """Load an existing template into Aimsun

        Parameters
        ----------
        path : str
            the path of the template to load
        """
        self.gui.loadNetwork(path)
        self.model = self.gui.getActiveModel()
        self.__wrap_object(self.model)

    def new_duplicate(self, path):
        """Create a new template by duplicating an existing one

        Parameters
        ----------
        path : str
            the path of the template to be duplicated
        """
        self.gui.newDoc(path)
        self.model = self.gui.getActiveModel()
        self.__wrap_object(self.model)

    def new_empty(self):
        """Create a new empty template"""
        self.gui.newSimpleDoc()
        self.model = self.gui.getActiveModel()
        self.__wrap_object(self.model)

    def save(self, path):
        """Save the current template

        Parameters
        ----------
        path : str
            the path where the current active template should be saved
        """
        self.gui.saveAs(path)

    def run_replication(self, replication, render=True):
        """Run a replication in Aimsun

        Parameters
        ----------
        replication : GKReplication
            the replication to be run; you can access the list of all available
            GKReplication objects by doing model.replications where
            model is an instance of the AimsunTemplate class.
        render : bool (default: True)
            whether or not the simulation should be rendered
        """
        # "play": run with GUI; "execute": run in batch mode
        mode = 'play' if render else 'execute'
        self.GKSystem.getSystem().executeAction(mode, replication, [], "")

    ####################################################################
    #                Methods to retrieve Aimsun objects                #
    ####################################################################

    def __getattr__(self, name):
        """If trying to access an attribute in this AimsunTemplate object
        fails, try to access it into the Aimsun model object
        """
        return getattr(self.model, name)

    def __wrap_object(self, obj):
        """Wrap Aimsun objects with custom __getattr__ and __setattr__
        functions in order to provide more pythonic attribute access
        and attribute modification.

        For instance:
        - s.getSpeed() becomes s.speed
        - t.getDestination().getName() becomes t.destination.name
        - t.getPolygon().length2D() becomes t.polygon.length2D()
        - exp.setDataValue(model.getColumn(...), ...) becomes
          exp.set_data_value(model.get_column(...), ...)
        - s.setName(new_name) becomes s.name = new_name
        etc.

        This method directly modifies the object and does not return anything.

        For back-compatibility, it is still possible to call the original
        Aimsun methods.
        """
        if obj is None:
            return

        # custom capitalize function that doesn't lowercase the suffix
        def capitalize(str):
            return str[0].upper() + str[1:]

        outer_self = self
        def custom_getattr(self, name):
            # transform name from attr_name to AttrName
            name = ''.join(map(capitalize, name.split('_')))

            # attempt to retrieve getAttrName, or attrName if the first fails
            name1 = 'get' + name
            name2 = name[0].lower() + name[1:]
            try:
                aimsun_fct = object.__getattribute__(self, name1)
            except AttributeError:
                try:
                    aimsun_fct = object.__getattribute__(self, name2)
                except AttributeError:
                    # if both attempts fail, raise an AttributeError with
                    # the original attribute name (instead of name1 or name2)
                    raise AttributeError(
                        '\'{}\' has no attribute \'{}\''.format(
                            self.__class__.__name__, name))

            # call the Aimsun function (which most likely is a getter)
            try:
                result = aimsun_fct()
            except TypeError:
                # if it is not a function, just return the attribute
                result = aimsun_fct

            # wrap whatever object the getter returns, so that we can access
            # deeper attributes (e.g. turning.destination.name)
            try:
                if type(result) is list:
                    map(outer_self.__wrap_object, result)
                else:
                    outer_self.__wrap_object(result)
            except TypeError:
                # we can't wrap a basic type like int; ignore the exception
                pass

            return result

        # assign this custom __getattr__ function to the object
        # note that it will only be called if __getattribute__ fails,
        # so we can still call the original Aimsun functions like s.getName()
        obj.__class__.__getattr__ = custom_getattr

        def custom_setattr(self, name, value):
            try:
                # transform name from attr_name to setAttrName
                name = 'set' + ''.join(map(capitalize, name.split('_')))
                # retrieve the Aimsun setter
                aimsun_setter = object.__getattribute__(self, name)
                # call the setter to set the new value to attribute 'name'
                aimsun_setter(value)
            except AttributeError:
                # if we couldn't retrieve an Aimsun setter, we set the
                # attribute manually
                # FIXME we might want this to raise an error instead
                object.__setattr__(self, name, value)
            return value

        # assign this custom __setattr__ function to the object
        obj.__class__.__setattr__ = custom_setattr

    def __wrap_objects(self, objects):
        """See __wrap_object"""
        map(self.__wrap_object, objects)

    def __get_objects_by_type(self, type_name):
        """Return all Aimsun objects whose type is type_name
        The returned objects are wrapped by __wrap_objects
        """
        type_obj = self.model.getType(type_name)
        objects = self.model.getCatalog().getObjectsByType(type_obj).values()
        self.__wrap_objects(objects)
        return objects

    def find_by_name(self, objects, name):
        """Return the first object in the list 'objects' of Aimsun objects
        whose name is 'name'
        """
        matches = (obj for obj in objects if obj.getName() == name)
        return self.__wrap_object(next(matches, None))

    def find_all_by_type(self, objects, type_name):
        """Return all objects in the list 'objects' of Aimsun objects
        whose type is 'type_name'
        """
        matches = [obj for obj in objects if obj.getTypeName() == type_name]
        return self.__wrap_objects(matches)

    @property
    def sections(self):
        return self.__get_objects_by_type("GKSection")

    @property
    def nodes(self):
        return self.__get_objects_by_type("GKNode")

    @property
    def turnings(self):
        return self.__get_objects_by_type("GKTurning")

    @property
    def cen_connections(self):
        return self.__get_objects_by_type("GKCenConnection")

    @property
    def replications(self):
        return self.__get_objects_by_type("GKReplication")

    @property
    def centroid_configurations(self):
        return self.__get_objects_by_type("GKCentroidConfiguration")

    @property
    def problem_nets(self):
        return self.__get_objects_by_type("GKProblemNet")
