import sys
import os
import types

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
    def __init__(self, GKGUISystem):
        """Initialize the template.

        This assumes that Aimsun is open, as it will try to access the
        current active Aimsun window.

        After that, this call provides different methods to manipulate
        templates in this Aimsun window:
        (1) to load a template, see 'load_template'
        (2) to create a new template by duplicating an existing one, see
            'duplicate_template'
        (3) to create a new blank template, see 'new_template'
        (4) if the template you want to use is already open in the Aimsun
            window, then you don't have to do annything
        """
        self.gui = GKGUISystem.getGUISystem().getActiveGui()
        self.model = self.gui.getActiveModel()



    def load(self, path):
        """Load an existing template into Aimsun
        
        Parameters
        ----------
        path : str
            the path of the template to load

        Raises
        ------
        ValueError
            if path is None
        FileNotFoundError
            if path is provided but does not exist
        RuntimeError
            if path exists but Aimsun couldn't open the template
            for some reason        
        """
        self.gui.loadNetwork(path)
        self.model = self.gui.getActiveModel()

    def new_duplicate(self, path):
        """Create a new template by duplicating an existing one
        
        Parameters
        ----------
        path : str
            the path of the template to be duplicated

        Raises
        ------
        ValueError
            if path is None
        FileNotFoundError
            if path is provided but does not exist
        RuntimeError
            if path exists but Aimsun couldn't open the template
            for some reason       
        """
        self.gui.newDoc(path)
        self.model = self.gui.getActiveModel()

    def new_empty(self):
        """Create a new empty template"""   
        self.gui.newSimpleDoc()
        self.model = self.gui.getActiveModel()

    # TODO add comments about saving the template
    # TODO add checks that gui is active

    ####################################################################
    #                Methods to retrieve Aimsun objects                #
    ####################################################################

    # TODO add checks that self.model is not None

    def __wrap_object(self, obj):
        # TODO clean this function
        """Wrap Aimsun objects to provide more pythonic attribute access.

        For instance, if s is a GKSection object wrapped by this function:
        - s.getSpeed() becomes s.speed
        - s.getName() becomes s.name
        - s.getNbFullLanes() becomes s.nb_full_lanes
        etc.
        """
        self_tmp = self
        def custom_getattr(self, name):

            no_attr_err = AttributeError('\'{}\' has no attribute \'{}\''.format(
                                         self.__class__.__name__, name))

            if name.startswith('get'): # FIXME necessary?
                raise no_attr_err 

            # transform name from attr_name to getAttrName
            aimsun_name = \
                ''.join(map(lambda x: x.capitalize(), name.split('_')))
            
            try:
                aimsun_function = object.__getattribute__(self, 'get' + aimsun_name)
            except AttributeError:
                try:
                    aimsun_function = object.__getattribute__(self, aimsun_name)
                except AttributeError:
                    raise no_attr_err


            try:
                result = aimsun_function()
            except TypeError:
                result = aimsun_function
            
            try:
                self_tmp.__wrap_object(result)
            except TypeError:
                pass
            
            return result
                
        obj.__class__.__getattr__ = custom_getattr

        def custom_setattr(self, name, value):
            try:
                new_name ='set' + ''.join(map(lambda x: x.capitalize(), name.split('_')))
                fct = object.__getattribute__(self, new_name)
                fct(value)
            except AttributeError:
                object.__setattr__(self, name, value)
            return value

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
    def centroid_connections(self):
        return self.__get_objects_by_type("GKCenConnection")

