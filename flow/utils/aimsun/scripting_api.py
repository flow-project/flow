import sys
import os
import flow.config as config

SITEPACKAGES = os.path.join(config.AIMSUN_SITEPACKAGES,
                            "lib/python2.7/site-packages")
sys.path.append(SITEPACKAGES)

sys.path.append(os.path.join(config.AIMSUN_NEXT_PATH,
                             'programming/Aimsun Next API/AAPIPython/Micro'))


class AimsunTemplate:
    """Interface to do scripting with Aimsun.

    This can be used to create Aimsun templates or to load existing ones.
    It provides a pythonic interface to manipulate the different objects
    accessible via scripting.
    """

    # TODO remove action from init since we can have None anyway

    def __init__(self, path, action='new'):
        """Initialize the template.

        There are several ways to do that:
        (1) load an existing template
        (2) create a new template by duplicating an existing one
        (3) create a new empty template
        (4) do nothing (this assumes a template is already open in Aimsun)

        Parameters
        ----------
        action : str
            Which method to use to initialize the template
            (1) -> "load"
            (2) -> "duplicate"
            (3) -> "new"
            (4) -> None
        path : str
            The path of the template that (1) should load or that (2) should
            duplicate. This argument will be ignored if action is (3) or (4). 

        Raises
        ------
        ValueError
            if action is not among "load", "duplicate", "new" or None.
        FileNotFoundError
            if action is eiher "load" or "duplicate" and path does not exist.
        RuntimeError
            if action is either "load" or "duplicate" and path exists but
            Aimsun couldn't open the template for some reason.
        """
        if action == 'load':
            load(path)
        elif action == 'duplicate':
            new_doc(path)
        elif action == 'new':
            new_empty()
        elif action == None:
            pass
        else:
            raise ValueError('Class {self.__class__.__name__} initialized '
                             'with invalid argument action="{action}". '
                             'Possible values for argument "action" include: '
                             '"new", "load".')

    def load(self, path):
        """Load an existing template into Aimsun"""
        gui = GKGUISystem.getGUISystem().getActiveGui()
        gui.loadNetwork(path)
        self.model = gui.getActiveModel()

    def new_duplicate(self, path):
        """Create a new template by duplicating an existing one"""
        gui = GKGUISystem.getGUISystem().getActiveGui()
        gui.newDoc(path)
        self.model = gui.getActiveModel()

    def new_empty(self):
        """Create a new empty template"""
        gui = GKGUISystem.getGUISystem().getActiveGui()
        gui.newSimpleDoc()
        self.model = gui.getActiveModel()

    ####################################################################
    #                Methods to retrieve Aimsun objects                #
    ####################################################################

    def __wrap_object(self, obj):
        """Wrap Aimsun objects to provide more pythonic attribute access.

        For instance, if s is a GKSection object wrapped by this function:
        - s.getSpeed() becomes s.speed
        - s.getName() becomes s.name
        - s.getNbFullLanes() becomes s.nb_full_lanes
        etc.
        """
        def custom_getattr(self, name):
            # transform name from attr_name to getAttrName
            aimsun_name = \
                'get' + ''.join(map(lambda x: x.capitalize(), name.split('_')))
            try:
                aimsun_function = getattr(self, aimsun_name)  
                # aimsun_function = self.__getattribute__(aimsun_name)
                return aimsun_function()
                # return __wrap_object(aimsun_function())
            except AttributeError:
                raise AttributeError('\'{}\' has no attribute \'{}\''.format(
                                     self.__class__.__name__, name))
        obj.__getattr__ = custom_getattr

    def __wrap_objects(self, objects):
        """See __wrap_object"""
        return [__wrap_object(obj) for obj in objects]

    def __get_objects_by_type(self, type_name):
        """Return all Aimsun objects whose type is type_name
        The returned objects are wrapped by __wrap_objects
        """
        type_obj = self.model.getType(type_name)
        objects = self.model.getCatalog().getObjectsByType(type_obj).values()
        return __wrap_objects(objects)

    @property
    def sections(self):
        return __get_objects_by_type("GKSection")

    @property
    def nodes(self):
        return __get_objects_by_type("GKNode")

    @property
    def turnings(self):
        return __get_objects_by_type("GKTurning")

    @property
    def centroid_connections(self):
        return __get_objects_by_type("GKCenConnection")
