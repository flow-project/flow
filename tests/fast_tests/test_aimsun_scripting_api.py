import os
import unittest

from flow.utils.aimsun.scripting_api import AimsunTemplate

os.environ['TEST_FLAG'] = 'True'


class TestGUISystemBase(object):
    """Substitution for Aimsun's GKGUISystem class"""
    def getGUISystem(self):
        return self

    def getActiveGui(self):
        return self

    def getActiveModel(self):
        return self


class TestObject(object):
    """Substitution for Aimsun's GKObject class"""
    def __init__(self, name, typename):
        self.var_name = name
        self.var_typename = typename

    def getName(self):
        return self.var_name

    def getTypeName(self):
        return self.var_typename


class TestAimsunScriptingAPI(unittest.TestCase):
    """Tests the functions in flow/utils/aimsun/scripting_api.py"""

    def test_new_and_load(self):
        """Tests that the new/load/save functions call the correct Aimsun
        methods and in the right order
        """
        class TestGUI(object):
            """Substitution for Aimsun's GGui class"""
            def __init__(self):
                self.called = []

            def getActiveModel(self):
                self.called += ['getActiveModel']

            def loadNetwork(self, path):
                self.called += ['loadNetwork']

            def newDoc(self, path):
                self.called += ['newDoc']

            def newSimpleDoc(self):
                self.called += ['newSimpleDoc']

            def saveAs(self, path):
                self.called += ['saveAs']

            def reset(self):
                self.called = []

        class TestGUISystem(object):
            """Substitution for Aimsun's GKGUISystem class"""
            def __init__(self):
                self.active_gui = None

            def getActiveGui(self):
                if self.active_gui is None:
                    self.active_gui = TestGUI()
                return self.active_gui

            def getGUISystem(self):
                return self

        test_gui_system = TestGUISystem()
        model = AimsunTemplate(GKSystem=None, GKGUISystem=test_gui_system)
        test_gui = test_gui_system.active_gui

        self.assertIsNotNone(test_gui)
        self.assertEqual(test_gui.called, ['getActiveModel'])

        test_gui.reset()
        model.load('path')
        self.assertEqual(test_gui.called, ['loadNetwork', 'getActiveModel'])

        test_gui.reset()
        model.new_duplicate('path')
        self.assertEqual(test_gui.called, ['newDoc', 'getActiveModel'])

        test_gui.reset()
        model.new_empty()
        self.assertEqual(test_gui.called, ['newSimpleDoc', 'getActiveModel'])

        test_gui.reset()
        model.save('path')
        self.assertEqual(test_gui.called, ['saveAs'])

    def test_run_replication(self):
        """Tests that the run_replication function calls the correct
        Aimsun function and with the right arguments w.r.t rendering
        """
        class TestSystem(object):
            """Substitution for Aimsun's GKSystem class"""
            def __init__(self):
                self.mode = None

            def getSystem(self):
                return self

            def executeAction(self, mode, *args):
                self.mode = mode

        class TestGUISystem(TestGUISystemBase):
            pass

        test_system = TestSystem()
        test_gui_system = TestGUISystem()
        model = AimsunTemplate(GKSystem=test_system,
                               GKGUISystem=test_gui_system)

        model.run_replication(replication='replication', render=True)
        self.assertEqual(test_system.mode, 'play')

        model.run_replication(replication='replication', render=False)
        self.assertEqual(test_system.mode, 'execute')

    def test_getattr_setattr(self):
        """Test the custom __getattr__ and __setattr__ methods"""
        class TestGUISystem(TestGUISystemBase):
            """Substitution for Aimsun's GKGUISystem class"""
            def getActiveModel(self):
                class Model(object):
                    def __init__(self):
                        self.var = 42
                return Model()

        class TestTurning(object):
            """Substitution for Aimsun's GKTurning class"""
            def getSpeed(self):
                return 13

            def getDestination(self):
                class Destination(object):
                    def getName(self):
                        return 'road66'
                return Destination()

            def getPolygon(self):
                class Polygon(object):
                    def length2D(self):
                        return 99
                return Polygon()

        class TestExperiment(object):
            """Substitution for Aimsun's GKExperiment class"""
            def __init__(self):
                self.exp_name = 'experiment'
                self.columns = {'A': 66}

            def getName(self):
                return self.exp_name

            def setName(self, name):
                self.exp_name = name

            def getColumn(self, col):
                return self.columns[col]

            def setColumn(self, col, val):
                self.columns[col] = val

        test_gui_system = TestGUISystem()
        model = AimsunTemplate(GKSystem=None, GKGUISystem=test_gui_system)

        self.assertEqual(model.var, 42)

        turning = TestTurning()
        exp = TestExperiment()

        model._AimsunTemplate__wrap_object(turning)
        model._AimsunTemplate__wrap_object(exp)

        self.assertEqual(turning.speed, 13)
        self.assertEqual(turning.destination.name, 'road66')
        self.assertEqual(turning.polygon.length2D(), 99)

        self.assertEqual(exp.name, 'experiment')
        exp.name = 'experiment2'
        self.assertEqual(exp.name, 'experiment2')
        self.assertEqual(exp.getName(), 'experiment2')
        exp.set_column('A', 67)
        self.assertEqual(exp.get_column('A'), 67)

        with self.assertRaises(AttributeError):
            exp.setHeight(12)

        class Polygon(object):
            def __init__(self, perim):
                self.perim = perim

            def getPerimeter(self):
                return self.perim

        class TestWrappedObject(object):
            def getPolygons(self):
                return [
                    Polygon(12),
                    Polygon(42)
                ]

        wrapper_object = TestWrappedObject()
        model._AimsunTemplate__wrap_object(wrapper_object)
        self.assertEqual(wrapper_object.polygons[1].perim, 42)

    def test_find(self):
        """Tests the find_by_name and find_all_by_type functions"""
        class TestGUISystem(TestGUISystemBase):
            pass

        test_gui_system = TestGUISystem()
        model = AimsunTemplate(GKSystem=None, GKGUISystem=test_gui_system)

        objects = [
            TestObject('Cat', 'Animal'),
            TestObject('Squirrel', 'Animal'),
            TestObject('Bach', 'Musician'),
            TestObject('Husky', 'Animal'),
            TestObject('Mozart', 'Musician')
        ]

        search1 = model.find_by_name(objects, 'Husky')
        self.assertEqual(search1.name, 'Husky')
        self.assertEqual(search1.type_name, 'Animal')

        search2 = model.find_all_by_type(objects, 'Musician')
        self.assertEqual([x.name for x in search2], ['Bach', 'Mozart'])
        self.assertEqual(set([x.type_name for x in search2]), {'Musician'})

    def test_get_objects_by_type(self):
        """Tests the property methods"""
        class TestGUISystem(TestGUISystemBase):
            """Substitution for Aimsun's GKGUISystem class"""
            def __init__(self):
                self.objects = [
                    TestObject('s1', '_GKSection'),
                    TestObject('s2', '_GKSection'),
                    TestObject('s3', '_GKSection'),
                    TestObject('n1', '_GKNode'),
                    TestObject('n2', '_GKNode'),
                    TestObject('t1', '_GKTurning'),
                    TestObject('t2', '_GKTurning'),
                    TestObject('c1', '_GKCenConnection'),
                    TestObject('r1', '_GKReplication'),
                    TestObject('c1', '_GKCentroidConfiguration'),
                    TestObject('p1', '_GKProblemNet')
                ]

            def getType(self, name):
                return '_' + name

            def getCatalog(self):
                return self

            def getObjectsByType(self, typename):
                res = {}
                for i, obj in enumerate(self.objects):
                    if obj.getTypeName() == typename:
                        res[i] = obj
                return res

        test_gui_system = TestGUISystem()
        model = AimsunTemplate(GKSystem=None, GKGUISystem=test_gui_system)

        section_names = [x.name for x in model.sections]
        section_types = [x.type_name for x in model.sections]
        self.assertEqual(sorted(section_names), ['s1', 's2', 's3'])
        self.assertEqual(set(section_types), {'_GKSection'})

        node_names = [x.name for x in model.nodes]
        node_types = [x.type_name for x in model.nodes]
        self.assertEqual(sorted(node_names), ['n1', 'n2'])
        self.assertEqual(set(node_types), {'_GKNode'})

        turning_names = [x.name for x in model.turnings]
        turning_types = [x.type_name for x in model.turnings]
        self.assertEqual(sorted(turning_names), ['t1', 't2'])
        self.assertEqual(set(turning_types), {'_GKTurning'})

        cen_connection_names = [x.name for x in model.cen_connections]
        cen_connection_types = [x.type_name for x in model.cen_connections]
        self.assertEqual(sorted(cen_connection_names), ['c1'])
        self.assertEqual(set(cen_connection_types), {'_GKCenConnection'})

        replication_names = [x.name for x in model.replications]
        replication_types = [x.type_name for x in model.replications]
        self.assertEqual(sorted(replication_names), ['r1'])
        self.assertEqual(set(replication_types), {'_GKReplication'})

        cen_config_names = [x.name for x in model.centroid_configurations]
        cen_config_types = [x.type_name for x in model.centroid_configurations]
        self.assertEqual(sorted(cen_config_names), ['c1'])
        self.assertEqual(set(cen_config_types), {'_GKCentroidConfiguration'})

        problem_net_names = [x.name for x in model.problem_nets]
        problem_net_types = [x.type_name for x in model.problem_nets]
        self.assertEqual(sorted(problem_net_names), ['p1'])
        self.assertEqual(set(problem_net_types), {'_GKProblemNet'})


if __name__ == '__main__':
    unittest.main()
