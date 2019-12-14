from flow.networks.base import Network
from flow.core.params import InitialConfig
from flow.core.params import TrafficLightParams
import xml.etree.ElementTree as ElementTree
from lxml import etree
from collections import defaultdict

class InductionNet(Network):

    def __init__(self,
                 name,
                 vehicles,
                 net_params,
                 detectors=None,
                 initial_config=InitialConfig(),
                 traffic_lights=TrafficLightParams()):

        if type(net_params.template) is dict:
            if 'det' in net_params.template:
                det = self._detector_infos(net_params.template['det'])
                self.template_detectors = det

        super().__init__(name, vehicles, net_params, initial_config,
                         traffic_lights)
    
    @staticmethod
    def _detector_infos(file_names):
        """Import of detector from a configuration file.

        This is a utility function for computing detector information. It
        imports a network configuration file, and returns the information on
        the detector and add it into the Detector object.

        Parameters
        ----------
        file_names : list of str
            path to the xml file to load

        Returns
        -------
        dict <dict>

            * Key = id of the detector
            * Element = dict of departure speed, vehicle type, depart Position,
              depart edges
        """
        # this is meant to deal with the case that there is only one rou file
        if isinstance(file_names, str):
            file_names = [file_names]

        detector_data = dict()
        type_data = defaultdict(int)

        for filename in file_names:
            # import the .net.xml file containing all edge/type data
            parser = etree.XMLParser(recover=True)
            tree = ElementTree.parse(filename, parser=parser)
            root = tree.getroot()

            # collect the departure properties and routes and vehicles whose
            # properties are instantiated within the .rou.xml file. This will
            # only apply if such data is within the file (it is not implemented
            # by networks in Flow).
            for area_detector in root.findall('e2Detector'):

                # collect the names of each detector type and number of detectors
                # of each type
                type_data["area_detector"] += 1

                detector_data[area_detector.attrib['id']] = {
                    'length': area_detector.attrib['length'],
                    'lane': area_detector.attrib['lane'],
                    'pos': area_detector.attrib['pos'],
                    'freq': area_detector.attrib['freq'],
                    # 'friendlyPos': area_detector.attrib['friendlyPos'],
                    'file': area_detector.attrib['file']
                }


        return detector_data