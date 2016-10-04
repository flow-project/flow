import xml.etree.ElementTree
root = xml.etree.ElementTree.parse('output.xml').getroot()

desired = ["id", "lane", "pos", "x", "y", "speed"]

for i in range(22):
	for timestep in root:
		for vehicle in timestep:
			if vehicle.attrib["id"] == str(i):
				vehstr = timestep.attrib["time"] + ", "
				for k in desired:
					vehstr += vehicle.attrib[k] + ", "
				print(vehstr[:-2])

