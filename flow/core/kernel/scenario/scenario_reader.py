import json


"""
structure of json:
keys: centroids, connections, turnings, nodes, sections
for each key: mapping id => infos

list of available infos:
centroids: type ("in"/"out")
connections: from_id (int), from_name (str), to_id, to_name
nodes: name (str), nb_turnings (int)
sections: length (float, TODO), name (str), numLanes (int), speed (float)
turnings: 
"""

with open('scenario_data.json') as f:  
    data = json.load(f)

print("")  
print("sections: " + ", ".join([data['sections'][x]['name'] for x in data['sections']]))
print("nodes: " + ", ".join([data['nodes'][x]['name'] for x in data['nodes']]))
print("connections: " + ", ".join([data['connections'][x]['from_name']+" -> "+data['connections'][x]['to_name'] for x in data['connections']]))
print("turnings: " + ", ".join([data['turnings'][x]['origin_section_name']+" -> "+data['turnings'][x]['dest_section_name'] for x in data['turnings']]))
print("")

for t in data['turnings'].itervalues():
    for i in range(t['origin_from_lane'], t['origin_to_lane']+1):
        for j in range(t['dest_from_lane'], t['dest_to_lane']+1):
            print("turning from {} (lane {}) to {} (lane {})".format(
                t['origin_section_name'], i,
                t['dest_section_name'], j
            ))
print("")

# FIXME
# centroid connections dest/origin may be reversed (seems ok but not 100% sure)
# length of sections is just an approximation (total lanes length / nb lanes)
# all turning lengths set to 10 (apparently no way to retrieve it except manually computing it from the shape)
# no handling of "half sections" for the moment (ie side input or output lanes)

# actually we can get specific lanes section.getLanes() and then get what we want on them

# we can approximate the length of turning with sth like 
# t.getPolygon().length2D()/2
# not very accurate but can get better