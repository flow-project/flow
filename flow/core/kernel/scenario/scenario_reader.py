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

# scenario_data.json is now deleted after being loaded
# if need to open it, comment the deletion in scenario/aimsun.py
with open('scenario_data.json') as f:  
    data = json.load(f)


# from left to right:
i210_section_ids_l2r = [
    [8009297,8009300],
    [7633314],
    [8009307],
    [21575,8008877],
    [22453,8008880],
    [22435,8008883],
    [21598,8008887],
    [21602,8004006],
    [21610,21614,21591],
    [8015788],
    [21618],
    [8015785,21652],
    [21651,8016318],
    [8004245,21664,8008896],
    [21668,8008899],
    [21673,21657],
    [21677],
    [21681,8008902],
    [21686,8008905],
    [21690,21682],
    [21700],
    [21710,8009118],
    [21711,8008908],
    [21759,8008911],
    [21763,8008914],
    [21767,21697],
    [21755],
    [21774,8005457],
    [21787,8008917],
    [21799,8008920],
    [21803,8008870,8008923],
    [21811,8008926],
    [21822,8003955],
    [21831,8008929],
    [21860,8008932],
    [21875,21781],
    [21879],
    [21886,8008935],
    [8014955,8008864],
    [21891,8008938],
    [22029,21887],
    [22032],
    [22036,8008941],
    [22041,8003964],
    [22072,8008944],
    [22076,8008947],
    [8009533,8003973],
    [22080],
    [22417,8008950],
    [22421,8008953],
    [22425,8008956],
    [22106,22037],
    [22101],
    [22378,22387,8008959],
    [8006440,8008962],
    [22363,8003997],
    [22359,8003994],
    [22167,8008965],
    [22200,8008968],
    [22198,22169],
    [22172],
    [7992606,8003979],
    [22175,8008971],
    [7603458,22176,7996414],
    [8005419,8008974],
    [7996274,7996411],
]

for p in i210_section_ids_l2r:
    for s in p:
        if str(s) not in data['sections']:
            print("pb")



exit(0)

print(len(data['sections']))

class Node(object):
    def __init__(self, data):
        self.data = data
        self.children = []

    def add_child(self, obj):
        self.children.append(obj)

root = Node(8009297)
seen = set()

def build_tree(node):
    seen.add(node.data)
    for t in data['turnings'].values():
        if t['origin_section_id'] == node.data:
            if t['dest_section_id'] not in  seen:
                n = Node(t['dest_section_id'])
                node.add_child(n)
                build_tree(n)

build_tree(root)

def get_leaves(node):
    if len(node.children) == 0:
        return [node.data]
    else:
        leaves = []
        for n in node.children:
            leaves += get_leaves(n)
    return leaves

leaves = get_leaves(root)

def get_path(origin, dest):
    if len(origin.children) == 0:
        if origin.data == dest:
            return [dest]
        else:
            return []
    paths = []
    for n in origin.children:
        paths.append(get_path(n, dest))
    for path in paths:
        if path:
            if path[-1] == dest:
                path = [origin.data] + path
                return path
    return []

print(get_path(root, 7603253))





exit(0)

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