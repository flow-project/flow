from enum import Enum

# Definitions of subnetworks
class SubRoute(Enum):
    ALL =  0
    SUB1 = 1 # top left with merge
    SUB2 = 2 # top center intersection
    SUB3 = 3 # top right intersection
    SUB4 = 4 # center intersection
    SUB5 = 5 # bottom left
    SUB6 = 6 # bottom right

    TOP_LEFT = 7 #previous breakdown
    TOP_RIGHT = 8 #previous breakdown
    BOTTOM = 9 #previous breakdown
    FULL_RIGHT = 10 # Aboudy's


#################################################################
# SUBNETWORK DEFINITIONS
#################################################################


# Denotes the route choice behavior of vehicles on an edge.
#
# The key is the name of the starting edge of the vehicle, and the element if
# the name of the next edge the vehicle will take.
#
# 1. If the element is a string name of an edge, the vehicle will always take
#    that edge as a next edge.
# 2. If the element is a list of strings, the vehicle will uniformly choose
#    among all written edges.
# 3. If the element is a list of tuples, the specific next edges are chosen
#    using the weighting specific by the integer second element.
SUBROUTE_EDGES = [
    # Full network
    {
    'e_12': ['e_18', 'e_13'],
    'e_18': 'e_19',
    'e_19': 'e_24',
    'e_24': 'e_24bis',
    'e_24bis': ['e_33','e_23'],
    'e_33': ['e_45', 'e_46', 'e_49'],
    'e_13': 'e_14',
    'e_14': ['e_22', 'e_15'],
    'e_22': 'e_22bis',
    'e_22bis':'e_33',
    'e_15': 'e_16',
    'e_16': 'e_20',
    'e_20': ['e_47', 'e_48'],
    'e_47': ['e_34', 'e_45', 'e_49'],
    'e_45': 'e_43',
    'e_43': 'e_41',
    'e_41': ['e_88','e_39', 'e_50'],
    'e_88': 'e_26',
    'e_26': ['e_12', 'e_2'],
    'e_34': 'e_34bis',
    'e_34bis':'e_23',
    'e_23': ['e_15', 'e_5'],
    'e_5': 'e_4',
    'e_4': 'e_3',
    'e_3': ['e_25', 'e_2'],
    'e_87': ['e_40', 'e_50', 'e_39'],
    'e_40': 'e_42',
    'e_42': 'e_44',
    'e_44': ['e_34', 'e_46'],
    'e_46': 'e_35',
    'e_35': 'e_27',
    'e_27': 'e_6',
    'e_6': ['e_22', 'e_5'],

    'e_48': ['e_78', 'e_81'],
    'e_78': ['e_86', 'e_76'],
    'e_86': ['e_55', 'e_59'],
    'e_56': 'e_89',
    'e_89': ['e_74', 'e_80', 'e_75'],
    'e_80': 'e_83',
    'e_83': 'e_82',
    'e_82': ['e_79', 'e_78'],
    'e_59': ['e_46', 'e_45', 'e_34'],
    'e_76': ['e_90', 'e_74', 'e_80'],
    'e_74': ['e_70', 'e_72'],
    'e_70': 'e_61',
    'e_61': 'e_54',
    'e_54': ['e_40', 'e_88', 'e_39'],
    'e_50': 'e_60',
    'e_60': 'e_69',
    'e_69': ['e_73', 'e_72'],
    'e_73': ['e_80', 'e_75', 'e_90'],
    'e_90': 'e_62',
    'e_62': 'e_57',
    'e_57': ['e_58', 'e_59'],
    'e_58': ['e_76', 'e_77'],
    'e_75': ['e_77', 'e_86'],
    'e_77': ['e_79', 'e_81'],
    'e_81': 'e_84',
    'e_84': 'e_85',
    'e_85': ['e_75', 'e_90', 'e_74'],
    'e_79': ['e_47', 'e_35'],
    'e_49': ['e_55', 'e_58'],
    'e_55': 'e_56',

    'e_25': ['e_87', 'e_30'],
    'e_30': 'e_31',
    'e_31': 'e_32',
    'e_32': 'e_21',
    'e_39': 'e_37',
    'e_37': 'e_29_u',
    'e_29_u': 'e_21',
    'e_21': 'e_8_u',
    'e_9': ['e_10', 'e_92'],
    'e_92': 'e_7',
    'e_7': ['e_8_b', 'e_17'],
    'e_8_b': 'e_8_u',
    'e_17': 'e_28_b',
    'e_28_b': 'e_36',
    'e_36': 'e_93',
    'e_93': 'e_53',
    'e_53': 'e_64',
    'e_64': ['e_65', 'e_67'],
    'e_65': 'e_66',
    'e_66': ['e_91', 'e_63'],
    'e_63': 'e_94',
    'e_94': ['e_51', 'e_52'],
    'e_51': 'e_29_u',
    'e_52': 'e_38',
    'e_38': ['e_50', 'e_88', 'e_40'],
    'e_72': 'e_68',
    'e_68': 'e_66',
    'e_67': 'e_71',
    'e_71': ['e_70', 'e_73'],
    'e_8_u': 'e_9',
    'e_10': 'e_11',
    'e_11': ['e_25', 'e_12'],
    'e_2': 'e_1',
    'e_1': 'e_7',
    'e_91': 'e_64'
    },

    # top left with merge
    {
    'e_22': 'e_22bis',
    'e_22bis':'e_33',
    'e_24': 'e_24bis',
    'e_24bis': ['e_33','e_23'],
    'e_18': 'e_19',
    'e_19': 'e_24',
    'e_34': 'e_34bis',
    'e_34bis':'e_23',
    },

    # top center intersection
    {
    'e_20': ['e_47', 'e_48'],
    'e_46': ['e_35', 'e_48'],
    'e_79': ['e_47', 'e_35'],
    'e_47': ['e_34', 'e_45', 'e_49'],
    'e_33': ['e_45', 'e_46', 'e_49'],
    'e_59': ['e_46', 'e_45', 'e_34'],
    'e_42': 'e_44',
    'e_44': ['e_34', 'e_46', 'e_49'],
    'e_45': 'e_43',
    'e_40': 'e_42',
    'e_43': 'e_41'
    },

    # top right intersection
    {
    'e_48': ['e_78', 'e_81'],
    'e_78': ['e_86', 'e_76'],
    'e_86': ['e_55', 'e_59'],
    'e_56': 'e_89',
    'e_89': ['e_74', 'e_80', 'e_75'],
    'e_80': 'e_83',
    'e_83': 'e_82',
    'e_82': ['e_79', 'e_78'],
    'e_76': ['e_90', 'e_74', 'e_80'],
    'e_73': ['e_80', 'e_75', 'e_90'],
    'e_90': 'e_62',
    'e_62': 'e_57',
    'e_57': ['e_58', 'e_59'],
    'e_58': ['e_76', 'e_77'],
    'e_75': ['e_77', 'e_86'],
    'e_77': ['e_79', 'e_81'],
    'e_81': 'e_84',
    'e_84': 'e_85',
    'e_85': ['e_75', 'e_90', 'e_74'],
    'e_49': ['e_55', 'e_58'],
    'e_55': 'e_56'
    },

    # center intersection
    {
    'e_11': ['e_12', 'e_25'],
    'e_26': ['e_12', 'e_2'],
    'e_3': ['e_25', 'e_2'],
    'e_25': 'e_87',
    'e_88': 'e_26',
    'e_45': 'e_43',
    'e_43': 'e_41',
    'e_41': ['e_88','e_39', 'e_50'],
    'e_38': ['e_50', 'e_88', 'e_40'],
    'e_87': ['e_39', 'e_50', 'e_40'],
    'e_54': ['e_40', 'e_88', 'e_39'],
    'e_40': 'e_42',
    'e_39': 'e_37',
    'e_50': 'e_60',
    'e_60': 'e_69',
    'e_69': ['e_73', 'e_72'],
    'e_74': ['e_72', 'e_70'],
    'e_70': 'e_61',
    'e_61': 'e_54',
    'e_71': ['e_73', 'e_70'],
    'e_52': 'e_38'
    },

    # bottom left
    {
    'e_2': 'e_1',
    'e_1': 'e_7',
    'e_7': ['e_8_b', 'e_17'],
    'e_8_b': 'e_8_u',
    'e_8_u': 'e_9',
    'e_9': ['e_10', 'e_92'],
    'e_92': 'e_7',
    'e_10': 'e_11',
    'e_17': 'e_28_b',
    'e_30': 'e_31',
    'e_31': 'e_32',
    'e_32': 'e_21',
    'e_21': 'e_8_u',
    'e_29_u': 'e_21'
    },

    # bottom right
    {
    'e_36': 'e_93',
    'e_93': 'e_53',
    'e_53': 'e_64',
    'e_64': ['e_65', 'e_67'],
    'e_65': 'e_66',
    'e_66': ['e_91', 'e_63'],
    'e_63': 'e_94',
    'e_91': 'e_64',
    'e_68': 'e_66',
    'e_72': 'e_68',
    'e_67': 'e_71'
    },

    # Top left (initial sub-networks)
    {
    'e_12': ['e_18', 'e_13'],
    'e_18': 'e_19',
    'e_19': 'e_24',
    'e_24': 'e_24bis',
    'e_24bis': ['e_33','e_23'],
    'e_33': ['e_45', 'e_46'],
    'e_13': 'e_14',
    'e_14': ['e_22', 'e_15'],
    'e_22': 'e_22bis',
    'e_22bis':'e_33',
    'e_15': 'e_16',
    'e_16': 'e_20',
    'e_20': 'e_47',
    'e_47': ['e_34', 'e_45'],
    'e_45': 'e_43',
    'e_43': 'e_41',
    'e_41': 'e_88',
    'e_88': 'e_26',
    'e_26': 'e_12',
    'e_34': 'e_34bis',
    'e_34bis':'e_23',
    'e_23': ['e_15', 'e_5'],
    'e_5': 'e_4',
    'e_4': 'e_3',
    'e_3': 'e_25',
    'e_25': 'e_87',
    'e_87': 'e_40',
    'e_40': 'e_42',
    'e_42': 'e_44',
    'e_44': ['e_34', 'e_46'],
    'e_46': 'e_35',
    'e_35': 'e_27',
    'e_27': 'e_6',
    'e_6': ['e_22', 'e_5']
    },

    # Top right
    {
    'e_40': 'e_42',
    'e_42': 'e_44',
    'e_44': ['e_49', 'e_46'],
    'e_46': 'e_48',
    'e_48': ['e_78', 'e_81'],
    'e_78': ['e_86', 'e_76'],
    'e_86': ['e_55', 'e_59'],
    'e_56': 'e_89',
    'e_89': ['e_74', 'e_80', 'e_75'],
    'e_80': 'e_83',
    'e_83': 'e_82',
    'e_82': ['e_79', 'e_78'],
    'e_59': ['e_46', 'e_45'],
    'e_76': ['e_90', 'e_74', 'e_80'],
    'e_74': 'e_70',
    'e_70': 'e_61',
    'e_61': 'e_54',
    'e_54': 'e_40',
    'e_45': 'e_43',
    'e_43': 'e_41',
    'e_41': 'e_50',
    'e_50': 'e_60',
    'e_60': 'e_69',
    'e_69': 'e_73',
    'e_73': ['e_80', 'e_75', 'e_90'],
    'e_90': 'e_62',
    'e_62': 'e_57',
    'e_57': ['e_58', 'e_59'],
    'e_58': ['e_76', 'e_77'],
    'e_75': ['e_77', 'e_86'],
    'e_77': ['e_79', 'e_81'],
    'e_81': 'e_84',
    'e_84': 'e_85',
    'e_85': ['e_75', 'e_90', 'e_74'],
    'e_79': 'e_47',
    'e_47': 'e_45',
    'e_49': ['e_55', 'e_58'],
    'e_55': 'e_56' 
    },

    # Bottom
    {
    'e_25': ['e_87', 'e_30'],
    'e_30': 'e_31',
    'e_31': 'e_32',
    'e_32': 'e_21',
    'e_87': ['e_39', 'e_50'],
    'e_39': 'e_37',
    'e_37': 'e_29_u',
    'e_29_u': 'e_21',
    'e_21': 'e_8_u',
    'e_9': ['e_10', 'e_92'],
    'e_92': 'e_7',
    'e_7': ['e_8_b', 'e_17'],
    'e_8_b': 'e_8_u',
    'e_17': 'e_28_b',
    'e_28_b': 'e_36',
    'e_36': 'e_93',
    'e_93': 'e_53',
    'e_53': 'e_64',
    'e_64': ['e_65', 'e_67'],
    'e_65': 'e_66',
    'e_66': ['e_91', 'e_63'],
    'e_63': 'e_94',
    'e_94': ['e_51', 'e_52'],
    'e_51': 'e_29_u',
    'e_52': 'e_38',
    'e_38': ['e_50', 'e_88'],
    'e_50': 'e_60',
    'e_60': 'e_69',
    'e_69': 'e_72',
    'e_72': 'e_68',
    'e_68': 'e_66',
    'e_67': 'e_71',
    'e_71': 'e_70',
    'e_70': 'e_61',
    'e_61': 'e_54',
    'e_54': ['e_88', 'e_39'],
    'e_8_u': 'e_9',
    'e_10': 'e_11',
    'e_11': 'e_25',
    'e_88': 'e_26',
    'e_26': 'e_2',
    'e_2': 'e_1',
    'e_1': 'e_7'
    },

    # Full right (Aboudy's)
    {
    'e_40': 'e_42',
    'e_42': 'e_44',
    'e_44': ['e_49', 'e_46'],
    'e_49': 'e_58',
    'e_58': 'e_76',
    'e_76': 'e_74',
    'e_74': ['e_70', 'e_72'],
    'e_70': 'e_61',
    'e_61': 'e_54',
    'e_54': 'e_40',
    'e_46': 'e_48',
    'e_48': 'e_78',
    'e_78': 'e_76',
    'e_72': 'e_68',
    'e_68': 'e_66',
    'e_66': 'e_63',
    'e_63': 'e_94',
    'e_94': 'e_52',
    'e_52': 'e_38',
    'e_38': 'e_40',
    },
]

# The cropping dimensions for a subnetwork out of whole Minicity.
# Contains (minWidth, maxWidth, minHeight, maxHeight) 

SUBNET_CROP = [
    (0, 2392, 0, 2404), # Full network
    (0, 5000, 0, 5000), #top left with merge
    (503, 703, 110, 330), # top center intersection
    (0, 5000, 0, 5000), # top right intersection
    (0, 5000, 0, 5000), # center intersection
    (0, 5000, 0, 5000), # bottom left
    (0, 5000, 0, 5000), # bottom right 

    (0, 920, 0, 1020),  # Top left
    (890, 5000, 0, 1020), # Top right
    (0, 3000, 970, 5000), # Bottom
    (2500, 5000, 0, 5000), # Full right
]

# Whether pre-defined subnetwork is not a self-contained loop.
# If routes are clipped and vehicles can exit subnetwork, requires vehicle inflows
# This contains the edges at the border of subnetworks to add inflows to.
SUBNET_INFLOWS = [
    # full network. self-contained, no inflows
    [],

    # top left with merge
    ['e_18', 'e_34', 'e_22'],

    # top center intersection
    ['e_20', 'e_79', 'e_33', 'e_59', 'e_40'],
     
    #top right intersection
    ['e_49', 'e_73', 'e_48'],
     
    # center intersection
    ['e_71', 'e_74', 'e_45', 'e_52', 'e_11', 'e_3', ],
     
    # bottom left
    ['e_2', 'e_29_u', 'e_30'],
     
    #bottom right
    ['e_72', 'e_36'],

    # top left/top right/bottom/Aboudy's full-right. All self-contained, no inflows
    [], [], [], []
]


# How many IDM vehicles to add to subnetwork
SUBNET_IDM = [
    250, # full network
    18,  # top left with merge
    60,  # top center intersection
    75,  # top right intersection
    140,  # center intersection
    75,  # bottom left
    90,  # bottom right
    100,  # top left
    100,  # top right
    100,  # bottom
    100   # Aboudy's full-right
]

# How many RL vehicles to add to subnetwork
SUBNET_RL = [
    0,  # full network
    0,  # top left with merge
    0,  # top center intersection
    0,  # top right intersection
    0,  # center intersection
    0,  # bottom left
    0,  # bottom right
    0,  # top left
    0,  # top right
    0,  # bottom
    0   # Aboudy's full-right
]