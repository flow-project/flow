import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.colors as colors
import sys
import argparse


cdict = {
        'red'  :  ((0., 0., 0.), (0.2, 1., 1.), (0.6, 1., 1.), (1., 0., 0.)),
        'green':  ((0., 0., 0.), (0.2, 0., 0.), (0.6, 1., 1.), (1., 1., 1.)),
        'blue' :  ((0., 0., 0.), (0.2, 0., 0.), (0.6, 0., 0.), (1., 0., 0.))
        }
my_cmap = colors.LinearSegmentedColormap('my_colormap', cdict, 1024)


def extract_xml_data(filename):
    ''' Extracts the data from the xml files and places them in a dict

    :param filename: location of the xml file with the data needed to be extracted
    :return: A dictionary with all the available data (see keys for each element in dict below)
    '''
    tree = ET.parse(filename)
    root = tree.getroot()

    out_data = {'timestep': [], 'CO': [], 'y': [], 'CO2': [], 'electricity': [], 'type': [], 'id': [], 'eclass': [],
                'waiting': [], 'NOx': [], 'fuel': [], 'HC': [], 'lane': [], 'x': [], 'route': [], 'pos': [],
                'noise': [], 'angle': [], 'PMx': [], 'speed': []}

    count = 0
    for time in root.findall('timestep'):
        t = float(time.attrib['time'])

        for car in time:
            out_data['timestep'].append(t)
            out_data['CO'].append(float(car.attrib['CO']))
            out_data['y'].append(float(car.attrib['y']))
            out_data['CO2'].append(float(car.attrib['CO2']))
            out_data['electricity'].append(float(car.attrib['electricity']))
            out_data['type'].append(car.attrib['type'])
            out_data['id'].append(car.attrib['id'])
            out_data['eclass'].append(car.attrib['eclass'])
            out_data['waiting'].append(float(car.attrib['waiting']))
            out_data['NOx'].append(float(car.attrib['NOx']))
            out_data['fuel'].append(float(car.attrib['fuel']))
            out_data['HC'].append(float(car.attrib['HC']))
            out_data['lane'].append(car.attrib['lane'])
            out_data['x'].append(float(car.attrib['x']))
            out_data['route'].append(car.attrib['route'])
            out_data['pos'].append(float(car.attrib['pos']))
            out_data['noise'].append(float(car.attrib['noise']))
            out_data['angle'].append(float(car.attrib['angle']))
            out_data['PMx'].append(float(car.attrib['PMx']))
            out_data['speed'].append(float(car.attrib['speed']))

    return out_data


def space_time_diagram(filename, edgestarts, show = True, save = False, savename = None):
    ''' Produces space_time diagram

    :param filename: location of the xml file with the data needed to be represented
    :param road_type: if road_type=='ring_road', then the position loops between a range; accordingly, the absolute
                      position is determined by adding to the position in the previous time step the product of the
                      velocity at that time step and the the change in time between the two steps
    :return: Space Time Diagram
    '''
    data = extract_xml_data(filename)
    print('Mean speed:', np.mean(data['speed']), 'm/s')

    unique_id = np.unique(data['id'])
    l = len(data['id'])

    fig = plt.figure(figsize=(16, 8))
    ax = plt.axes()
    
    # norm = plt.Normalize(min(data['speed']), max(data['speed']))
    norm = plt.Normalize(0, 28) # TODO: Make this more modular
    # norm = plt.Normalize(0, max(data['speed']))
    cols = []
    for car in unique_id:
        indx_car = np.where([data['id'][i] == car for i in range(l)])[0]
        
        unique_car_time = [data['timestep'][i] for i in indx_car]
        unique_car_pos = []
        for i in indx_car:
            unique_car_pos.append(data['pos'][i] + edgestarts[data['lane'][i][:-2]])


        # discontinuity from wraparound
        disc = np.where(np.abs(np.diff(unique_car_pos)) >= 0.5)[0]+1
        unique_car_time = np.insert(unique_car_time, disc, np.nan)
        unique_car_pos = np.insert(unique_car_pos, disc, np.nan)
        unique_car_speed = np.insert([data['speed'][i] for i in indx_car], disc, np.nan)

        points = np.array([unique_car_time, unique_car_pos]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        # lc = LineCollection(segments, cmap='inferno', norm=norm)
        lc = LineCollection(segments, cmap=my_cmap, norm=norm)
        
        # Set the values used for colormapping
        lc.set_array(unique_car_speed)
        lc.set_linewidth(1.75)
        cols = np.append(cols, lc)
    for col in cols: line = ax.add_collection(col)
    cbar = fig.colorbar(line, ax = ax)
    cbar.set_label('Velocity (m/s)', fontsize = 15)
    plt.title('Space-Time Diagram', fontsize=20)
    plt.ylabel('Ring Position (m)', fontsize=15)
    plt.xlabel('Time (s)', fontsize=15)    
    
    xmin, xmax = min(data['timestep']), max(data['timestep'])
    xbuffer = (xmax - xmin) * 0.025 # 2.5% of range
    ymin, ymax = min(data['pos']), max(data['pos']) + max(edgestarts.values()) # account for edges
    ybuffer = (ymax - ymin) * 0.025 # 2.5% of range
    
    ax.set_xlim(xmin - xbuffer, xmax + xbuffer)
    ax.set_ylim(ymin - ybuffer, ymax + ybuffer)
    if show:
        plt.show()
    if save and savename:
        fig.savefig(savename + ".png")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the emission.xml file')
    parser.add_argument('length', type=int,
                        help='Length of ring road')
    parser.add_argument('--show', type=str, default='True', 
                        help='Boolean flag to show plot')
    parser.add_argument('--save', type=str, default='False', 
                        help='Boolean flag to save plot')
    parser.add_argument('--imgname', type=str,
                        help='Name for saved image (".png" automatically added)')
    args = parser.parse_args()


    length = args.length
    edgelen = length/4
    edgestarts = dict([("bottom", 0), ("right", edgelen), ("top", 2 * edgelen), ("left", 3 * edgelen)])

    show = False if args.show == 'False' else True
    save = False if args.save == 'False' else True

    if save:
        savename = args.imgname
    else:
        savename = None

    space_time_diagram(args.file, edgestarts, show = show, save = save, savename = savename)