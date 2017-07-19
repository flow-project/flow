from lxml import objectify
from scipy import interpolate
import numpy as np
from numpy import transpose as T
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import sys

def plot(file, length, edgestarts, num_lanes, speedLimit = 55, show=True, save=False, speedRange=None, fuelRange=None):
    # Plot results

    trng, xrng, avgspeeds, lanespeeds, (laneoccupancy, typecolors), totfuel, looptimes = parsexml(file, edgestarts, length, speedLimit)

    if speedRange == 'avg':
        mnspeed = min([min(s) for s in avgspeeds.values()])
        mxspeed = max([max(s) for s in avgspeeds.values()])
    elif speedRange == 'tight':
        mnspeed = min([np.percentile(s,5) for s in avgspeeds.values()])
        mxspeed = max([np.percentile(s,95) for s in avgspeeds.values()])
    elif speedRange == 'road':
        mnspeed, mxspeed = 0, speedLimit
    elif speedRange is None or speedRange == 'car':
        mnspeed, mxspeed = 0, speedLimit # TODO should this be a different max speed?
    else:
        mnspeed, mxspeed = speedRange

    if fuelRange == 'avg':
        mnfuel = min([min(s) for s in avgfuels.values()])
        mxfuel = max([max(s) for s in avgfuels.values()])
    elif fuelRange == 'tight':
        mnfuel = min([np.percentile(s,5) for s in avgfuels.values()])
        mxfuel = max([np.percentile(s,95) for s in avgfuels.values()])
    elif fuelRange is None or fuelRange == 'car':
        mnfuel, mxfuel = None, None
    else:
        mnfuel, mxfuel = fuelRange

    print("Generating interpolated plot...")
    plt = pcolor_multi("Traffic jams (%d lanes, %s)" % (num_lanes, 'exp'), 
            (xrng, "Position along loop (m)"),
            (trng, "Time (s)"),
            (avgspeeds, "Average loop speed (m/s)"),
            (lanespeeds, mnspeed, mxspeed, "Speed (m/s)"),
            (looptimes, "Loop transit time (s)"),
            (totfuel, mnfuel, mxfuel, "Speed std. dev. (m/s)"))

    # plt = spacetime_plot((trng, "Time (s)"),
    #                 (xrng, "Position along loop (m)"),
    #                 (lanespeeds, mnspeed, mxspeed, "Speed (m/s)"))

    fig = plt.gcf()
    if show:
        plt.show()
    if save:
        fig.savefig('debug/img/' + file[15:-13] + ".png")
    return plt


def spacetime_plot(title, x_s, y_s, v_s, s_s, l_s, f_s):
    xrng, xlabel = x_s
    yrng, ylabel = y_s
    vdict, vlabel = v_s
    sdict, smin, smax, slabel = s_s
    ldict, llabel = l_s
    fdict, fmin, fmax, flabel = f_s

    numlanes = len(sdict)

    fig, axarr = plt.subplots(numlanes, 1, 
            figsize=(8, 8), dpi=100)

    axarr.axis('off')


    # axarr[0,0].axis('off')
    # axarr[-1,1].axis('off')
    # vax = axarr[-1,0]
    # fax = axarr[-1,1]

    x, y = np.meshgrid(xrng, yrng)

    for (ax, sid) in zip([axarr], sorted(sdict)):
        tv = T(np.array(sdict[sid]))
        cax = ax.pcolormesh(T(y), T(x), tv,
                vmin=smin, vmax=smax, 
                cmap=my_cmap)
        ax.set_ylabel(xlabel + "\nLane %s"%sid)
        ax.axis('tight')

        # vmn = np.min(tv, axis=0)
        # v25 = np.percentile(tv, 25, axis=0)
        # v75 = np.percentile(tv, 75, axis=0)
        # vmx = np.max(tv, axis=0)

        # fax.plot(yrng, fdict[sid], label="lane %s" % sid)

        # handles, labels = fax.get_legend_handles_labels()
        # lbl = handles[-1]
        # linecolor = lbl.get_c()
        # linecolor = 'b'
        # fig.text(0.5, 0.95, 'Lane %s' %s, transform=fig.transFigure, horizontalalignment='center')
        # ax.set_title("lane %s" % sid, color=linecolor, x=-0.1)

        # lc = colors.colorConverter.to_rgba(linecolor, alpha=0.1)
        # ax2.fill_between(yrng, vmn, vmx, color=lc)
        # lc = colors.colorConverter.to_rgba(linecolor, alpha=0.25)
        # ax2.fill_between(yrng, v25, v75, color=lc)
        # ax2.plot(yrng, vdict[sid], label="lane %s" % sid, color=linecolor)

        # ax2.set_ylabel(vlabel + "\nLane %s"%sid)
        # ax2.set_ylim([smin, smax])

    ax.set_xlabel(ylabel)

    # boxplotdata1 = [[]]
    # boxplotdata2 = [[]]
    # boxplotpos1 = [1]
    # boxplotpos2 = [2]
    # boxplotlabels = ["All lanes"]
    # bp = 4
    # for lid in sorted(ldict):
    #     #boxplotdata.append(lt)
    #     lt = ldict[lid]
    #     ft = fdict[lid]
    #     # print(boxplotdata1[0])
    #     # print(len(lt))

    #     boxplotdata1[0].extend(lt[len(lt)//2:])
    #     boxplotdata2[0].extend(ft[len(ft)//2:])
    #     boxplotdata1.append(lt[len(lt)//2:])
    #     boxplotdata2.append(ft[len(ft)//2:])
    #     boxplotlabels.append("lane %s" % lid)
    #     boxplotpos1.append(bp)
    #     boxplotpos2.append(bp+1)
    #     bp+=3
    #     # print "Total looptime, lane %s:" % lid, np.mean(lt[100:]), np.percentile(lt[100:], (0, 25, 75, 100))
    #     # print "Total fuel consumed, lane %s:" % lid, np.mean(ft[100:]), np.percentile(ft[100:], (0, 25, 75, 100))

    # vax2 = vax.twinx()
    # mybp(vax, boxplotdata1, boxplotpos1, llabel, '#9999ff', 'b')
    # mybp(vax2, boxplotdata2, boxplotpos2, flabel, '#ff9999', 'r')

    # vax.set_xticklabels([""] + boxplotlabels)
    # vax.set_xticks([0] + [x*3+1.5 for x in range(len(boxplotlabels))])
    # vax2.set_xticks([0] + [x*3+1.5 for x in range(len(boxplotlabels))])
    # vax.set_title(title)

    ax.set_ylabel(flabel)
    # if fmin is not None and fmax is not None:
    #     fax.set_ylim([fmin, fmax])
    ax.set_xlabel(ylabel)
    '''
    fig.text(0.5, 0.975, title, 
            horizontalalignment='center', verticalalignment='top')
    '''
    # Add colorbar, and adjust various fudge factors to make things align properly
    fig.subplots_adjust(right=0.85)
    if numlanes == 2:
        cbm = 0.52
        cbl = 0.38
    elif numlanes == 1:
        cbm=0.665
        cbl=0.235
    else:
        cbm=0.1
        cbl=0.8

    cbar_ax = fig.add_axes([0.89, cbm, 0.02, cbl])

    ticks = np.linspace(smin, smax, 6)
    cbar = fig.colorbar(cax, cax=cbar_ax, ticks=ticks)
    cbar.ax.set_yticklabels(ticks)  # vertically oriented colorbar
    cbar.ax.set_ylabel(slabel, rotation=270, labelpad=20)

    return plt



def interp(x, y, xmax, vdefault=0):
        if len(x) == 0:
            x = [0]
            y = [vdefault]

        newx = []; newy = []

        newindex = x.index(max(x))
        newx.append(x[newindex] - xmax)
        newy.append(y[newindex])

        newindex = x.index(min(x))
        newx.append(x[newindex] + xmax)
        newy.append(y[newindex])

        x.extend(newx); y.extend(newy)

        f = interpolate.interp1d(x, y, assume_sorted=False)
        return f

def parsexml(fn, edgestarts, xmax, vdefault=0):
    obj = objectify.parse(open(fn)).getroot()

    trng = []
    xrng = range(0, xmax)
    looptimes = {}
    lanespeeds = {}
    laneoccupancy = {}
    avgspeeds = {}
    totfuel = {}
    typecolors = {}

    for timestep in obj.timestep:
        t = float(timestep.get("time"))

        lanedata = {}
        try:
            for vehicle in timestep.vehicle:
                d = {}
                d["name"] = vehicle.get("id")
                d["type"] = vehicle.get("id")[:-4]
                d["edge"] = vehicle.get("lane")[:-2]
                d["v"] = float(vehicle.get("speed"))
                d["pos"] = float(vehicle.get("pos"))
                d["x"] = d["pos"] + edgestarts[d["edge"]]

                d["CO2"] = float(vehicle.get("CO2"))
                d["CO"] = float(vehicle.get("CO"))
                d["fuel"] = float(vehicle.get("fuel"))

                lid = vehicle.get("lane")[-1]
                lanedata.setdefault(lid, []).append(d)
        except AttributeError:
            pass

        for lid, thislane in lanedata.items():
            # interpolate the values of velocity, fuel to get a _f_unction of loop position
            vf = interp([x["x"] for x in thislane], [x["v"] for x in thislane], xmax, vdefault)
            ff = interp([x["x"] for x in thislane], [x["fuel"] for x in thislane], xmax, 0)
            types = set([x["type"] for x in thislane])
            for tp in types:
                if tp not in typecolors:
                    typecolors[tp] = len(typecolors)+1

            intx = dict((int(x["x"]), typecolors[x["type"]]) for x in thislane)

            '''
            fuel = ff(xrng)/vf(xrng)
            '''

            fx = vf(xrng)
            lanespeeds.setdefault(lid, [[0]*len(xrng)]*len(trng)).append(fx)
            laneoccupancy.setdefault(lid, [[0]*len(xrng)]*len(trng)).append([intx.get(x, 0) for x in xrng])

            '''
            dx = np.diff(np.array(xrng + [xrng[0] + xmax]))
            dt = dx * 1./fx
            looptime = np.sum(dt)
            avgspeed = xmax / looptime
            loopfuel = np.sum(fuel*dx)
            '''

            avgspeed = np.mean(vf(xrng))
            looptime = xmax/avgspeed
            loopfuel = np.mean([x["fuel"] for x in thislane])*looptime

            # XXX Quick hack : Plot std dev of velocity instead.
            loopfuel = np.std([x["v"] for x in thislane])

            avgspeeds.setdefault(lid, [vdefault]*len(trng)).append(avgspeed)
            totfuel.setdefault(lid, [0]*len(trng)).append(loopfuel)
            looptimes.setdefault(lid, [xmax * 1.0 / vdefault]*len(trng)).append(looptime)

        trng.append(t)

    '''
    for lid, lt in looptimes.iteritems():
        print "Total looptime, lane %s:" % lid, np.mean(lt[100:]), np.percentile(lt[100:], (0, 25, 75, 100))
    for lid, ft in totfuel.iteritems():
        print "Total fuel consumed, lane %s:" % lid, np.mean(ft[100:]), np.percentile(ft[100:], (0, 25, 75, 100))
    '''
    return trng, xrng, avgspeeds, lanespeeds, (laneoccupancy, typecolors), totfuel, looptimes


cdict = {
        'red'  :  ((0., 0., 0.), (0.2, 1., 1.), (0.6, 1., 1.), (1., 0., 0.)),
        'green':  ((0., 0., 0.), (0.2, 0., 0.), (0.6, 1., 1.), (1., 1., 1.)),
        'blue' :  ((0., 0., 0.), (0.2, 0., 0.), (0.6, 0., 0.), (1., 0., 0.))
        }
my_cmap = colors.LinearSegmentedColormap('my_colormap', cdict, 1024)

def scatter(title, x_s, y_s, s_s):
    x, xlabel = x_s
    y, ylabel = y_s
    s, smin, smax, slabel = s_s

    fig, ax = plt.subplots()

    cax = ax.scatter(x, y, c=s, 
            vmin=smin, vmax=smax, 
            s=3, edgecolors='none', 
            cmap=my_cmap)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.margins(0)
    ax.invert_yaxis()

    # Add colorbar
    ticks = [smin, (smin+smax)/2, smax]
    cbar = fig.colorbar(cax, ticks=ticks)
    cbar.ax.set_yticklabels(ticks)  # vertically oriented colorbar
    cbar.ax.set_ylabel(slabel, rotation=270, labelpad=20)

    return plt

def pcolor(title, x_s, y_s, s_s):

    xrng, xlabel = x_s 
    yrng, ylabel = y_s
    s, smin, smax, slabel = s_s

    fig, ax = plt.subplots()

    #y, x = mgrid[yrng, xrng]
    #cax = ax.pcolor(x, y, s, 
    x, y = np.meshgrid(xrng, yrng)
    cax = ax.pcolormesh(x, y, np.array(s),
            vmin=smin, vmax=smax, 
            cmap=my_cmap)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.axis('tight')
    ax.invert_yaxis()

    # Add colorbar
    ticks = [smin, (smin+smax)/2, smax]
    cbar = fig.colorbar(cax, ticks=ticks)
    cbar.ax.set_yticklabels(ticks)  # vertically oriented colorbar
    cbar.ax.set_ylabel(slabel, rotation=270, labelpad=20)

    return plt

def mybp(vax, bpdata, bppos, label, ltcolor, dkcolor):
    bp = vax.boxplot(bpdata, positions=bppos, widths=0.6, patch_artist=True)
    for box in bp['boxes']:
        box.set( facecolor = ltcolor )
    for line in bp['medians']:
            # get position data for median line
            x1, y = line.get_xydata()[0] # left of median line
            x2, y = line.get_xydata()[1] # right of median line
            # overlay median value
            vax.text((x1+x2)/2, y, '%.1f' % y, horizontalalignment='center') 
    vax.set_ylabel(label, color=dkcolor)
    for tl in vax.get_yticklabels():
        tl.set_color(dkcolor)

def pcolor_multi(title, x_s, y_s, v_s, s_s, l_s, f_s):

    xrng, xlabel = x_s
    yrng, ylabel = y_s
    vdict, vlabel = v_s
    sdict, smin, smax, slabel = s_s
    ldict, llabel = l_s
    fdict, fmin, fmax, flabel = f_s

    numlanes = len(sdict)

    fig, axarr = plt.subplots(numlanes+2, 2, 
            figsize=(16, 16), dpi=100)

    axarr[-2,0].axis('off')
    axarr[-2,1].axis('off')
    vax = axarr[-1,0]
    fax = axarr[-1,1]

    x, y = np.meshgrid(xrng, yrng)

    for (ax2, ax, sid) in zip(axarr[:,0], axarr[:,1], sorted(sdict)):
        tv = T(np.array(sdict[sid]))
        cax = ax.pcolormesh(T(y), T(x), tv,
                vmin=smin, vmax=smax, 
                cmap=my_cmap)
        ax.set_ylabel(xlabel + "\nLane %s"%sid)
        ax.axis('tight')

        vmn = np.min(tv, axis=0)
        v25 = np.percentile(tv, 25, axis=0)
        v75 = np.percentile(tv, 75, axis=0)
        vmx = np.max(tv, axis=0)

        fax.plot(yrng, fdict[sid], label="lane %s" % sid)

        handles, labels = fax.get_legend_handles_labels()
        lbl = handles[-1]
        linecolor = lbl.get_c()
        linecolor = 'b'
        #fig.text(0.5, 0.95, 'Lane %s'%s, transform=fig.transFigure, horizontalalignment='center')
        #ax.set_title("lane %s" % sid, color=linecolor, x=-0.1)

        lc = colors.colorConverter.to_rgba(linecolor, alpha=0.1)
        ax2.fill_between(yrng, vmn, vmx, color=lc)
        lc = colors.colorConverter.to_rgba(linecolor, alpha=0.25)
        ax2.fill_between(yrng, v25, v75, color=lc)
        ax2.plot(yrng, vdict[sid], label="lane %s" % sid, color=linecolor)

        ax2.set_ylabel(vlabel + "\nLane %s"%sid)
        ax2.set_ylim([smin, smax])

    ax.set_xlabel(ylabel)
    ax2.set_xlabel(ylabel)

    boxplotdata1 = [[]]
    boxplotdata2 = [[]]
    boxplotpos1 = [1]
    boxplotpos2 = [2]
    boxplotlabels = ["All lanes"]
    bp = 4
    for lid in sorted(ldict):
        #boxplotdata.append(lt)
        lt = ldict[lid]
        ft = fdict[lid]
        # print(boxplotdata1[0])
        # print(len(lt))

        boxplotdata1[0].extend(lt[len(lt)//2:])
        boxplotdata2[0].extend(ft[len(ft)//2:])
        boxplotdata1.append(lt[len(lt)//2:])
        boxplotdata2.append(ft[len(ft)//2:])
        boxplotlabels.append("lane %s" % lid)
        boxplotpos1.append(bp)
        boxplotpos2.append(bp+1)
        bp+=3
        # print "Total looptime, lane %s:" % lid, np.mean(lt[100:]), np.percentile(lt[100:], (0, 25, 75, 100))
        # print "Total fuel consumed, lane %s:" % lid, np.mean(ft[100:]), np.percentile(ft[100:], (0, 25, 75, 100))

    vax2 = vax.twinx()
    mybp(vax, boxplotdata1, boxplotpos1, llabel, '#9999ff', 'b')
    mybp(vax2, boxplotdata2, boxplotpos2, flabel, '#ff9999', 'r')

    vax.set_xticklabels([""] + boxplotlabels)
    vax.set_xticks([0] + [x*3+1.5 for x in range(len(boxplotlabels))])
    vax2.set_xticks([0] + [x*3+1.5 for x in range(len(boxplotlabels))])
    vax.set_title(title)

    fax.set_ylabel(flabel)
    if fmin is not None and fmax is not None:
        fax.set_ylim([fmin, fmax])
    fax.set_xlabel(ylabel)
    '''
    fig.text(0.5, 0.975, title, 
            horizontalalignment='center', verticalalignment='top')
    '''
    # Add colorbar, and adjust various fudge factors to make things align properly
    fig.subplots_adjust(right=0.85)
    if numlanes == 2:
        cbm = 0.52
        cbl = 0.38
    elif numlanes == 1:
        cbm=0.665
        cbl=0.235
    else:
        cbm=0.1
        cbl=0.8

    cbar_ax = fig.add_axes([0.89, cbm, 0.02, cbl])

    ticks = np.linspace(smin, smax, 6)
    cbar = fig.colorbar(cax, cax=cbar_ax, ticks=ticks)
    cbar.ax.set_yticklabels(ticks)  # vertically oriented colorbar
    cbar.ax.set_ylabel(slabel, rotation=270, labelpad=20)

    return plt

args = sys.argv
fname = 'debug/cfg/data/' + args[1] + '.emission.xml'
    # example filename:
    # debug/cfg/data/sugiyama_test_eugene-230m1l.emission.xml
    # so command line input would be 
    # sugiyama_test_eugene-230m1l
    # length is 230

length = int(args[2])

edgelen = length/4
edgestarts = dict([("bottom", 0), ("right", edgelen), ("top", 2 * edgelen), ("left", 3 * edgelen)])

plot(file = fname, length = length, num_lanes = 1, edgestarts = edgestarts, speedLimit = 30, save = True)
