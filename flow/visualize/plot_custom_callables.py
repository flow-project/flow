"""Generate charts from with .npy files containing custom callables through replay."""

import argparse
from datetime import datetime
import errno
import numpy as np
import matplotlib.pyplot as plt
import os
import pytz
import sys

def make_bar_plot(vals, title):
    print(len(vals))
    fig = plt.figure()
    plt.hist(vals, 10, facecolor='blue', alpha=0.5)
    plt.title(title)
    plt.xlim(1000,3000)
    return fig

def plot_trip_distribution(all_trip_energy_distribution):
    non_av_vals = []
    figures = []
    figure_names = []
    for key in all_trip_energy_distribution:
        if key != 'av':
            non_av_vals.extend(all_trip_energy_distribution[key])
        figures.append(make_bar_plot(all_trip_energy_distribution[key], key))
        figure_names.append(key)
    
    figure_names.append('All Non-AV')
    figures.append(make_bar_plot(non_av_vals, 'All Non-AV'))

    figure_names.append('All')
    figures.append(make_bar_plot(non_av_vals + all_trip_energy_distribution['av'], 'All'))

    return figure_names, figures
    
    

def parse_flags(args):
    """Parse training options user can specify in command line.

    Returns
    -------
    argparse.Namespace
        the output parser object
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Parse argument used when running a Flow simulation.",
        epilog="python train.py EXP_CONFIG")
    parser.add_argument("target_folder", type=str,
                        help='Folder containing results')
    parser.add_argument("--output_folder", type=str, required=False, default=None,
                        help='Folder to save charts to.')
    parser.add_argument("--show_images", action='store_true',
                        help='Whether to display charts.')
    parser.add_argument("--heatmap", type=str, required=False,
                        help='Make a heatmap of the supplied variable.')
    return parser.parse_args(args)


if __name__ == "__main__":
    flags = parse_flags(sys.argv[1:])

    date = datetime.now(tz=pytz.utc)
    date = date.astimezone(pytz.timezone('US/Pacific')).strftime("%m-%d-%Y")

    if flags.output_folder:
        if not os.path.exists(flags.output_folder):
            try:
                os.makedirs(flags.output_folder)
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise

    info_dicts = []
    custom_callable_names = set()
    exp_names = []
    for (dirpath, dir_names, file_names) in os.walk(flags.target_folder):
        for file_name in file_names:
            if file_name[-8:] == "info.npy":
                exp_name = os.path.basename(dirpath)
                info_dict = np.load(os.path.join(dirpath, file_name), allow_pickle=True).item()

                info_dicts.append(info_dict)
                print(info_dict.keys())
                exp_names.append(exp_name)
                custom_callable_names.update(info_dict.keys())

    idxs = np.argsort(exp_names)
    exp_names = [exp_names[i] for i in idxs]
    info_dicts = [info_dicts[i] for i in idxs]

    if flags.heatmap is not None:
        heatmap = np.zeros((4, 6))
        pr_spacing = np.around(np.linspace(0, 0.3, 4), decimals=2)
        apr_spacing = np.around(np.linspace(0, 0.5, 6), decimals=2)
        for exp_name, info_dict in zip(exp_names, info_dicts):
            apr_bucket = int(np.around(float(exp_name.split('_')[1][3:]) / 0.1))
            pr_bucket = int(np.around(float(exp_name.split('_')[0][2:]) / 0.1))

            if flags.heatmap not in info_dict:
                print(exp_name)
                continue
            else:
                val = np.mean(info_dict[flags.heatmap])
                print(exp_name, pr_bucket, pr_spacing[pr_bucket], apr_bucket, apr_spacing[apr_bucket], val)
                heatmap[pr_bucket, apr_bucket] = val

        fig = plt.figure()
        plt.imshow(heatmap, interpolation='nearest', cmap='seismic', aspect='equal', vmin=1500, vmax=3000)
        plt.title(flags.heatmap)
        plt.yticks(ticks=np.arange(len(pr_spacing)), labels=pr_spacing)
        plt.ylabel("AV Penetration")
        plt.xticks(ticks=np.arange(len(apr_spacing)), labels=apr_spacing)
        plt.xlabel("Aggressive Driver Penetration")
        plt.colorbar()
        plt.show()
        plt.close(fig)

    else:
        for name in custom_callable_names:
            y_vals = [np.mean(info_dict[name]) for info_dict in info_dicts]
            y_stds = [np.std(info_dict[name]) for info_dict in info_dicts]
            x_pos = np.arange(len(exp_names))

            plt.bar(x_pos, y_vals, align='center', alpha=0.5)
            plt.xticks(x_pos, [exp_name for exp_name in exp_names], rotation=60)
            plt.xlabel('Experiment')
            plt.title('I210 Replay Result: {}'.format(name))
            plt.tight_layout()
            if flags.output_folder:
                plt.savefig(os.path.join(flags.output_folder, '{}-plot.png'.format(name)))

            plt.show()
