"""Generate charts from with .npy files containing custom callables through replay."""

import argparse
from datetime import datetime
import errno
import numpy as np
import matplotlib.pyplot as plt
import os
import pytz
import sys


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
            if file_name[-4:] == ".npy":
                exp_name = os.path.basename(dirpath)
                info_dict = np.load(os.path.join(dirpath, file_name), allow_pickle=True).item()

                info_dicts.append(info_dict)
                exp_names.append(exp_name)
                custom_callable_names.update(info_dict.keys())

    idxs = np.argsort(exp_names)
    exp_names = [exp_names[i] for i in idxs]
    info_dicts = [info_dicts[i] for i in idxs]

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
