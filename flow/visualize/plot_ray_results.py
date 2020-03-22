"""Plot results from ray-based simulations.

This method accepts as input the progress file generated by ray
(usually stored at ~/ray_results/.../progress.csv)
as well as the column(s) to be plotted.

If no column is specified, all existing columns will be printed.

Example usage
-----
::
    python plot_ray_results.py </path/to/file1>.csv </path/to/file2>.csv mean_reward max_reward
"""

import csv
import argparse
import matplotlib.pyplot as plt
from collections import defaultdict


EXAMPLE_USAGE = 'plot_ray_results.py ' + \
    '~/ray_results/experiment-tag/experiment-name/seed-id-1/progress.csv ' + \
    '~/ray_results/experiment-tag/experiment-name/seed-id-2/progress.csv ' + \
    'evaluation/return-average training/return-average'

def plot_multi_progresses(files_columns):
    """Plot ray results from multiple csv files."""
    data = defaultdict(list)
    plt.ion()
     
    filenames = [filename for filename in files_columns if '.csv' in filename]
    columnnames = [column for column in files_columns if '.csv' not in column]
    for filecsv in filenames:
        data = plot_progress(filecsv, columnnames) 
        if not data:
            return
        for col_name, values in data.items():
            plt.plot(values, label=col_name+'/'+filecsv)
        plt.legend()
    plt.show()
    plt.savefig('testresult.png')

def plot_progress(filepath, columns):
    """Plot ray results from a csv file.

    Plot the values contained in the csv file at <filepath> for each column
    in the list of string columns.
    """
    data = defaultdict(list)

    with open(filepath) as f:
        # if columns list is empty, print a list of all columns and return
        if not columns:
            reader = csv.reader(f)
            print('Columns are: ' + ', '.join(next(reader)))
            return

        try:
            reader = csv.DictReader(f)
            for row in reader:
                for col in columns:
                    data[col].append(float(row[col]))
        except KeyError:
            print('Error: {} was called with an unknown column name "{}".\n'
                  'Run "python {} {}" to get a list of all the existing '
                  'columns'.format(__file__, col, __file__, filepath))
            raise
        except ValueError:
            print('Error: {} was called with an invalid column name "{}".\n'
                  'This column contains values that are not convertible to '
                  'floats.'.format(__file__, col))
            raise
    return data

def create_parser():
    """Parse visualization options user can specify in command line.

    Returns
    -------
    argparse.Namespace
        the output parser object
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='[Flow] Plots progress.csv file generated by ray.',
        epilog='Example usage:\n\t' + EXAMPLE_USAGE)

    parser.add_argument('files_columns', type=str, nargs='*', help='Path to the csv files, and names of the columns to plot.')

    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    plot_multi_progresses(args.files_columns)
