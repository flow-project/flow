"""Script to visualize the fuel consumption x speed x accel trajectory."""
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import argparse
import os
import os.path

try:
    import imageio
except ImportError:
    print('Please install the imageio library (eg. `pip install imageio`)')
    exit(0)

import flow.config as config
from flow.energy_models.power_demand import PDMCombustionEngine

EXAMPLE_USAGE = """
example usage:
    python ./visualize_energy /path/to/emissions.csv
"""


def generate_fuel_graph(args):
    """Generate the trajectory graph."""
    sim_step = args.sim_step
    verif_path = os.path.join(config.PROJECT_PATH,
                              'flow/visualize/energy_model_verification.csv')
    plot_all_time_steps = args.plot_all_time_steps
    step_interval = args.step_interval

    azim = 225
    verif = pd.read_csv(verif_path)
    fig = plt.figure(figsize=(10, 10))
    fig.gca(projection=Axes3D.name)
    ax1 = fig.add_subplot(1, 1, 1, projection='3d')
    ax1.plot_trisurf(verif['speed(m/s)'],
                     verif['acceleration(m/s^2)'],
                     verif['Tacoma_fit(gal/hr)'] * sim_step / 3600.0,
                     linewidth=0.2)
    ax1.set_title('Tacoma-fit Fuel Consumption', fontsize=18)
    ax1.set_xlabel('Speed (m/s)', fontsize=14)
    ax1.set_ylabel('Acceleration (m/s/s)', fontsize=14)
    ax1.set_zlabel('Fuel Rate', fontsize=14)
    ax1.set_zticks([])
    ax1.view_init(azim=azim)
    energy_model = PDMCombustionEngine()

    baseline = energy_model.get_instantaneous_fuel_consumption(0, 12, 0)
    baseline = baseline * sim_step / 3600.0
    ax1.plot([12], [0], [baseline],
             marker='o', markersize=4, color='#5AFE3E', zorder=20)

    filename = args.emission_file
    df = pd.read_csv(filename)
    sim_step = sorted(df['time'].unique())[1] - sorted(df['time'].unique())[0]
    df.set_index('time', inplace=True)
    df['fuel'] = df.apply(lambda x: energy_model.get_instantaneous_fuel_consumption(
            x['target_accel_no_noise_with_failsafe'], x['speed'], 0), axis=1)

    png_files = []
    for i, row in df.iterrows():
        if (plot_all_time_steps or i == int(i)) and i % step_interval == 0:
            print(i)
            first = max(i-1.0, 0.0)
            p1 = ax1.plot(df.loc[first:i]['speed'],
                          df.loc[first:i]['target_accel_no_noise_with_failsafe'],
                          df.loc[first:i]['fuel'] * sim_step / 3600.0,
                          linewidth=1, color='r', zorder=20)
            p2 = ax1.plot([row['speed']],
                          [row['target_accel_no_noise_with_failsafe']],
                          [row['fuel'] * sim_step / 3600.0],
                          marker='o', markersize=4, color='y', zorder=20)
            t = ax1.annotate('Time: {} s'.format(i), (0.2, 0.9),
                             xycoords='figure fraction', fontsize=20)
            png_name = 'traj_{}.png'.format(i)
            plt.savefig(png_name)
            png_files.append(png_name)
            p1.pop(0).remove()
            p2.pop(0).remove()
            t.remove()
    print()

    if not args.no_gif:
        out_path = args.emission_file.replace('csv', 'gif')
        with imageio.get_writer(out_path, mode='I') as writer:
            for filename in png_files:
                image = imageio.imread(filename)
                writer.append_data(image)
        print('Generate fuel visualization at ' + out_path)

        for filename in png_files:
            os.remove(filename)


def create_parser():
    """Create the parser to capture CLI arguments."""
    parser = argparse.ArgumentParser(
        description='[Flow] Visualize trajectory of agent on speed x '
                    'acceleration x fuel rate graph.',
        epilog=EXAMPLE_USAGE)

    # required input parameters
    parser.add_argument(
        'emission_file',
        type=str,
        help='Path to emission .csv file')

    # optional input parameters
    parser.add_argument(
        '--sim_step',
        type=float,
        default=0.1,
        help='The sim step used when generating the emissions.')
    parser.add_argument(
        '--plot_all_time_steps',
        type=bool,
        default=False,
        help='Specifies whether to visualize all time steps or just the first '
             'one. For instance, if sims_per_step is 5 and this is set, all '
             '5 time steps of each simulation step will be visualized.')
    parser.add_argument(
        '--step_interval',
        type=int,
        default=1,
        help='Specifies n where every n simulation steps will be visualized.')
    parser.add_argument(
        '--no_gif',
        type=bool,
        default=False,
        help='If this is set, the png will not be turned into a gif.')

    return parser


def main():
    """Generate the fuel consumption graph."""
    parser = create_parser()
    args = parser.parse_args()
    generate_fuel_graph(args)


if __name__ == '__main__':
    main()
