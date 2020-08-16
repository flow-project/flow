from flow.replay.transfer_tests import create_parser, generate_graphs
import os.path
import os

for (dirpath, _, o) in os.walk(os.path.expanduser("~/ray_results")):
    dirname = dirpath.split('/')[-1]
    if dirname.startswith('CustomPPOTrainer'):
        checkpoint_path = dirpath
        print('CP PATH =', checkpoint_path)

        # in case there is no checkpoint, abort
        hay_cp = False
        for dir in os.listdir(checkpoint_path):
            if dir.startswith('checkpoint'):
                hay_cp = True
                break
        if not hay_cp:
            print('No CP, aborted')
            continue
        
        # create dir for graphs output
        output_dir = os.path.join(checkpoint_path, 'output_graphs')
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        # run graph generation script
        parser = create_parser()

        strategy_name_full = '_'.join(dirpath.split('/')[-2:])

        args = parser.parse_args([
            '-r', checkpoint_path, # '-c', str(checkpoint_number),
            '--gen_emission', '--use_s3', '--num_cpus', str(35),
            '--output_dir', output_dir,
            '--submitter_name', "NatNrj",
            '--strategy_name', strategy_name_full.replace(',', '_').replace(';', '_')
        ])
        generate_graphs(args)