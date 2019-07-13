from flow.core.experiment import Experiment


def create_parser():
    return object()


if __name__ == "__main__":
    args = create_parser()

    # Get the flow_params object.
    flow_params = __import__("exp_configs.non_rl.{}".format(args.name))

    # Update some variables based on inputs.
    flow_params['sim'].render = args.render
    flow_params['simulator'] = 'aimsun' if args.aimsun else 'traci'

    # Create the experiment object.
    exp = Experiment(flow_params)

    # Run for the specified number of rollouts.
    exp.run(args.num_runs, convert_to_csv=args.convert_to_csv)
