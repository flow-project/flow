"""contains class and helper functions for the data pipeline."""
import pandas as pd


def write_dict_to_csv(data_path, extra_info, include_header=False):
    """Write extra to the CSV file at data_path, create one if not exist.

    Parameters
    ----------
    data_path : str
        output file path
    extra_info: dict
        extra information needed in the trajectory table, collected from flow
    include_header: bool
        whether or not to include the header in the output, this should be set to
        True for the first write to the a empty or newly created CSV, and set to
        False for subsequent appends.
    """
    extra_info = pd.DataFrame.from_dict(extra_info)
    extra_info.to_csv(data_path, mode='a+', index=False, header=include_header)


def get_extra_info(veh_kernel, extra_info, veh_ids, source_id, run_id):
    """Get all the necessary information for the trajectory output from flow."""
    for vid in veh_ids:
        extra_info["time_step"].append(veh_kernel.get_timestep(vid) / 1000)
        extra_info["id"].append(vid)
        position = veh_kernel.get_2d_position(vid)
        extra_info["x"].append(position[0])
        extra_info["y"].append(position[1])
        extra_info["speed"].append(veh_kernel.get_speed(vid))
        extra_info["headway"].append(veh_kernel.get_headway(vid))
        extra_info["leader_id"].append(veh_kernel.get_leader(vid))
        extra_info["follower_id"].append(veh_kernel.get_follower(vid))
        extra_info["leader_rel_speed"].append(veh_kernel.get_speed(
            veh_kernel.get_leader(vid)) - veh_kernel.get_speed(vid))
        extra_info["target_accel_with_noise_with_failsafe"].append(
            veh_kernel.get_accel(vid, noise=True, failsafe=True))
        extra_info["target_accel_no_noise_no_failsafe"].append(
            veh_kernel.get_accel(vid, noise=False, failsafe=False))
        extra_info["target_accel_with_noise_no_failsafe"].append(
            veh_kernel.get_accel(vid, noise=True, failsafe=False))
        extra_info["target_accel_no_noise_with_failsafe"].append(
            veh_kernel.get_accel(vid, noise=False, failsafe=True))
        extra_info["realized_accel"].append(veh_kernel.get_realized_accel(vid))
        extra_info["road_grade"].append(veh_kernel.get_road_grade(vid))
        extra_info["edge_id"].append(veh_kernel.get_edge(vid))
        extra_info["lane_id"].append(veh_kernel.get_lane(vid))
        extra_info["distance"].append(veh_kernel.get_distance(vid))
        extra_info["relative_position"].append(veh_kernel.get_position(vid))
        extra_info["source_id"].append(source_id)
        extra_info["run_id"].append(run_id)
