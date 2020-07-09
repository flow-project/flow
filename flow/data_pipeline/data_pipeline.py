"""contains class and helper functions for the data pipeline."""
import pandas as pd
import boto3
from flow.data_pipeline.query import QueryStrings
from time import time
from datetime import date
import csv
from io import StringIO


def generate_trajectory_table(data_path, extra_info, partition_name):
    """Generate desired output for the trajectory_table based on standard SUMO emission.

    Parameters
    ----------
    data_path : str
        path to the standard SUMO emission
    extra_info : dict
        extra information needed in the trajectory table, collected from flow
    partition_name : str
        the name of the partition to put this output to

    Returns
    -------
    output_file_path : str
        the local path of the outputted csv file
    """
    raw_output = pd.read_csv(data_path, index_col=["time", "id"])
    required_cols = {"time", "id", "speed", "x", "y"}
    raw_output = raw_output.drop(set(raw_output.columns) - required_cols, axis=1)

    extra_info = pd.DataFrame.from_dict(extra_info)
    extra_info.set_index(["time", "id"])
    raw_output = raw_output.merge(extra_info, how="left", left_on=["time", "id"], right_on=["time", "id"])

    # add the partition column
    # raw_output['partition'] = partition_name
    raw_output = raw_output.sort_values(by=["time", "id"])
    output_file_path = data_path[:-4]+"_trajectory.csv"
    raw_output.to_csv(output_file_path, index=False)
    return output_file_path


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


def upload_to_s3(bucket_name, bucket_key, file_path, metadata={}):
    """Upload a file to S3 bucket.

    Parameters
    ----------
    bucket_name : str
        the bucket to upload to
    bucket_key: str
        the key within the bucket for the file
    file_path: str
        the path of the file to be uploaded
    metadata: dict
        all the metadata that should be attached to this simulation
    """
    s3 = boto3.resource("s3")
    s3.Bucket(bucket_name).upload_file(file_path, bucket_key,
                                       ExtraArgs={"Metadata": metadata})
    return


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
        extra_info["target_accel_with_noise_with_failsafe"].append(veh_kernel.get_accel(vid))
        extra_info["target_accel_no_noise_no_failsafe"].append(
            veh_kernel.get_accel_no_noise_no_failsafe(vid))
        extra_info["target_accel_with_noise_no_failsafe"].append(
            veh_kernel.get_accel_with_noise_no_failsafe(vid))
        extra_info["target_accel_no_noise_with_failsafe"].append(
            veh_kernel.get_accel_no_noise_with_failsafe(vid))
        extra_info["realized_accel"].append(veh_kernel.get_realized_accel(vid))
        extra_info["road_grade"].append(veh_kernel.get_road_grade(vid))
        extra_info["edge_id"].append(veh_kernel.get_edge(vid))
        extra_info["lane_id"].append(veh_kernel.get_lane(vid))
        extra_info["distance"].append(veh_kernel.get_distance(vid))
        extra_info["relative_position"].append(veh_kernel.get_position(vid))
        extra_info["source_id"].append(source_id)
        extra_info["run_id"].append(run_id)


def get_configuration(submitter_name=None, strategy_name=None):
    """Get configuration for the metadata table."""
    try:
        config_df = pd.read_csv('./data_pipeline_config')
    except FileNotFoundError:
        config_df = pd.DataFrame(data={"submitter_name": [""], "strategy": [""]})

    if not config_df['submitter_name'][0]:
        if submitter_name:
            name = submitter_name
        else:
            name = input("Please enter your name:").strip()
            while not name:
                name = input("Please enter a non-empty name:").strip()
        config_df['submitter_name'] = [name]

    if strategy_name:
        strategy = strategy_name
    else:
        strategy = input(
            "Please enter strategy name (current: \"{}\"):".format(config_df["strategy"][0])).strip()
    if strategy:
        config_df['strategy'] = [strategy]

    config_df.to_csv('./data_pipeline_config', index=False)

    return config_df['submitter_name'][0], config_df['strategy'][0]


def delete_obsolete_data(s3, latest_key, table, bucket="circles.data.pipeline"):
    """Delete the obsolete data on S3."""
    response = s3.list_objects_v2(Bucket=bucket)
    keys = [e["Key"] for e in response["Contents"] if e["Key"].find(table) == 0 and e["Key"][-4:] == ".csv"]
    keys.remove(latest_key)
    for key in keys:
        s3.delete_object(Bucket=bucket, Key=key)


def update_baseline(s3, baseline_network, baseline_source_id):
    """Update baseline data on S3."""
    obj = s3.get_object(Bucket='circles.data.pipeline', Key='baseline_table/baselines.csv')['Body']
    original_str = obj.read().decode()
    reader = csv.DictReader(StringIO(original_str))
    new_str = StringIO()
    writer = csv.DictWriter(new_str, fieldnames=['network', 'source_id'])
    writer.writeheader()
    writer.writerow({'network': baseline_network, 'source_id': baseline_source_id})
    for row in reader:
        if row['network'] != baseline_network:
            writer.writerow(row)
    s3.put_object(Bucket='circles.data.pipeline', Key='baseline_table/baselines.csv',
                  Body=new_str.getvalue().replace('\r', '').encode())


class AthenaQuery:
    """Class used to run queries.

    Act as a query engine, maintains an open session with AWS Athena.

    Attributes
    ----------
    MAX_WAIT : int
        maximum number of seconds to wait before declares time-out
    client : boto3.client
        the athena client that is used to run the query
    existing_partitions : list
        a list of partitions that is already recorded in Athena's datalog,
        this is obtained through query at the initialization of this class
        instance.
    """

    def __init__(self):
        """Initialize AthenaQuery instance.

        initialize a client session with AWS Athena,
        query Athena to obtain extisting_partition.
        """
        self.MAX_WAIT = 60
        self.client = boto3.client("athena")
        self.existing_partitions = {}

    def get_existing_partitions(self, table):
        """Return the existing partitions in the S3 bucket.

        Returns
        -------
        partitions: a list of existing partitions on S3 bucket
        """
        response = self.client.start_query_execution(
            QueryString='SHOW PARTITIONS {}'.format(table),
            QueryExecutionContext={
                'Database': 'circles'
            },
            WorkGroup='primary'
        )
        if self.wait_for_execution(response['QueryExecutionId']):
            raise RuntimeError("get current partitions timed out")
        response = self.client.get_query_results(
            QueryExecutionId=response['QueryExecutionId'],
            MaxResults=1000
        )
        return [data['Data'][0]['VarCharValue'] for data in response['ResultSet']['Rows']]

    def check_status(self, execution_id):
        """Return the status of the execution with given id.

        Parameters
        ----------
        execution_id : string
            id of the execution that is checked for
        Returns
        -------
        status: str
            QUEUED|RUNNING|SUCCEEDED|FAILED|CANCELLED
        """
        response = self.client.get_query_execution(
            QueryExecutionId=execution_id
        )
        return response['QueryExecution']['Status']['State']

    def wait_for_execution(self, execution_id):
        """Wait for the execution to finish or time-out.

        Parameters
        ----------
        execution_id : str
            id of the execution this is watiing for
        Returns
        -------
        time_out: bool
            True if time-out, False if success
        Raises
        ------
            RuntimeError: if execution failed or get canceled
        """
        start = time()
        while time() - start < self.MAX_WAIT:
            state = self.check_status(execution_id)
            if state == 'FAILED' or state == 'CANCELLED':
                raise RuntimeError("update partition failed")
            elif state == 'SUCCEEDED':
                return False
        return True

    def update_partition(self, table, submission_date, partition):
        """Load the given partition to the trajectory_table on Athena.

        Parameters
        ----------
        table : str
            the name of the table to update
        submission_date : str
            the new partition date that needs to be loaded
        partition : str
            the new partition that needs to be loaded
        """
        response = self.client.start_query_execution(
            QueryString=QueryStrings['UPDATE_PARTITION'].value.format(table=table, date=submission_date,
                                                                      partition=partition),
            QueryExecutionContext={
                'Database': 'circles'
            },
            WorkGroup='primary'
        )
        if self.wait_for_execution(response['QueryExecutionId']):
            raise RuntimeError("update partition timed out")
        self.existing_partitions[table].append("date={}/partition_name={}".format(submission_date, partition))
        return

    def repair_partition(self, table, submission_date, partition):
        """Load the missing partitions."""
        if table not in self.existing_partitions.keys():
            self.existing_partitions[table] = self.get_existing_partitions(table)
        if "date={}/partition_name={}".format(submission_date, partition) not in \
                self.existing_partitions[table]:
            self.update_partition(table, submission_date, partition)

    def run_query(self, query_name, result_location="s3://circles.data.pipeline/result/",
                  submission_date="today", partition="default", **kwargs):
        """Start the execution of a query, does not wait for it to finish.

        Parameters
        ----------
        query_name : str
            name of the query in QueryStrings enum that will be run
        result_location: str, optional
            location on the S3 bucket where the result will be stored
        submission_date : str
            name of the partition date to run this query on
        partition: str, optional
            name of the partition to run this query on
        Returns
        -------
        execution_id: str
            the execution id of the execution started by this method
        Raises
        ------
            ValueError: if tries to run a query not existed in QueryStrings enum
        """
        if query_name not in QueryStrings.__members__:
            raise ValueError("query not existed: please add it to query.py")

        if submission_date == "today":
            submission_date = date.today().isoformat()

        source_id = "flow_{}".format(partition.split('_')[1])

        response = self.client.start_query_execution(
            QueryString=QueryStrings[query_name].value.format(date=submission_date, partition=source_id, **kwargs),
            QueryExecutionContext={
                'Database': 'circles'
            },
            ResultConfiguration={
                'OutputLocation': result_location,
            },
            WorkGroup='primary'
        )
        return response['QueryExecutionId']
