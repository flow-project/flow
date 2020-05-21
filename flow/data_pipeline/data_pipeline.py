"""contains class and helper functions for the data pipeline."""
import pandas as pd
import boto3
from flow.data_pipeline.query import QueryStrings
from time import time
from datetime import date


def generate_trajectory_table(data_path, extra_info, partition_name):
    """Generate desired output for the trajectory_table based on standard SUMO emission.

    Parameters
    ----------
    data_path : str
        path to the standard SUMO emission
    extra_info: dict
        extra information needed in the trajectory table, collected from flow
    partition_name: str
        the name of the partition to put this output to
    Returns
    -------
    output_file_path: str
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


def generate_trajectory_from_flow(data_path, extra_info, partition_name=None):
    """Generate desired output for the trajectory_table based only on flow output.

    Parameters
    ----------
    data_path : str
        output file path
    extra_info: dict
        extra information needed in the trajectory table, collected from flow
    partition_name: str
        the name of the partition to put this output to
    Returns
    -------
    output_file_path: str
        the local path of the outputted csv file that should be used for
        upload to s3 only, it does not the human readable column names and
        will be deleted after uploading to s3. A copy of this file with all
        the column name will remain in the ./data folder
    """
    extra_info = pd.DataFrame.from_dict(extra_info)
    # extra_info["partition"] = partition_name
    extra_info.to_csv(data_path, index=False)
    upload_only_file_path = data_path[:-4] + "_upload" + ".csv"
    extra_info.to_csv(upload_only_file_path, index=False, header=False)
    return upload_only_file_path


def upload_to_s3(bucket_name, bucket_key, file_path, only_query):
    """Upload a file to S3 bucket.

    Parameters
    ----------
    bucket_name : str
        the bucket to upload to
    bucket_key: str
        the key within the bucket for the file
    file_path: str
        the path of the file to be uploaded
    only_query: str
        specify which query should be run on this file by lambda:
        if empty: run none of them
        if "all": run all available analysis query
        if a string of list of queries: run only those mentioned in the list
    """
    s3 = boto3.resource("s3")
    s3.Bucket(bucket_name).upload_file(file_path, bucket_key,
                                       ExtraArgs={"Metadata": {"run-query": only_query}})
    return


def get_extra_info(veh_kernel, extra_info, veh_ids):
    """Get all the necessary information for the trajectory output from flow."""
    for vid in veh_ids:
        extra_info["time_step"].append(veh_kernel.get_timestep(vid) / 1000)
        extra_info["id"].append(vid)
        extra_info["headway"].append(veh_kernel.get_headway(vid))
        extra_info["acceleration"].append(veh_kernel.get_accel(vid))
        extra_info["leader_id"].append(veh_kernel.get_leader(vid))
        extra_info["follower_id"].append(veh_kernel.get_follower(vid))
        extra_info["leader_rel_speed"].append(veh_kernel.get_speed(
            veh_kernel.get_leader(vid)) - veh_kernel.get_speed(vid))
        extra_info["accel_without_noise"].append(veh_kernel.get_accel_without_noise(vid))
        extra_info["realized_accel"].append(veh_kernel.get_realized_accel(vid))
        extra_info["road_grade"].append(veh_kernel.get_road_grade(vid))
        position = veh_kernel.get_2d_position(vid)
        extra_info["x"].append(position[0])
        extra_info["y"].append(position[1])
        extra_info["speed"].append(veh_kernel.get_speed(vid))


class AthenaQuery:
    """
    Class used to run query.

    Act as a query engine, maintains an open session with AWS Athena.

    Attributes
    ----------
    MAX_WAIT: int
        maximum number of seconds to wait before declares time-out
    client: boto3.client
        the athena client that is used to run the query
    existing_partitions: list
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
        self.existing_partitions = self.get_existing_partitions()

    def get_existing_partitions(self):
        """Return the existing partitions in the S3 bucket.

        Returns
        -------
        partitions: a list of existing partitions on S3 bucket
        """
        response = self.client.start_query_execution(
            QueryString='SHOW PARTITIONS trajectory_table',
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

    def update_partition(self, query_date, partition):
        """Load the given partition to the trajectory_table on Athena.

        Parameters
        ----------
        query_date : str
            the new partition date that needs to be loaded
        partition : str
            the new partition that needs to be loaded
        """
        response = self.client.start_query_execution(
            QueryString=QueryStrings['UPDATE_PARTITION'].value.format(date=query_date, partition=partition),
            QueryExecutionContext={
                'Database': 'circles'
            },
            WorkGroup='primary'
        )
        if self.wait_for_execution(response['QueryExecutionId']):
            raise RuntimeError("update partition timed out")
        self.existing_partitions.append("date={}/partition_name={}".format(query_date, partition))
        return

    def run_query(self, query_name, result_location="s3://circles.data.pipeline/result/",
                  query_date="today", partition="default"):
        """Start the execution of a query, does not wait for it to finish.

        Parameters
        ----------
        query_name : str
            name of the query in QueryStrings enum that will be run
        result_location: str, optional
            location on the S3 bucket where the result will be stored
        query_date : str
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

        if query_date == "today":
            query_date = date.today().isoformat()

        if "date={}/partition_name={}".format(query_date, partition) not in self.existing_partitions:
            self.update_partition(query_date, partition)

        response = self.client.start_query_execution(
            QueryString=QueryStrings[query_name].value.format(date=query_date, partition=partition),
            QueryExecutionContext={
                'Database': 'circles'
            },
            ResultConfiguration={
                'OutputLocation': result_location,
            },
            WorkGroup='primary'
        )
        return response['QueryExecutionId']
