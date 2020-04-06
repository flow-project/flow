import pandas as pd
import boto3
from botocore.exceptions import ClientError
from examples.query import QueryStrings
from time import time


def generate_trajectory_table(data_path, extra_info, partition_name):
    """ generate desired output for the trajectory_table based on standard SUMO emission

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
    raw_output['partition'] = partition_name

    output_file_path = data_path[:-4]+"_trajectory.csv"
    raw_output.to_csv(output_file_path, index=False)
    return output_file_path


def upload_to_s3(bucket_name, bucket_key, file_path):
    """ upload a file to S3 bucket

    Parameters
    ----------
    bucket_name : str
        the bucket to upload to
    bucket_key: str
        the key within the bucket for the file
    file_path: str
        the path of the file to be uploaded
    """
    s3 = boto3.resource("s3")
    s3.Bucket(bucket_name).upload_file(file_path, bucket_key)
    return


class AthenaQuery:

    def __init__(self):
        self.MAX_WAIT = 60
        self.client = boto3.client("athena")
        self.existing_partitions = self.get_existing_partitions()

    def get_existing_partitions(self):
        """prints the existing partitions in the S3 bucket"""

        response = self.client.start_query_execution(
            QueryString='SHOW PARTITIONS trajectory_table',
            QueryExecutionContext={
                'Database': 'simulation'
            },
            WorkGroup='primary'
        )
        if self.wait_for_execution(response['QueryExecutionId']):
            raise RuntimeError("get current partitions timed out")
        response = self.client.get_query_results(
            QueryExecutionId=response['QueryExecutionId'],
            MaxResults=1000
        )
        return [data['Data'][0]['VarCharValue'].split('=')[-1] for data in response['ResultSet']['Rows']]

    def check_status(self, execution_id):
        """ Return the status of the execution with given id

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
        """ wait for the execution to finish or time-out

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

    def update_partition(self, partition):
        """ load the given partition to the trajectory_table on Athena

        Parameters
        ----------
        partition : str
            the new partition that needs to be loaded
        """
        response = self.client.start_query_execution(
            QueryString=QueryStrings['UPDATE_PARTITION'].value.format(partition=partition),
            QueryExecutionContext={
                'Database': 'simulation'
            },
            WorkGroup='primary'
        )
        if self.wait_for_execution(response['QueryExecutionId']):
            raise RuntimeError("update partition timed out")
        self.existing_partitions.append(partition)
        return

    def run_query(self, query_name, result_location="s3://brent.experiments/query-result/", partition="default"):
        """ start the execution of a query, does not wait for it to finish

        Parameters
        ----------
        query_name : str
            name of the query in QueryStrings enum that will be run
        result_location: str, optional
            location on the S3 bucket where the result will be stored
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

        if partition not in self.existing_partitions:
            self.update_partition(partition)

        response = self.client.start_query_execution(
            QueryString=QueryStrings[query_name].value.format(partition=partition),
            QueryExecutionContext={
                'Database': 'simulation'
            },
            ResultConfiguration={
                'OutputLocation': result_location,
            },
            WorkGroup='primary'
        )
        return response['QueryExecutionId']