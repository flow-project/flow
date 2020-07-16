"""contains class and helper functions for the data pipeline."""
import pandas as pd
import boto3
from botocore.exceptions import ClientError
from flow.data_pipeline.query import QueryStrings, prerequisites, tables
from time import time
from datetime import date
import csv
from io import StringIO
import json
import collections
from collections import defaultdict


def generate_trajectory_table(emission_files, trajectory_table_path, source_id):
    """Generate desired output for the trajectory_table based on SUMO emissions.

    Parameters
    ----------
    emission_files : list
        paths to the SUMO emission
    trajectory_table_path : str
        path to the file for S3 upload only
    source_id : str
        a unique id for the simulation that generate these emissions
    """
    for i in range(len(emission_files)):
        emission_output = pd.read_csv(emission_files[i])
        emission_output['source_id'] = source_id
        emission_output['run_id'] = "run_{}".format(i)
        # add header row to the file only at the first run (when i==0)
        emission_output.to_csv(trajectory_table_path, mode='a+', index=False, header=(i == 0))


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


def get_configuration():
    """Get configuration for the metadata table."""
    try:
        config_df = pd.read_csv('./data_pipeline_config')
    except FileNotFoundError:
        config_df = pd.DataFrame(data={"submitter_name": [""], "strategy": [""]})

    if not config_df['submitter_name'][0]:
        name = input("Please enter your name:").strip()
        while not name:
            name = input("Please enter a non-empty name:").strip()
        config_df['submitter_name'] = [name]

    strategy = input(
        "Please enter strategy name (current: \"{}\"):".format(config_df["strategy"][0])).strip()
    if strategy:
        config_df['strategy'] = [strategy]

    config_df.to_csv('./data_pipeline_config', index=False)

    return config_df['submitter_name'][0], config_df['strategy'][0]


def delete_obsolete_data(s3, latest_key, table, bucket="circles.data.pipeline"):
    """Delete the obsolete data on S3."""
    keys = list_object_keys(s3, bucket=bucket, prefixes=table, suffix='.csv')
    keys.remove(latest_key)
    for key in keys:
        s3.delete_object(Bucket=bucket, Key=key)


def update_baseline(s3, baseline_network, baseline_source_id):
    """Update the baseline table on S3 if new baseline run is added."""
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


def get_completed_queries(s3, source_id):
    """Return the deserialized list of completed queries from S3."""
    try:
        completed_queries_obj = \
            s3.get_object(Bucket='circles.data.pipeline', Key='lambda_temp/{}'.format(source_id))['Body']
        completed_queries = json.loads(completed_queries_obj.read().decode('utf-8'))
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            completed_queries = set()
        else:
            raise
    return set(completed_queries)


def put_completed_queries(s3, source_id, completed_queries_set):
    """Put the completed queries list into S3 as in a serialized json format."""
    completed_queries_list = list(completed_queries_set)
    completed_queries_json = json.dumps(completed_queries_list)
    s3.put_object(Bucket='circles.data.pipeline', Key='lambda_temp/{}'.format(source_id),
                  Body=completed_queries_json.encode('utf-8'))


def get_ready_queries(completed_queries, new_query):
    """Return queries whose prerequisite queries are completed."""
    readied_queries = []
    unfinished_queries = prerequisites.keys() - completed_queries
    upadted_completed_queries = completed_queries.copy()
    upadted_completed_queries.add(new_query)
    for query_name in unfinished_queries:
        if not prerequisites[query_name][1].issubset(completed_queries):
            if prerequisites[query_name][1].issubset(upadted_completed_queries):
                readied_queries.append((query_name, prerequisites[query_name][0]))
    return readied_queries


def list_object_keys(s3, bucket='circles.data.pipeline', prefixes='', suffix=''):
    """Return all keys in the given bucket that start with prefix and end with suffix. Not limited by 1000."""
    contents = []
    if not isinstance(prefixes, collections.Iterable) or type(prefixes) is str:
        prefixes = [prefixes]
    for prefix in prefixes:
        response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
        if 'Contents' in response:
            contents.extend(response['Contents'])
        while response['IsTruncated']:
            response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix,
                                          ContinuationToken=response['NextContinuationToken'])
            contents.extend(response['Contents'])
    keys = [content['Key'] for content in contents if content['Key'].endswith(suffix)]
    return keys


def delete_table(s3, bucket='circles.data.pipeline', only_query_result=True, table='', source_id=''):
    """Deletes the specified the table files in S3"""
    queries = ["lambda_temp"]
    if table:
        queries.append(table)
    else:
        queries = tables
        if only_query_result:
            queries.remove('fact_vehicle_trace')
            queries.remove('metadata_table')
        if source_id:
            queries.remove('leaderboard_chart_agg')
            queries.remove('fact_top_scores')
    keys = list_object_keys(s3, bucket=bucket, prefixes=queries)
    if source_id:
        keys = [key for key in keys if source_id in key]
    for key in keys:
        s3.delete_object(Bucket=bucket, Key=key)


def rerun_query(s3, bucket='circles.data.pipeline', source_id=''):
    """Re-run queries for simulation datas that has been uploaded to s3, will delete old data before re-run."""
    vehicle_trace_keys = list_object_keys(s3, bucket=bucket, prefixes="fact_vehicle_trace", suffix='.csv')
    delete_table(s3, bucket=bucket, source_id=source_id)
    if source_id:
        vehicle_trace_keys = [key for key in vehicle_trace_keys if source_id in key]
    sqs_client = boto3.client('sqs')
    event_template = """
        {{
          "Records": [
            {{
              "eventVersion": "2.0",
              "eventSource": "aws:s3",
              "awsRegion": "us-west-2",
              "eventTime": "1970-01-01T00:00:00.000Z",
              "eventName": "ObjectCreated:Put",
              "userIdentity": {{
                "principalId": "EXAMPLE"
              }},
              "requestParameters": {{
                "sourceIPAddress": "127.0.0.1"
              }},
              "responseElements": {{
                "x-amz-request-id": "EXAMPLE123456789",
                "x-amz-id-2": "EXAMPLE123/5678abcdefghijklambdaisawesome/mnopqrstuvwxyzABCDEFGH"
              }},
              "s3": {{
                "s3SchemaVersion": "1.0",
                "configurationId": "testConfigRule",
                "bucket": {{
                  "name": "{bucket}",
                  "ownerIdentity": {{
                    "principalId": "EXAMPLE"
                  }},
                  "arn": "arn:aws:s3:::{bucket}"
                }},
                "object": {{
                  "key": "{key}",
                  "size": 1024,
                  "eTag": "0123456789abcdef0123456789abcdef",
                  "sequencer": "0A1B2C3D4E5F678901"
                }}
              }}
            }}
          ]
        }}"""
    for key in vehicle_trace_keys:
        response = sqs_client.send_message(QueueUrl="https://sqs.us-west-2.amazonaws.com/409746595792/S3CreateEvents",
                                           MessageBody=event_template.format(bucket=bucket, key=key))


def list_source_ids(s3, bucket='circles.data.pipeline'):
    """Return a list of the source_id of all simulations which has been uploaded to s3."""
    vehicle_trace_keys = list_object_keys(s3, bucket=bucket, prefixes="fact_vehicle_trace", suffix='csv')
    source_ids = ['flow_{}'.format(key.split('/')[2].split('=')[1].split('_')[1]) for key in vehicle_trace_keys]
    return source_ids


def sanity_check(s3, bucket='circles.data.pipeline'):
    """Check if all the expected queries get run without error. Note that this does not check the correctness of
       the content of the query, only that it finish without error."""
    queries = tables
    queries.append('lambda_temp')
    queries.remove('leaderboard_chart_agg')
    queries.remove('fact_top_scores')
    expected_count = len(queries)
    keys = list_object_keys(s3, bucket=bucket, prefixes=queries, suffix='.csv')
    source_ids = list_source_ids(s3, bucket=bucket)
    counts = defaultdict(lambda: [])
    for key in keys:
        source_id = 'flow_{}'.format(key.split('/')[2].split('=')[1].split('_')[1])
        table = key.split('/')[0]
        counts[source_id].append(table)
    for sid in source_ids:
        count = len(counts[sid])
        if count < expected_count:
            missing = []
            for q in queries:
                if q not in counts[sid]:
                    missing.append(q)
            print("Simulation {} is missing the following queries: \n    {}".format(sid, str(missing)))
        elif count > expected_count:
            extra = counts[sid].copy()
            for q in queries:
                if q not in counts[sid]:
                    extra.remove(q)
            print("Simulation {} is having too much of the following queries: \n    {}".format(sid, str(extra)))


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
