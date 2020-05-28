"""APIs for the leader board front end"""
import os
import boto3
import pandas as pd
from io import StringIO


def key_to_name(key):
    """Return the standard formatted file name from object key."""
    k_list = key.split("/")
    date = k_list[1].replace("date=", "")
    name = k_list[2].replace("partition_name=", "")
    index = name.find("_", 5)
    source_id = name
    query_name = ""
    if index != -1:
        source_id = name[0:index]
        query_name = "_" + name[index+1:].replace("_", "-")
    return "{}_{}{}.csv".format(date, source_id.replace("_", "-"), query_name)


def get_table_disk(table_name="fact_vehicle_trace", bucket="circles.data.pipeline"):
    """Fetch tables from s3 and store in ./result directory.

    Parameters
    ----------
    table_name: str
        The name of table to retrieve from S3, the current available tables are:
            fact_vehicle_trace
            fact_energy_trace
            fact_network_throughput_agg
            fact_network_inflows_outflows
            fact_vehicle_fuel_efficiency_agg
            fact_network_metrics_by_distance_agg
            fact_network_metrics_by_time_agg
            fact_network_fuel_efficiency_agg
            leaderboard_chart
            leaderboard_chart_agg
        Note that leaderboard_chart_agg is a combination of all previous
        learderboard_chart entries in one CSV file. It's only used to
        avoid burdening the web server with more calculation. The date
        and source_id in its name is always going to reflect the latest
        leaderboard_chart entry.
    bucket: str
        the S3 bucket that holds these tables
    """
    try:
        os.makedirs("result/{}".format(table_name))
    except FileExistsError as e:
        pass
    s3 = boto3.client("s3")
    response = s3.list_objects_v2(Bucket=bucket)
    keys = [e["Key"] for e in response["Contents"] if e["Key"].find(table_name) == 0 and e["Key"][-4:] == ".csv"]
    names = [key_to_name(k) for k in keys]
    existing_results = os.listdir("./result/{}".format(table_name))
    if table_name == "leaderboard_chart_agg":
        for p in existing_results:
            os.remove("./result/{}/{}".format(table_name, p))
    for index in range(len(keys)):
        if names[index] not in existing_results:
            s3.download_file(bucket, keys[index], "./result/{}/{}".format(table_name, names[index]))


def get_table_memory(table_name="fact_vehicle_trace", bucket="circles.data.pipeline", existing_results=()):
    """Fetch tables from s3 and return them as in-memory pandas dataframe objects.

    Parameters
    ----------
    bucket: str
        the S3 bucket that holds the tables
    table_name: str
        the name of the name to retrieve from S3, for detail see get_table_disk
    existing_results: list
        tables that should not be fetched,
        the names must follow the convention:
        {date}_{source_id(no run number)}_{query_name}.csv

    Returns
    -------
    file_list: dict
        a dictionary of pandas dataframes, each contains a table from S3
        The dataframs are keyed by their name: {source_id(no run number)}_{query_name}.csv

    """
    s3 = boto3.client("s3")
    response = s3.list_objects_v2(Bucket=bucket)
    keys = [e["Key"] for e in response["Contents"] if e["Key"].find(table_name) == 0 and e["Key"][-4:] == ".csv"]
    names = [key_to_name(k) for k in keys]
    results = dict()
    for index in range(len(keys)):
        if names[index] not in existing_results:
            obj = s3.get_object(Bucket=bucket, Key=keys[index])["Body"]
            obj_str = obj.read().decode("utf-8")
            results[names[index]] = pd.read_csv(StringIO(obj_str))
    return results


def get_table_url(table_name="fact_vehicle_trace", bucket="circles.data.pipeline", existing_results=()):
    """Fetch tables from s3 and return as urls, requires the bucket to have public access.

    Parameters
    ----------
    bucket: str
        the S3 bucket that holds the tables
    table_name: str
        the name of the name to retrieve from S3, for detail see get_table_disk
    existing_results: list
        tables that should not be fetched,
        the names must follow the convention:
        {date}_{source_id(no run number)}_{query_name}.csv

    Returns
    -------
    file_list: dict
        a dictionary of urls, each contains a table from S3
        The urls are keyed by their name: {source_id(no run number)}_{query_name}.csv

    """
    s3 = boto3.client("s3")
    response = s3.list_objects_v2(Bucket=bucket)
    keys = [e["Key"] for e in response["Contents"] if e["Key"].find(table_name) == 0 and e["Key"][-4:] == ".csv"]
    names = [key_to_name(k) for k in keys]
    results = dict()
    for index in range(len(keys)):
        if names[index] not in existing_results:
            results[names[index]] = "https://{}.s3.{}.amazonaws.com/{}".format(bucket, "us-west-2", keys[index])
    return results


def get_metadata(name, bucket="circles.data.pipeline"):
    """Get the metadata by name.

    Parameters
    ----------
    name: str
        the name of the table whose metadata will be returned
    bucket: str
        the bucket that hold the table

    Returns
    -------
    metadata: dict
        a dictionary of all the metadata, there is no guarantee
        for which keys are included
    """
    s3 = boto3.client("s3")
    name_list = name.split('_')
    source_id = "flow_{}".format(name_list[2])
    response = s3.head_object(Bucket=bucket,
                              Key="fact_vehicle_trace/date={0}/partition_name={1}/{1}.csv".format(name_list[0],
                                                                                                   source_id))
    return response["Metadata"]
