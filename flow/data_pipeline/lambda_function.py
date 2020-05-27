"""lambda function on AWS Lambda."""
import boto3
from urllib.parse import unquote_plus
from flow.data_pipeline.data_pipeline import AthenaQuery
from flow.data_pipeline.query import tags, tables

s3 = boto3.client('s3')
queryEngine = AthenaQuery()


def lambda_handler(event, context):
    """Handle S3 put event on AWS Lambda."""
    records = []
    # delete all unwanted metadata
    for record in event['Records']:
        bucket = record['s3']['bucket']['name']
        key = unquote_plus(record['s3']['object']['key'])
        table = key.split('/')[0]
        if table not in tables:
            continue
        if key[-9:] == '.metadata':
            s3.delete_object(Bucket=bucket, Key=key)
            continue
        if table in tags.keys():
            records.append((bucket, key, table))
    # initialize the queries
    for bucket, key, table in records:
        query_date = key.split('/')[-3].split('=')[-1]
        partition = key.split('/')[-2].split('=')[-1]
        source_id = "flow_{}".format(partition.split('_')[1])
        # response = s3.head_object(Bucket=bucket, Key=key)
        # required_query = response["Metadata"]["run-query"]

        query_dict = tags[table]

        # handle different energy models
        if table == "fact_energy_trace":
            energy_model_id = partition.replace(source_id, "")[1:]
            query_dict = tags[energy_model_id]

        # initialize queries and store them at appropriate locations
        for table_name, query_list in query_dict.items():
            for query_name in query_list:
                result_location = 's3://circles.data.pipeline/{}/date={}/partition_name={}_{}'.format(table_name,
                                                                                                      query_date,
                                                                                                      source_id,
                                                                                                      query_name)
                queryEngine.run_query(query_name, result_location, query_date, partition, table)
