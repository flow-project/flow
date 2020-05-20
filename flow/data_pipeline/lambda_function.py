"""lambda function on AWS Lambda."""
import boto3
from urllib.parse import unquote_plus
from flow.data_pipeline.data_pipeline import AthenaQuery
from flow.data_pipeline.query import tags

s3 = boto3.client('s3')
queryEngine = AthenaQuery()


def lambda_handler(event, context):
    for record in event['Records']:
        bucket = record['s3']['bucket']['name']
        key = unquote_plus(record['s3']['object']['key'])
        query_date = key.split('/')[-3].split('=')[-1]
        partition = key.split('/')[-2].split('=')[-1]
        response = s3.head_object(Bucket=bucket, Key=key)
        required_query = response["Metadata"]["run-query"]

        if bucket == 'circles.data.pipeline' and 'trajectory-output/' in key:
            if required_query == "all":
                query_list = tags["energy"]
            elif not required_query:
                break
            else:
                query_list = required_query.split("\', \'")
            for query_name in query_list:
                queryEngine.run_query(query_name, 's3://circles.data.pipeline/result/auto/', query_date, partition)
