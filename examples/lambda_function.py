import boto3
from urllib.parse import unquote_plus
from examples.data_pipeline import AthenaQuery
from examples.query import tags

s3 = boto3.client('s3')
queryEngine = AthenaQuery()


def lambda_handler(event, context):
    for record in event['Records']:
        bucket = record['s3']['bucket']['name']
        key = unquote_plus(record['s3']['object']['key'])
        partition = key.split('/')[-2].split('=')[-1]
        response = s3.head_object(Bucket=bucket, Key=key)
        run_query = response["Metadata"]["run-query"]

        if bucket == 'brent.experiments' and 'trajectory-output/' in key:
            if run_query == "all":
                query_list = tags["analysis"]
            elif not run_query:
                break
            else:
                query_list = run_query.split("\', \'")
            for query_name in query_list:
                queryEngine.run_query(query_name, 's3://brent.experiments/query-result/auto/', partition)