"""lambda function on AWS Lambda."""
import boto3
import json
from urllib.parse import unquote_plus
from flow.data_pipeline.data_pipeline import AthenaQuery, delete_obsolete_data, update_baseline, \
    get_ready_queries, get_completed_queries, put_completed_queries
from flow.data_pipeline.query import tables, network_filters, summary_tables, triggers, max_decel, leader_max_decel

s3 = boto3.client('s3')
queryEngine = AthenaQuery()
sqs = boto3.client('sqs')


def lambda_handler(event, context):
    """Handle S3 put event on AWS Lambda."""
    # stores all lists of completed query for each source_id
    completed = {}
    records = []
    event_records = []
    # do a pre-sweep to put all s3 records in one list
    for event_record in event['Records']:
        if event_record["eventSource"] == "aws:s3":
            event_records.append(event_record)
        elif event_record['eventSource'] == "aws:sqs":
            s3_event = json.loads(event_record['body'])
            event_records.extend(s3_event['Records'])
    # do a pre-sweep to handle tasks other than initalizing a query
    for record in event_records:
        bucket = record['s3']['bucket']['name']
        key = unquote_plus(record['s3']['object']['key'])
        table = key.split('/')[0]
        if table not in tables:
            continue
        # delete unwanted metadata files
        s3.delete_object(Bucket=bucket, Key=(key + '.metadata'))
        # load the partition for newly added table
        query_date = key.split('/')[-3].split('=')[-1]
        partition = key.split('/')[-2].split('=')[-1]
        source_id = "flow_{}".format(partition.split('_')[1])
        if table == "fact_vehicle_trace":
            query_name = "FACT_VEHICLE_TRACE"
        else:
            query_name = partition.replace(source_id, "")[1:]
        queryEngine.repair_partition(table, query_date, partition)
        # delete obsolete data
        if table in summary_tables:
            delete_obsolete_data(s3, key, table)
        # add table that need to start a query to list
        if query_name in triggers:
            records.append((bucket, key, table, query_name, query_date, partition, source_id))

    # initialize the queries
    for bucket, key, table, query_name, query_date, partition, source_id in records:
        # retrieve the set of completed query for this source_id if not already available
        if source_id not in completed.keys():
            completed[source_id] = get_completed_queries(s3, source_id)
        # if query already recorded before, skip it. This is to tolerate repetitive execution by Lambda
        if query_name in completed[source_id]:
            continue
        # retrieve metadata and use it to determine the right location filters
        metadata_key = "fact_vehicle_trace/date={0}/partition_name={1}/{1}.csv".format(query_date, source_id)
        response = s3.head_object(Bucket=bucket, Key=metadata_key)
        if 'network' in response["Metadata"]:
            network = response["Metadata"]['network']
            inflow_filter = network_filters[network]['inflow_filter']
            outflow_filter = network_filters[network]['outflow_filter']
            start_filter = network_filters[network]['warmup_steps']

            # update baseline if needed
            if table == 'fact_vehicle_trace' \
                    and 'is_baseline' in response['Metadata'] and response['Metadata']['is_baseline'] == 'True':
                update_baseline(s3, network, source_id)

        readied_queries = get_ready_queries(completed[source_id], query_name)
        completed[source_id].add(query_name)
        # stores the updated list of completed queries back to S3
        put_completed_queries(s3, source_id, completed[source_id])
        # initialize queries and store them at appropriate locations
        for readied_query_name, table_name in readied_queries:
            result_location = 's3://circles.data.pipeline/{}/date={}/partition_name={}_{}'.format(table_name,
                                                                                                  query_date,
                                                                                                  source_id,
                                                                                                  readied_query_name)
            message_body = (readied_query_name, result_location, query_date, partition, inflow_filter, outflow_filter,
                            start_filter, max_decel, leader_max_decel)
            message_body = json.dumps(message_body)
            sqs.send_message(
                QueueUrl="",
                MessageBody=message_body)
