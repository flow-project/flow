"""lambda function on AWS Lambda."""
import boto3
from urllib.parse import unquote_plus
from flow.data_pipeline.data_pipeline import AthenaQuery, delete_obsolete_data, update_baseline, \
    get_ready_queries, get_completed_queries, put_completed_queries
from flow.data_pipeline.query import prerequisites, tables,  network_using_edge, summary_tables
from flow.data_pipeline.query import X_FILTER, EDGE_FILTER, WARMUP_STEPS, HORIZON_STEPS

s3 = boto3.client('s3')
queryEngine = AthenaQuery()


def lambda_handler(event, context):
    """Handle S3 put event on AWS Lambda."""
    # stores all lists of completed query for each source_id
    completed = {}
    records = []
    # do a pre-sweep to handle tasks other than initalizing a query
    for record in event['Records']:
        bucket = record['s3']['bucket']['name']
        key = unquote_plus(record['s3']['object']['key'])
        table = key.split('/')[0]
        if table not in tables:
            continue
        # delete unwanted metadata files
        if key[-9:] == '.metadata':
            s3.delete_object(Bucket=bucket, Key=key)
            continue
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
        if query_name in prerequisites.keys():
            records.append((bucket, key, table, query_name, query_date, partition, source_id))

    # initialize the queries
    start_filter = WARMUP_STEPS
    stop_filter = WARMUP_STEPS + HORIZON_STEPS
    for bucket, key, table, query_name, query_date, partition, source_id in records:
        # add this query to the list of completed queries for this source_id
        if source_id not in completed.keys():
            completed[source_id] = get_completed_queries(s3, source_id)
        completed[source_id].append(query_name)
        # retrieve metadata and use it to determine the right loc_filter
        metadata_key = "fact_vehicle_trace/date={0}/partition_name={1}/{1}.csv".format(query_date, source_id)
        response = s3.head_object(Bucket=bucket, Key=metadata_key)
        loc_filter = X_FILTER
        if 'network' in response["Metadata"]:
            if response["Metadata"]['network'] in network_using_edge:
                loc_filter = EDGE_FILTER
            # update baseline if needed
            if table == 'fact_vehicle_trace' \
                    and 'is_baseline' in response['Metadata'] and response['Metadata']['is_baseline'] == 'True':
                update_baseline(s3, response["Metadata"]['network'], source_id)

        readied_queries = get_ready_queries(completed[source_id])
        # initialize queries and store them at appropriate locations
        for readied_query_name, table_name in readied_queries:
            result_location = 's3://circles.data.pipeline/{}/date={}/partition_name={}_{}'.format(table_name,
                                                                                                  query_date,
                                                                                                  source_id,
                                                                                                  readied_query_name)
            queryEngine.run_query(readied_query_name, result_location, query_date, partition, loc_filter=loc_filter,
                                  start_filter=start_filter, stop_filter=stop_filter)
    # stores all the updated lists of completed queries back to S3
    put_completed_queries(s3, completed)
