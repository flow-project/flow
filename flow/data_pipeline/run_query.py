"""runner script for invoking query manually."""
import argparse
from flow.data_pipeline.data_pipeline import AthenaQuery, test_sql_query
from flow.data_pipeline.query import QueryStrings

parser = argparse.ArgumentParser(prog="run_query", description="runs query on AWS Athena and stores the result to"
                                                               "a S3 location")
parser.add_argument("--run", type=str, nargs="+")
parser.add_argument("--result_location", type=str, nargs='?', default="s3://circles.data.pipeline/query-result/")
parser.add_argument("--partition", type=str, nargs='?', default="default")
parser.add_argument("--list_partitions", action="store_true")
parser.add_argument("--check_status", type=str, nargs='+')
parser.add_argument("--list_queries", action="store_true")
parser.add_argument("--test_query", nargs=1)


if __name__ == "__main__":
    args = parser.parse_args()
    queryEngine = AthenaQuery()

    if args.run:
        execution_ids = []
        for query_name in args.run:
            execution_ids.append(queryEngine.run_query(query_name, args.result_location, args.partition))
        print(execution_ids)
    if args.list_partitions:
        print(queryEngine.existing_partitions)
    if args.check_status:
        status = dict()
        for execution_id in args.check_status:
            status[execution_id] = queryEngine.check_status(execution_id)
        print(status)
    if args.list_queries:
        for q in QueryStrings:
            print(q)
    if args.test_query:
        test_sql_query(args.test_query[0])
