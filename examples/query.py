from enum import Enum

tags = {}


class QueryStrings(Enum):
    SAMPLE = "SELECT * FROM trajectory_table WHERE partition_name=\'{partition}\' LIMIT 15;"
    UPDATE_PARTITION = "ALTER TABLE trajectory_table ADD IF NOT EXISTS PARTITION (partition_name=\'{partition}\');"