from enum import Enum
from examples.datapipeline_test import apply_energy_one

tags = {"energy": ["ENERGY_ONE"]}

testing_functions = {"ENERGY_ONE": apply_energy_one}


class QueryStrings(Enum):
    SAMPLE = "SELECT * FROM trajectory_table WHERE partition_name=\'{partition}\' LIMIT 15;"
    UPDATE_PARTITION = "ALTER TABLE trajectory_table ADD IF NOT EXISTS PARTITION (partition_name=\'{partition}\');"
    ENERGY_ONE = "SELECT id, time, 1200 * speed * " \
                 "((CASE WHEN acceleration > 0 THEN 1 ELSE 0 END * (1-0.8) * acceleration) + 0.8 " \
                 "+ 9.81 * SIN(road_grade)) + 1200 * 9.81 * 0.005 * speed + 0.5 * 1.225 * 2.6 * 0.3 " \
                 "* POW(speed,3) AS power, 1 AS energy_model_id, source_id " \
                 "FROM trajectory_table " \
                 "WHERE partition_name=\'{partition}\'"