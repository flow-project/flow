"""stores all the pre-defined query strings."""
from enum import Enum
from flow.data_pipeline.datapipeline_test import apply_energy_one

# tags for different queries
tags = {"energy": ["POWER_DEMAND_MODEL", "POWER_DEMAND_MODEL_DENOISED_ACCEL", "POWER_DEMAND_MODEL_DENOISED_ACCEL_VEL"],
        "analysis": ["POWER_DEMAND_MODEL"]}

# specify the function to calculate the expected result of each query
testing_functions = {"POWER_DEMAND_MODEL": apply_energy_one}


class QueryStrings(Enum):
    """An enumeration of all the pre-defined query strings."""

    SAMPLE = """
        SELECT *
        FROM trajectory_table
        WHERE partition_name=\'{partition}\'
        LIMIT 15;
        """

    UPDATE_PARTITION = """
        ALTER TABLE trajectory_table
        ADD IF NOT EXISTS PARTITION (partition_name=\'{partition}\');
        """

    POWER_DEMAND_MODEL = VEHICLE_POWER_DEMAND_SUBQUERY.format('trajectory_table')

    POWER_DEMAND_MODEL_DENOISED_ACCEL = """
        WITH denoised_accel_cte AS (
            SELECT
                id,
                "time",
                speed,
                accel_without_noise AS acceleration,
                road_grade,
                source_id
            FROM trajectory_table
        )
        {}""".format(VEHICLE_POWER_DEMAND_SUBQUERY.format('denoised_accel_cte'))

    POWER_DEMAND_MODEL_DENOISED_ACCEL_VEL = """
        WITH lagged_timestep AS (
            SELECT
                "time",
                id,
                accel_without_noise,
                road_grade,
                source_id,
                "time" - LAG("time", 1)
                    OVER (PARTITION BY id ORDER BY "time" ASC ROWS BETWEEN 1 PRECEDING and CURRENT ROW) AS sim_step,
                LAG(speed, 1)
                    OVER (PARTITION BY id ORDER BY "time" ASC ROWS BETWEEN 1 PRECEDING and CURRENT ROW) AS prev_speed
            FROM trajectory_table
            WHERE 1 = 1
                AND partition_name=\'{partition}\'
        ), denoised_speed_cte AS (
            SELECT
                id,
                "time",
                prev_speed + accel_without_noise * sim_step AS speed,
                accel_without_noise AS acceleration,
                road_grade,
                source_id
            FROM lagged_timestep
        )
        {}""".format(VEHICLE_POWER_DEMAND_SUBQUERY.format('denoised_speed_cte'))
