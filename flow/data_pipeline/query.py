"""stores all the pre-defined query strings."""
from enum import Enum
from flow.data_pipeline.datapipeline_test import apply_energy_one

# tags for different queries
tags = {"energy": ["POWER_DEMAND_MODEL", "POWER_DEMAND_MODEL_DENOISED_ACCEL", "POWER_DEMAND_MODEL_DENOISED_ACCEL_VEL"],
        "analysis": ["POWER_DEMAND_MODEL"]}

VEHICLE_POWER_DEMAND_FINAL_SELECT = """
    SELECT
        id,
        "time",
        speed,
        acceleration,
        road_grade,
        1200 * speed * (
            (CASE WHEN acceleration > 0 THEN 1 ELSE 0 END * (1-0.8) * acceleration)
            + 0.8 + 9.81 * SIN(road_grade)
            ) + 1200 * 9.81 * 0.005 * speed + 0.5 * 1.225 * 2.6 * 0.3 * POW(speed,3) AS power,
        'POWER_DEMAND_MODEL' AS energy_model_id,
        source_id
    FROM {}
    ORDER BY id, "time"
    """

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

    POWER_DEMAND_MODEL = """
        WITH regular_cte AS (
            SELECT
                id,
                "time",
                speed,
                acceleration,
                road_grade,
                source_id
            FROM trajectory_table
            WHERE 1 = 1
                AND partition_name=\'{{partition}}\'
        )
        {}""".format(VEHICLE_POWER_DEMAND_FINAL_SELECT.format('regular_cte'))

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
            WHERE 1 = 1
                AND partition_name=\'{{partition}}\'
        )
        {}""".format(VEHICLE_POWER_DEMAND_FINAL_SELECT.format('denoised_accel_cte'))

    POWER_DEMAND_MODEL_DENOISED_ACCEL_VEL = """
        WITH lagged_timestep AS (
            SELECT
                id,
                "time",
                accel_without_noise,
                road_grade,
                source_id,
                "time" - LAG("time", 1)
                    OVER (PARTITION BY id ORDER BY "time" ASC ROWS BETWEEN 1 PRECEDING and CURRENT ROW) AS sim_step,
                LAG(speed, 1)
                    OVER (PARTITION BY id ORDER BY "time" ASC ROWS BETWEEN 1 PRECEDING and CURRENT ROW) AS prev_speed
            FROM trajectory_table
            WHERE 1 = 1
                AND partition_name=\'{{partition}}\'
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
        {}""".format(VEHICLE_POWER_DEMAND_FINAL_SELECT.format('denoised_speed_cte'))
