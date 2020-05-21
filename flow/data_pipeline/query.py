"""stores all the pre-defined query strings."""
from enum import Enum
from flow.data_pipeline.datapipeline_test import apply_energy_one

# tags for different queries
tags = {
    "energy": [
        "POWER_DEMAND_MODEL",
        "POWER_DEMAND_MODEL_DENOISED_ACCEL",
        "POWER_DEMAND_MODEL_DENOISED_ACCEL_VEL"
        ]
    }

VEHICLE_POWER_DEMAND_FINAL_SELECT = """
    SELECT
        id,
        time_step,
        speed,
        acceleration,
        road_grade,
        1200 * speed * MAX(0, (
            CASE
                WHEN acceleration > 0 THEN 1
                WHEN acceleration < 0 THEN 0
                ELSE 0.5
            END * (1 - 0.8) + 0.8) * acceleration + 9.81 * SIN(road_grade)
            ) + 1200 * 9.81 * 0.005 * speed + 0.5 * 1.225 * 2.6 * 0.3 * POW(speed,3) AS power,
        \'{}\' AS energy_model_id,
        source_id
    FROM {}
    ORDER BY id, time_step
    """


class QueryStrings(Enum):
    """An enumeration of all the pre-defined query strings."""

    SAMPLE = """
        SELECT *
        FROM fact_vehicle_trace
        WHERE date = \'{date}\'
            AND partition_name=\'{partition}\'
        LIMIT 15;
        """

    UPDATE_PARTITION = """
        ALTER TABLE fact_vehicle_trace
        ADD IF NOT EXISTS PARTITION (date = \'{date}\', partition_name=\'{partition}\');
        """

    POWER_DEMAND_MODEL = """
        WITH regular_cte AS (
            SELECT
                id,
                time_step,
                speed,
                target_accel_with_noise_with_failsafe AS acceleration,
                road_grade,
                source_id
            FROM fact_vehicle_trace
            WHERE 1 = 1
                AND date = \'{{date}}\'
                AND partition_name=\'{{partition}}\'
        )
        {}""".format(VEHICLE_POWER_DEMAND_FINAL_SELECT.format('POWER_DEMAND_MODEL',
                                                              'regular_cte'))

    POWER_DEMAND_MODEL_DENOISED_ACCEL = """
        WITH denoised_accel_cte AS (
            SELECT
                id,
                time_step,
                speed,
                target_accel_no_noise_with_failsafe AS acceleration,
                road_grade,
                source_id
            FROM fact_vehicle_trace
            WHERE 1 = 1
                AND date = \'{{date}}\'
                AND partition_name=\'{{partition}}\'
        )
        {}""".format(VEHICLE_POWER_DEMAND_FINAL_SELECT.format('POWER_DEMAND_MODEL_DENOISED_ACCEL',
                                                              'denoised_accel_cte'))

    POWER_DEMAND_MODEL_DENOISED_ACCEL_VEL = """
        WITH lagged_timestep AS (
            SELECT
                id,
                time_step,
                target_accel_no_noise_with_failsafe,
                road_grade,
                source_id,
                time_step - LAG(time_step, 1)
                  OVER (PARTITION BY id ORDER BY time_step ASC ROWS BETWEEN 1 PRECEDING and CURRENT ROW) AS sim_step,
                LAG(speed, 1)
                  OVER (PARTITION BY id ORDER BY time_step ASC ROWS BETWEEN 1 PRECEDING and CURRENT ROW) AS prev_speed
            FROM fact_vehicle_trace
            WHERE 1 = 1
                AND date = \'{{date}}\'
                AND partition_name=\'{{partition}}\'
        ), denoised_speed_cte AS (
            SELECT
                id,
                time_step,
                prev_speed + target_accel_no_noise_with_failsafe * sim_step AS speed,
                target_accel_no_noise_with_failsafe AS acceleration,
                road_grade,
                source_id
            FROM lagged_timestep
        )
        {}""".format(VEHICLE_POWER_DEMAND_FINAL_SELECT.format('POWER_DEMAND_MODEL_DENOISED_ACCEL_VEL',
                                                              'denoised_speed_cte'))
