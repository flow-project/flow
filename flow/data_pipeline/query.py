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

    SAMPLE = "SELECT * FROM trajectory_table WHERE partition_name=\'{partition}\' LIMIT 15;"
    UPDATE_PARTITION = "ALTER TABLE trajectory_table ADD IF NOT EXISTS PARTITION (partition_name=\'{partition}\');"
    POWER_DEMAND_MODEL = """
                         SELECT id, time, speed, acceleration, 1200 * speed *
                         ((CASE WHEN acceleration > 0 THEN 1 ELSE 0 END * (1-0.8) * acceleration) + 0.8
                         + 9.81 * SIN(road_grade)) + 1200 * 9.81 * 0.005 * speed + 0.5 * 1.225 * 2.6 * 0.3
                         * POW(speed,3) AS power,
                         'POWER_DEMAND_MODEL' AS energy_model_id, source_id
                         FROM trajectory_table
                         WHERE partition_name=\'{partition}\'
                         ORDER BY id, time"""
    POWER_DEMAND_MODEL_DENOISED_ACCEL = """
                         SELECT id, time, speed, accel_without_noise,
                         1200 * speed * ((CASE WHEN accel_without_noise > 0 THEN 1 ELSE 0 END * (1-0.8)
                         * accel_without_noise)+0.8 + 9.81 * SIN(road_grade))
                         + 1200 * 9.81 * 0.005 * speed + 0.5 * 1.225 * 2.6 * 0.3 * POW(speed,3) AS power,
                         'POWER_DEMAND_MODEL_DENOISED_ACCEL' AS energy_model_id, source_id
                         FROM trajectory_table
                         WHERE partition_name=\'{partition}\'
                         ORDER BY id, time"""
    POWER_DEMAND_MODEL_DENOISED_ACCEL_VEL = """
    WITH lagged_timestep AS (
        SELECT time, id, speed, acceleration, accel_without_noise, road_grade, source_id,
        time - LAG(time, 1)
            OVER (PARTITION BY id ORDER BY time ASC ROWS BETWEEN 1 PRECEDING and CURRENT ROW) AS sim_step,
        LAG(speed, 1)
            OVER (PARTITION BY id ORDER BY time ASC ROWS BETWEEN 1 PRECEDING and CURRENT ROW) AS prev_speed,
        LAG(acceleration, 1)
            OVER (PARTITION BY id ORDER BY time ASC ROWS BETWEEN 1 PRECEDING and CURRENT ROW) AS prev_accel,
        LAG(accel_without_noise, 1)
            OVER (PARTITION BY id ORDER BY time ASC ROWS BETWEEN 1 PRECEDING and CURRENT ROW) AS prev_accel_denoised
        FROM trajectory_table
        WHERE partition_name=\'{partition}\'),
        speed_denoised_table AS (
        SELECT time, id, speed, acceleration, accel_without_noise, road_grade, source_id,
        prev_speed+accel_without_noise*sim_step AS speed_denoised
        FROM lagged_timestep)
    SELECT id, time, speed_denoised, accel_without_noise,
    1200 * speed_denoised * ((CASE WHEN accel_without_noise > 0
    THEN 1 ELSE 0 END * (1-0.8) * accel_without_noise) + 0.8 + 9.81
    * SIN(road_grade)) + 1200 * 9.81 * 0.005 * speed_denoised + 0.5 * 1.225
    * 2.6 * 0.3 * POW(speed_denoised,3) AS power,
    'POWER_DEMAND_MODEL_DENOISED_ACCEL_VEL' AS energy_model, source_id
    FROM speed_denoised_table
    ORDER BY id, time"""
