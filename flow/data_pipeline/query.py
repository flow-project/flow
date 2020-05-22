"""stores all the pre-defined query strings."""
from enum import Enum

# tags for different queries
tags = {"fact_vehicle_trace": {"fact_energy_trace": ["POWER_DEMAND_MODEL", "POWER_DEMAND_MODEL_DENOISED_ACCEL",
                               "POWER_DEMAND_MODEL_DENOISED_ACCEL_VEL"],
                               "fact_network_throughput_agg": ["FACT_NETWORK_THROUGHPUT_AGG"],
                               "fact_network_inflows_outflows": ["FACT_NETWORK_INFLOWS_OUTFLOWS"]},
        "fact_energy_trace": {"fact_vehicle_fuel_efficiency_agg": ["FACT_VEHICLE_FUEL_EFFICIENCY_AGG"],
                              "fact_network_metrics_by_distance_agg": ["FACT_NETWORK_METRICS_BY_DISTANCE_AGG"],
                              "fact_network_metrics_by_time_agg": ["FACT_NETWORK_METRICS_BY_TIME_AGG"]},
        "fact_vehicle_fuel_efficiency_agg": ["FACT_NETWORK_FUEL_EFFICIENCY_AGG"],
        "fact_network_fuel_efficiency_agg": ["LEADERBOARD_CHART"]
        }

VEHICLE_POWER_DEMAND_FINAL_SELECT = """
    SELECT
        id,
        time_step,
        speed,
        acceleration,
        road_grade,
        1200 * speed * (
            (CASE WHEN acceleration > 0 THEN 1 ELSE 0 END * (1-0.8) * acceleration)
            + 0.8 + 9.81 * SIN(road_grade)
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
        FROM trajectory_table
        WHERE date = \'{date}\'
            AND partition_name=\'{partition}\'
        LIMIT 15;
        """

    UPDATE_PARTITION = """
        ALTER TABLE {table}
        ADD IF NOT EXISTS PARTITION (date = \'{date}\', partition_name=\'{partition}\');
        """

    POWER_DEMAND_MODEL = """
        WITH regular_cte AS (
            SELECT
                id,
                time_step,
                speed,
                acceleration,
                road_grade,
                source_id
            FROM fact_vehicle_trace
            WHERE 1 = 1
                AND date = \'{{date}}\'
                AND partition_name=\'{{partition}}\'
        )
        {}""".format(VEHICLE_POWER_DEMAND_FINAL_SELECT.format('POWER_DEMAND_MODEL', 'regular_cte'))

    POWER_DEMAND_MODEL_DENOISED_ACCEL = """
        WITH denoised_accel_cte AS (
            SELECT
                id,
                time_step,
                speed,
                accel_without_noise AS acceleration,
                road_grade,
                source_id
            FROM fact_vehicle_trace
            WHERE 1 = 1
                AND date = \'{{date}}\'
                AND partition_name=\'{{partition}}\'
        )
        {}""".format(VEHICLE_POWER_DEMAND_FINAL_SELECT.format('POWER_DEMAND_MODEL_DENOISED_ACCEL', 'denoised_accel_cte'))

    POWER_DEMAND_MODEL_DENOISED_ACCEL_VEL = """
        WITH lagged_timestep AS (
            SELECT
                id,
                time_step,
                accel_without_noise,
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
                prev_speed + accel_without_noise * sim_step AS speed,
                accel_without_noise AS acceleration,
                road_grade,
                source_id
            FROM lagged_timestep
        )
        {}""".format(VEHICLE_POWER_DEMAND_FINAL_SELECT.format('POWER_DEMAND_MODEL_DENOISED_ACCEL_VEL',
                                                              'denoised_speed_cte'))

    FACT_NETWORK_THROUGHPUT_AGG = """
        WITH agg AS (
            SELECT 
                source_id,
                COUNT(DISTINCT id) AS n_vehicles,
                MAX(time_step) - MIN(time_step) AS total_time_seconds
            FROM fact_vehicle_trace
            WHERE 1 = 1
                AND date = \'{date}\'
                AND partition_name = \'{partition}\'
                AND x BETWEEN 500 AND 2300
                AND time_step >= 600
            GROUP BY 1
        )
        SELECT
            source_id,
            n_vehicles * 3600 / total_time_seconds AS throughput_per_hour
        FROM agg
        ;"""

    FACT_VEHICLE_FUEL_EFFICIENCY_AGG = """
        WITH distance AS (
            SELECT
                id,
                source_id,
                MAX(x) AS distance_meters
            FROM fact_vehicle_trace
            WHERE 1 = 1
                AND date = \'{date}\'
                AND partition_name = \'{partition}\'
                AND source_id = 
                AND x BETWEEN 500 AND 2300
                AND time_step >= 600
            GROUP BY 1, 2
        ), energy AS (
            SELECT
                id,
                source_id,
                energy_model_id,
                (MAX(time_step) - MIN(time_step)) / (COUNT(DISTINCT time_step) - 1) AS time_step_size_seconds,
                SUM(power) AS power_watts
            FROM fact_energy_trace
            WHERE 1 = 1
                AND date = \'{date}\'
                AND partition_name = \'{partition}_POWER_DEMAND_MODEL_DENOISED_ACCEL\'
                AND energy_model_id = 'POWER_DEMAND_MODEL_DENOISED_ACCEL'
                AND x BETWEEN 500 AND 2300
                AND time_step >= 600
            GROUP BY 1, 2, 3
            HAVING COUNT(DISTINCT time_step) > 1
        )
        SELECT
            d.id,
            d.source_id,
            e.energy_model_id,
            distance_meters,
            power_watts * time_step_size_seconds AS energy_joules,
            distance_meters / (power_watts * time_step_size_seconds) AS efficiency_meters_per_joules,
            74564 * distance_meters / (power_watts * time_step_size_seconds) AS efficiency_miles_per_gallon
        FROM distance d
        JOIN energy e ON 1=1 
            AND d.id = e.id 
            AND d.source_id = e.source_id
        ;
    """

    FACT_NETWORK_FUEL_EFFICIENCY_AGG = """
        SELECT
            source_id,
            energy_model_id,
            SUM(distance_meters) AS distance_meters,
            SUM(energy_joules) AS energy_joules,
            SUM(distance_meters) / SUM(energy_joules) AS efficiency_meters_per_joules,
            74564 * SUM(distance_meters) / SUM(energy_joules) AS efficiency_miles_per_gallon
        FROM fact_vehicle_fuel_efficiency_agg
        WHERE 1 = 1
            AND date = \'{date}\'
            AND parititon_name = \'{partition}_FACT_VEHICLE_FUEL_EFFICIENCY_AGG\'
            AND energy_model_id = 'POWER_DEMAND_MODEL_DENOISED_ACCEL'
        GROUP BY 1, 2
        ;"""

    LEADERBOARD_CHART = """
        SELECT
            t.source_id,
            e.energy_model_id,
            e.efficiency_meters_per_joules,
            74564 * e.efficiency_meters_per_joules AS efficiency_miles_per_gallon
            t.throughput_per_hour
        FROM fact_network_throughput_agg t 
        JOIN fact_network_fuel_efficiency_agg e ON 1 = 1
            AND t.date = \'{date}\'
            AND t.partition_name = \'{partition}_FACT_NETWORK_THROUGHPUT_AGG\'
            AND e.date = \'{date}\'
            AND e.partition_name = \'{partition}_FACT_NETWORK_FUEL_EFFICIENCY_AGG\'
            AND t.source_id = e.source_id
            AND e.energy_model_id = 'POWER_DEMAND_MODEL_DENOISED_ACCEL'
        WHERE 1 = 1
        ;"""

    FACT_NETWORK_INFLOWS_OUTFLOWS = """
        WITH min_max_time_step AS (
            SELECT 
                id,
                source_id,
                MIN(time_step) AS min_time_step,
                MAX(time_step) AS max_time_step
            FROM fact_vehicle_trace
            WHERE 1 = 1
                AND date = \'{date}\'
                AND partition_name = \'{partition}\'
                AND x BETWEEN 500 AND 2300
                AND time_step >= 600
            GROUP BY 1, 2
        ), inflows AS (
            SELECT
                CAST(min_time_step / 60 AS INTEGER) * 60 AS time_step,
                source_id,
                60 * COUNT(DISTINCT id) AS inflow_rate
            FROM min_max_time_step
            GROUP BY 1, 2
        ), outflows AS (
            SELECT
                CAST(max_time_step / 60 AS INTEGER) * 60 AS time_step,
                source_id,
                60 * COUNT(DISTINCT id) AS outflow_rate
            FROM min_max_time_step
            GROUP BY 1, 2
        )
        SELECT
            COALESCE(i.time_step, o.time_step) AS time_step,
            COALESCE(i.source_id, o.source_id) AS source_id,
            COALESCE(i.inflow_rate, 0) AS inflow_rate,
            COALESCE(o.outflow_rate, 0) AS outflow_rate
        FROM inflows i 
        FULL OUTER JOIN outflows o ON 1 = 1
            AND i.time_step = o.time_step 
            AND i.source_id = o.source_id 
        ;"""

    FACT_NETWORK_METRICS_BY_DISTANCE_AGG = """
        WITH joined_trace AS (
            SELECT
                id,
                source_id,
                time_step,
                x,
                energy_model_id,
                time_step - LAG(time_step, 1)
                    OVER (PARTITION BY id ORDER BY time_step ASC ROWS BETWEEN 1 PRECEDING and CURRENT ROW) AS sim_step,
                SUM(power)
                    OVER (PARTITION BY id ORDER BY time_step ASC ROWS BETWEEN UNBOUNDED PRECEDING and CURRENT ROW) AS
                     cumulative_power
            FROM fact_vehicle_trace vt 
            JOIN fact_energy_trace et ON 1 = 1
                AND vt.date = \'{date}\'
                AND vt.partition_name = \'{partition}\'
                AND et.date = \'{date}\'
                AND et.partition_name = \'{partition}_POWER_DEMAND_MODEL_DENOISED_ACCEL\'
                AND vt.id = et.id
                AND vt.source_id = et.source_id
                AND vt.time_step = et.time_step
                AND vt.x BETWEEN 500 AND 2300
                AND vt.time_step >= 600
                AND et.energy_model_id = 'POWER_DEMAND_MODEL_DENOISED_ACCEL'
            WHERE 1 = 1
        ), cumulative_energy AS (
            SELECT
                id,
                source_id,
                time_step,
                x,
                energy_model_id,
                cumulative_power * sim_step AS energy_joules
            FROM joined_trace
            WHERE 1 = 1
                AND energy_model_id = 'POWER_DEMAND_MODEL_DENOISED_ACCEL'
                AND x BETWEEN 500 AND 2300
                AND time_step >= 600
            GROUP BY 1, 2, 3
            HAVING COUNT(DISTINCT time_step) > 1
        ), binned_cumulative_energy AS (
            SELECT
                source_id,
                CAST(x/10 AS INTEGER) * 10 AS distance_meters_bin,
                AVG(energy_joules) AS cumulative_energy_avg,
                AVG(energy_joules) + STDEV(energy_joules) AS cumulative_energy_upper_bound,
                AVG(energy_joules) - STDEV(energy_joules) AS cumulative_energy_lower_bound
            FROM cumulative_energy
            GROUP BY 1, 2
        ), binned_speed_accel AS (
            SELECT
                source_id,
                CAST(x/10 AS INTEGER) * 10 AS distance_meters_bin,
                AVG(speed) AS speed_avg,
                AVG(speed) + STDEV(speed) AS speed_upper_bound,
                AVG(speed) - STDEV(speed) AS speed_lower_bound,
                AVG(accel_without_noise) AS accel_avg,
                AVG(accel_without_noise) + STDEV(accel_without_noise) AS accel_upper_bound,
                AVG(accel_without_noise) - STDEV(accel_without_noise) AS accel_lower_bound
            FROM fact_vehicle_trace
            WHERE 1 = 1
                AND date =  \'{date}\'
                AND partition_name = \'{partition}\'
                AND x BETWEEN 500 AND 2300
                AND time_step >= 600
            GROUP BY 1, 2
        ), binned_energy_start_end AS (
            SELECT DISTINCT
                source_id,
                id,
                CAST(x/10 AS INTEGER) * 10 AS distance_meters_bin,
                FIRST_VALUE(energy_joules) OVER (PARTITION BY id, CAST(x/10 AS INTEGER) * 10 ORDER BY x ASC) AS energy_start,
                LAST_VALUE(energy_joules) OVER (PARTITION BY id, CAST(x/10 AS INTEGER) * 10 ORDER BY x ASC) AS energy_end
            FROM cumulative_energy
        ), binned_energy AS (
            SELECT
                source_id,
                distance_meters_bin,
                AVG(energy_end - energy_start) AS instantaneous_energy_avg,
                AVG(energy_end - energy_start) + STDEV(energy_end - energy_start) AS instantaneous_energy_upper_bound,
                AVG(energy_end - energy_start) - STDEV(energy_end - energy_start) AS instantaneous_energy_lower_bound
            FROM binned_energy_start_end
            GROUP BY 1, 2
        )
        SELECT
            COALESCE(bce.source_id, bsa.source_id, be.source_id) AS source_id,
            COALESCE(bce.distance_meters_bin, bsa.distance_meters_bin, be.distance_meters_bin) AS distance_meters_bin,
            bce.cumulative_energy_avg,
            bce.cumulative_energy_lower_bound,
            bce.cumulative_energy_upper_bound,
            bsa.speed_avg,
            bsa.speed_upper_bound,
            bsa.speed_lower_bound,
            bsa.accel_avg,
            bsa.accel_upper_bound,
            bsa.accel_lower_bound,
            be.instantaneous_energy_avg,
            be.instantaneous_energy_upper_bound,
            be.instantaneous_energy_lower_bound
        FROM binned_cumulative_energy bce 
        FULL OUTER JOIN binned_speed_accel bsa ON 1 = 1
            AND bce.source_id = bsa.source_id
            AND bce.distance_meters_bin = bsa.distance_meters_bin
        FULL OUTER JOIN binned_energy be ON 1 = 1
            AND COALESCE(bce.source_id, bsa.source_id) = be.source_id
            AND COALESCE(bce.distance_meters_bin, bce.distance_meters_bin) = be.distance_meters_bin
        ;"""

    FACT_NETWORK_METRICS_BY_TIME_AGG = """
        WITH joined_trace AS (
            SELECT
                id,
                source_id,
                time_step,
                x,
                energy_model_id,
                time_step - LAG(time_step, 1)
                    OVER (PARTITION BY id ORDER BY time_step ASC ROWS BETWEEN 1 PRECEDING and CURRENT ROW) AS sim_step,
                SUM(power)
                    OVER (PARTITION BY id ORDER BY time_step ASC ROWS BETWEEN UNBOUNDED PRECEDING and CURRENT ROW) AS
                     cumulative_power
            FROM fact_vehicle_trace vt 
            JOIN fact_energy_trace et ON 1 = 1
                AND vt.date = \'{date}\'
                AND vt.partition_name = \'{partition}\'
                AND et.date = \'{date}\'
                AND et.partition_name = \'{partitio}_POWER_DEMAND_MODEL_DENOISED_ACCEL\'
                AND vt.id = et.id
                AND vt.source_id = et.source_id
                AND vt.time_step = et.time_step
                AND vt.x BETWEEN 500 AND 2300
                AND vt.time_step >= 600
                AND et.energy_model_id = 'POWER_DEMAND_MODEL_DENOISED_ACCEL'
            WHERE 1 = 1
        ), cumulative_energy AS (
            SELECT
                id,
                source_id,
                time_step,
                x,
                energy_model_id,
                cumulative_power * sim_step AS energy_joules
            FROM joined_trace
            WHERE 1 = 1
                AND date = 
                AND partition_name = 
                AND source_id = 
                AND energy_model_id = 'POWER_DEMAND_MODEL_DENOISED_ACCEL'
                AND x BETWEEN 500 AND 2300
                AND time_step >= 600
            GROUP BY 1, 2, 3
            HAVING COUNT(DISTINCT time_step) > 1
        ), binned_cumulative_energy AS (
            SELECT
                source_id,
                CAST(time_step/60 AS INTEGER) * 60 AS time_seconds_bin,
                AVG(energy_joules) AS cumulative_energy_avg,
                AVG(energy_joules) + STDEV(energy_joules) AS cumulative_energy_upper_bound,
                AVG(energy_joules) - STDEV(energy_joules) AS cumulative_energy_lower_bound
            FROM cumulative_energy
            GROUP BY 1, 2
        ), binned_speed_accel AS (
            SELECT
                source_id,
                CAST(time_step/60 AS INTEGER) * 60 AS time_seconds_bin,
                AVG(speed) AS speed_avg,
                AVG(speed) + STDEV(speed) AS speed_upper_bound,
                AVG(speed) - STDEV(speed) AS speed_lower_bound,
                AVG(accel_without_noise) AS accel_avg,
                AVG(accel_without_noise) + STDEV(accel_without_noise) AS accel_upper_bound,
                AVG(accel_without_noise) - STDEV(accel_without_noise) AS accel_lower_bound
            FROM fact_vehicle_trace
            WHERE 1 = 1
                AND date =  \'{date}\'
                AND partition_name = \'{partition}\'
                AND x BETWEEN 500 AND 2300
                AND time_step >= 600
            GROUP BY 1, 2
        ), binned_energy_start_end AS (
            SELECT DISTINCT
                source_id,
                id,
                CAST(time_step/60 AS INTEGER) * 60 AS time_seconds_bin,
                FIRST_VALUE(energy_joules) OVER (PARTITION BY id, CAST(x/10 AS INTEGER) * 10 ORDER BY x ASC) AS energy_start,
                LAST_VALUE(energy_joules) OVER (PARTITION BY id, CAST(x/10 AS INTEGER) * 10 ORDER BY x ASC) AS energy_end
            FROM cumulative_energy
        ), binned_energy AS (
            SELECT
                source_id,
                time_seconds_bin,
                AVG(energy_end - energy_start) AS instantaneous_energy_avg,
                AVG(energy_end - energy_start) + STDEV(energy_end - energy_start) AS instantaneous_energy_upper_bound,
                AVG(energy_end - energy_start) - STDEV(energy_end - energy_start) AS instantaneous_energy_lower_bound
            FROM binned_energy_start_end
            GROUP BY 1, 2
        )
        SELECT
            COALESCE(bce.source_id, bsa.source_id, be.source_id) AS source_id,
            COALESCE(bce.time_seconds_bin, bsa.time_seconds_bin, be.time_seconds_bin) AS time_seconds_bin,
            bce.cumulative_energy_avg,
            bce.cumulative_energy_lower_bound,
            bce.cumulative_energy_upper_bound,
            bsa.speed_avg,
            bsa.speed_upper_bound,
            bsa.speed_lower_bound,
            bsa.accel_avg,
            bsa.accel_upper_bound,
            bsa.accel_lower_bound,
            be.instantaneous_energy_avg,
            be.instantaneous_energy_upper_bound,
            be.instantaneous_energy_lower_bound
        FROM binned_cumulative_energy bce 
        FULL OUTER JOIN binned_speed_accel bsa ON 1 = 1
            AND bce.source_id = bsa.source_id
            AND bce.time_seconds_bin = bsa.time_seconds_bin
        FULL OUTER JOIN binned_energy be ON 1 = 1
            AND COALESCE(bce.source_id, bsa.source_id) = be.source_id
            AND COALESCE(bce.time_seconds_bin, bce.time_seconds_bin) = be.time_seconds_bin
        ;"""
