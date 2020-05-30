"""stores all the pre-defined query strings."""
from enum import Enum

# tags for different queries
tags = {"fact_vehicle_trace": {"fact_energy_trace": ["POWER_DEMAND_MODEL", "POWER_DEMAND_MODEL_DENOISED_ACCEL",
                               "POWER_DEMAND_MODEL_DENOISED_ACCEL_VEL"],
                               "fact_network_throughput_agg": ["FACT_NETWORK_THROUGHPUT_AGG"],
                               "fact_network_inflows_outflows": ["FACT_NETWORK_INFLOWS_OUTFLOWS"]},
        "fact_energy_trace": {},
        "POWER_DEMAND_MODEL_DENOISED_ACCEL": {"fact_vehicle_fuel_efficiency_agg": ["FACT_VEHICLE_FUEL_EFFICIENCY_AGG"],
                                              "fact_network_metrics_by_distance_agg":
                                                  ["FACT_NETWORK_METRICS_BY_DISTANCE_AGG"],
                                              "fact_network_metrics_by_time_agg": ["FACT_NETWORK_METRICS_BY_TIME_AGG"]},
        "fact_vehicle_fuel_efficiency_agg": {"fact_network_fuel_efficiency_agg": ["FACT_NETWORK_FUEL_EFFICIENCY_AGG"]},
        "fact_network_fuel_efficiency_agg": {"leaderboard_chart": ["LEADERBOARD_CHART"]},
        "leaderboard_chart": {"leaderboard_chart_agg": ["LEADERBOARD_CHART_AGG"]}
        }

tables = ["fact_vehicle_trace", "fact_energy_trace", "fact_network_throughput_agg", "fact_network_inflows_outflows",
          "fact_vehicle_fuel_efficiency_agg", "fact_network_metrics_by_distance_agg",
          "fact_network_metrics_by_time_agg", "fact_network_fuel_efficiency_agg", "leaderboard_chart",
          "leaderboard_chart_agg"]

VEHICLE_POWER_DEMAND_COMBUSTION_FINAL_SELECT = """
    SELECT
        id,
        time_step,
        speed,
        acceleration,
        road_grade,
        GREATEST(0, 1200 * speed * ((
            CASE
                WHEN acceleration > 0 THEN 1
                WHEN acceleration < 0 THEN 0
                ELSE 0.5
            END * (1 - {0}) + {0}) * acceleration + 9.81 * SIN(road_grade)
            ) + 1200 * 9.81 * 0.005 * speed + 0.5 * 1.225 * 2.6 * 0.3 * POW(speed,3)) AS power,
        \'{1}\' AS energy_model_id,
        source_id
    FROM {2}
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
                COALESCE (target_accel_with_noise_with_failsafe, realized_accel) AS acceleration,
                road_grade,
                source_id
            FROM fact_vehicle_trace
            WHERE 1 = 1
                AND date = \'{{date}}\'
                AND partition_name=\'{{partition}}\'
        )
        {}""".format(VEHICLE_POWER_DEMAND_COMBUSTION_FINAL_SELECT.format(1,
                                                                         'POWER_DEMAND_MODEL',
                                                                         'regular_cte'))

    POWER_DEMAND_MODEL_DENOISED_ACCEL = """
        WITH denoised_accel_cte AS (
            SELECT
                id,
                time_step,
                speed,
                COALESCE (target_accel_no_noise_with_failsafe,
                          target_accel_no_noise_no_failsafe,
                          realized_accel) AS acceleration,
                road_grade,
                source_id
            FROM fact_vehicle_trace
            WHERE 1 = 1
                AND date = \'{{date}}\'
                AND partition_name=\'{{partition}}\'
        )
        {}""".format(VEHICLE_POWER_DEMAND_COMBUSTION_FINAL_SELECT.format(1,
                                                                         'POWER_DEMAND_MODEL_DENOISED_ACCEL',
                                                                         'denoised_accel_cte'))

    POWER_DEMAND_MODEL_DENOISED_ACCEL_VEL = """
        WITH lagged_timestep AS (
            SELECT
                id,
                time_step,
                COALESCE (target_accel_no_noise_with_failsafe,
                          target_accel_no_noise_no_failsafe,
                          realized_accel) AS acceleration,
                road_grade,
                source_id,
                speed AS cur_speed,
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
                COALESCE (prev_speed + acceleration * sim_step, cur_speed) AS speed,
                acceleration,
                road_grade,
                source_id
            FROM lagged_timestep
        )
        {}""".format(VEHICLE_POWER_DEMAND_COMBUSTION_FINAL_SELECT.format(1,
                                                                         'POWER_DEMAND_MODEL_DENOISED_ACCEL_VEL',
                                                                         'denoised_speed_cte'))

    FACT_NETWORK_THROUGHPUT_AGG = """
        WITH min_time AS (
            SELECT
                source_id,
                id,
                MIN(time_step) AS enter_time
            FROM fact_vehicle_trace
            WHERE 1 = 1
                AND date = \'{date}\'
                AND partition_name = \'{partition}\'
                AND x BETWEEN 500 AND 2300
            GROUP BY 1, 2
        ), agg AS (
            SELECT
                source_id,
                COUNT(DISTINCT id) AS n_vehicles,
                MAX(enter_time) - MIN(enter_time) AS total_time_seconds
            FROM min_time
            WHERE 1 = 1
                AND enter_time >= 600
            GROUP BY 1
        )
        SELECT
            source_id,
            n_vehicles * 3600 / total_time_seconds AS throughput_per_hour
        FROM agg
        ;"""

    FACT_VEHICLE_FUEL_EFFICIENCY_AGG = """
        WITH sub_fact_vehicle_trace AS (
            SELECT 
                v.id,
                v.source_id,
                e.energy_model_id,
                MAX(distance) - MIN(distance) AS distance_meters,
                (MAX(e.time_step) - MIN(e.time_step)) / (COUNT(DISTINCT e.time_step) - 1) AS time_step_size_seconds,
                SUM(e.power) AS power_watts
            FROM fact_vehicle_trace v
            JOIN fact_energy_trace AS e ON  1 = 1
                AND e.id = v.id
                AND e.time_step = v.time_step
                AND e.source_id = v.source_id
                AND e.date = \'{date}\'
                AND e.partition_name = \'{partition}_POWER_DEMAND_MODEL_DENOISED_ACCEL\'
                AND e.energy_model_id = 'POWER_DEMAND_MODEL_DENOISED_ACCEL'
                AND e.time_step >= 600
            WHERE 1 = 1
                AND v.date = \'{date}\'
                AND v.partition_name = \'{partition}\'
                AND v.x BETWEEN 500 AND 2300
            GROUP BY 1, 2, 3
            HAVING 1 = 1
                AND MIN(distance) - MIN(distance) > 10
                AND COUNT(DISTINCT e.time_step) > 10
        )
        SELECT
            id,
            source_id,
            energy_model_id,
            distance_meters,
            power_watts * time_step_size_seconds AS energy_joules,
            distance_meters / (power_watts * time_step_size_seconds) AS efficiency_meters_per_joules,
            19972 * distance_meters / (power_watts * time_step_size_seconds) AS efficiency_miles_per_gallon
        FROM sub_fact_vehicle_trace
        WHERE 1 = 1
            AND ABS(power_watts * time_step_size_seconds) > 0
        ;
    """

    FACT_NETWORK_FUEL_EFFICIENCY_AGG = """
        SELECT
            source_id,
            energy_model_id,
            SUM(distance_meters) AS distance_meters,
            SUM(energy_joules) AS energy_joules,
            SUM(distance_meters) / SUM(energy_joules) AS efficiency_meters_per_joules,
            19972 * SUM(distance_meters) / SUM(energy_joules) AS efficiency_miles_per_gallon
        FROM fact_vehicle_fuel_efficiency_agg
        WHERE 1 = 1
            AND date = \'{date}\'
            AND partition_name = \'{partition}_FACT_VEHICLE_FUEL_EFFICIENCY_AGG\'
            AND energy_model_id = 'POWER_DEMAND_MODEL_DENOISED_ACCEL'
            AND SUM(energy_joules) != 0
        GROUP BY 1, 2
        ;"""

    LEADERBOARD_CHART = """
        SELECT
            t.source_id,
            e.energy_model_id,
            e.efficiency_meters_per_joules,
            19972 * e.efficiency_meters_per_joules AS efficiency_miles_per_gallon,
            t.throughput_per_hour
        FROM fact_network_throughput_agg AS t
        JOIN fact_network_fuel_efficiency_agg AS e ON 1 = 1
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
            GROUP BY 1, 2
        ), inflows AS (
            SELECT
                CAST(min_time_step / 60 AS INTEGER) * 60 AS time_step,
                source_id,
                60 * COUNT(DISTINCT id) AS inflow_rate
            FROM min_max_time_step
            WHERE 1 = 1
                AND min_time_step >= 600
            GROUP BY 1, 2
        ), outflows AS (
            SELECT
                CAST(max_time_step / 60 AS INTEGER) * 60 AS time_step,
                source_id,
                60 * COUNT(DISTINCT id) AS outflow_rate
            FROM min_max_time_step
            WHERE 1 = 1
                AND max_time_step < 1000
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
        ORDER BY time_step
        ;"""

    FACT_NETWORK_METRICS_BY_DISTANCE_AGG = """
        WITH joined_trace AS (
            SELECT
                vt.id,
                vt.source_id,
                vt.time_step,
                vt.distance - FIRST_VALUE(vt.distance)
                    OVER (PARITION BY vt.id, vt.source_id ORDER BY vt.time_step ASC) AS distance_meters,
                energy_model_id,
                et.speed,
                et.acceleration,
                vt.time_step - LAG(vt.time_step, 1)
                    OVER (PARTITION BY vt.id, vt.source_id ORDER BY vt.time_step ASC) AS sim_step,
                SUM(power)
                    OVER (PARTITION BY vt.id, vt.source_id ORDER BY vt.time_step ASC
                    ROWS BETWEEN UNBOUNDED PRECEDING and CURRENT ROW) AS cumulative_power
            FROM fact_vehicle_trace vt 
            JOIN fact_energy_trace et ON 1 = 1
                AND et.date = \'{date}\'
                AND et.partition_name = \'{partition}_POWER_DEMAND_MODEL_DENOISED_ACCEL\'
                AND vt.id = et.id
                AND vt.source_id = et.source_id
                AND vt.time_step = et.time_step
                AND et.energy_model_id = 'POWER_DEMAND_MODEL_DENOISED_ACCEL'
            WHERE 1 = 1
                AND vt.date = \'{date}\'
                AND vt.partition_name = \'{partition}\'
                AND vt.x BETWEEN 500 AND 2300
                AND vt.time_step >= 600
        ), cumulative_energy AS (
            SELECT
                id,
                source_id,
                time_step,
                distance_meters,
                energy_model_id,
                speed,
                acceleration,
                cumulative_power * sim_step AS energy_joules
            FROM joined_trace
        ), binned_cumulative_energy AS (
            SELECT
                source_id,
                CAST(distance_meters/10 AS INTEGER) * 10 AS distance_meters_bin,
                AVG(speed) AS speed_avg,
                AVG(speed) + STDDEV(speed) AS speed_upper_bound,
                AVG(speed) - STDDEV(speed) AS speed_lower_bound,
                AVG(target_accel_no_noise_with_failsafe) AS accel_avg,
                AVG(target_accel_no_noise_with_failsafe) + STDDEV(target_accel_no_noise_with_failsafe) AS accel_upper_bound,
                AVG(target_accel_no_noise_with_failsafe) - STDDEV(target_accel_no_noise_with_failsafe) AS accel_lower_bound,
                AVG(energy_joules) AS cumulative_energy_avg,
                AVG(energy_joules) + STDDEV(energy_joules) AS cumulative_energy_upper_bound,
                AVG(energy_joules) - STDDEV(energy_joules) AS cumulative_energy_lower_bound
            FROM cumulative_energy
            GROUP BY 1, 2
            HAVING 1 = 1
                AND COUNT(DISTINCT time_step) > 1
        ), binned_energy_start_end AS (
            SELECT DISTINCT
                source_id,
                id,
                CAST(distance_meters/10 AS INTEGER) * 10 AS distance_meters_bin,
                FIRST_VALUE(energy_joules)
                    OVER (PARTITION BY id, CAST(distance_meters/10 AS INTEGER) * 10 ORDER BY x ASC) AS energy_start,
                LAST_VALUE(energy_joules)
                    OVER (PARTITION BY id, CAST(distance_meters/10 AS INTEGER) * 10 ORDER BY x ASC) AS energy_end
            FROM cumulative_energy
        ), binned_energy AS (
            SELECT
                source_id,
                distance_meters_bin,
                AVG(energy_end - energy_start) AS instantaneous_energy_avg,
                AVG(energy_end - energy_start) + STDDEV(energy_end - energy_start) AS instantaneous_energy_upper_bound,
                AVG(energy_end - energy_start) - STDDEV(energy_end - energy_start) AS instantaneous_energy_lower_bound
            FROM binned_energy_start_end
            GROUP BY 1, 2
        )
        SELECT
            COALESCE(bce.source_id, be.source_id) AS source_id,
            COALESCE(bce.distance_meters_bin, be.distance_meters_bin) AS distance_meters_bin,
            bce.cumulative_energy_avg,
            bce.cumulative_energy_lower_bound,
            bce.cumulative_energy_upper_bound,
            bce.speed_avg,
            bce.speed_upper_bound,
            bce.speed_lower_bound,
            bce.accel_avg,
            bce.accel_upper_bound,
            bce.accel_lower_bound,
            be.instantaneous_energy_avg,
            be.instantaneous_energy_upper_bound,
            be.instantaneous_energy_lower_bound
        FROM binned_cumulative_energy bce 
        FULL OUTER JOIN binned_energy be ON 1 = 1
            AND bce.source_id = be.source_id
            AND bce.distance_meters_bin = be.distance_meters_bin
        ORDER BY distance_meters_bin ASC
        ;"""

    FACT_NETWORK_METRICS_BY_TIME_AGG = """
        WITH joined_trace AS (
            SELECT
                vt.id,
                vt.source_id,
                vt.time_step,
                energy_model_id,
                et.speed,
                et.acceleration,
                vt.time_step - LAG(vt.time_step, 1)
                    OVER (PARTITION BY vt.id, vt.source_id ORDER BY vt.time_step ASC) AS sim_step,
                SUM(power)
                    OVER (PARTITION BY vt.id, vt.source_id ORDER BY vt.time_step ASC
                    ROWS BETWEEN UNBOUNDED PRECEDING and CURRENT ROW) AS cumulative_power
            FROM fact_vehicle_trace vt 
            JOIN fact_energy_trace et ON 1 = 1
                AND et.date = \'{date}\'
                AND et.partition_name = \'{partition}_POWER_DEMAND_MODEL_DENOISED_ACCEL\'
                AND vt.id = et.id
                AND vt.source_id = et.source_id
                AND vt.time_step = et.time_step
                AND et.energy_model_id = 'POWER_DEMAND_MODEL_DENOISED_ACCEL'
            WHERE 1 = 1
                AND vt.date = \'{date}\'
                AND vt.partition_name = \'{partition}\'
                AND vt.x BETWEEN 500 AND 2300
                AND vt.time_step >= 600
        ), cumulative_energy AS (
            SELECT
                id,
                source_id,
                time_step,
                energy_model_id,
                speed,
                acceleration,
                cumulative_power * sim_step AS energy_joules
            FROM joined_trace
        ), binned_cumulative_energy AS (
            SELECT
                source_id,
                CAST(time_step/60 AS INTEGER) * 60 AS time_seconds_bin,
                AVG(speed) AS speed_avg,
                AVG(speed) + STDDEV(speed) AS speed_upper_bound,
                AVG(speed) - STDDEV(speed) AS speed_lower_bound,
                AVG(target_accel_no_noise_with_failsafe) AS accel_avg,
                AVG(target_accel_no_noise_with_failsafe) + STDDEV(target_accel_no_noise_with_failsafe) AS accel_upper_bound,
                AVG(target_accel_no_noise_with_failsafe) - STDDEV(target_accel_no_noise_with_failsafe) AS accel_lower_bound,
                AVG(energy_joules) AS cumulative_energy_avg,
                AVG(energy_joules) + STDDEV(energy_joules) AS cumulative_energy_upper_bound,
                AVG(energy_joules) - STDDEV(energy_joules) AS cumulative_energy_lower_bound
            FROM cumulative_energy
            GROUP BY 1, 2
            HAVING 1 = 1
                AND COUNT(DISTINCT time_step) > 1
        ), binned_energy_start_end AS (
            SELECT DISTINCT
                source_id,
                id,
                CAST(time_step/60 AS INTEGER) * 60 AS time_seconds_bin,
                FIRST_VALUE(energy_joules)
                    OVER (PARTITION BY id, CAST(time_step/60 AS INTEGER) * 60 ORDER BY x ASC) AS energy_start,
                LAST_VALUE(energy_joules)
                    OVER (PARTITION BY id, CAST(time_step/60 AS INTEGER) * 60 ORDER BY x ASC) AS energy_end
            FROM cumulative_energy
        ), binned_energy AS (
            SELECT
                source_id,
                time_seconds_bin,
                AVG(energy_end - energy_start) AS instantaneous_energy_avg,
                AVG(energy_end - energy_start) + STDDEV(energy_end - energy_start) AS instantaneous_energy_upper_bound,
                AVG(energy_end - energy_start) - STDDEV(energy_end - energy_start) AS instantaneous_energy_lower_bound
            FROM binned_energy_start_end
            GROUP BY 1, 2
        )
        SELECT
            COALESCE(bce.source_id, be.source_id) AS source_id,
            COALESCE(bce.time_seconds_bin, be.time_seconds_bin) AS time_seconds_bin,
            bce.cumulative_energy_avg,
            bce.cumulative_energy_lower_bound,
            bce.cumulative_energy_upper_bound,
            bce.speed_avg,
            bce.speed_upper_bound,
            bce.speed_lower_bound,
            bce.accel_avg,
            bce.accel_upper_bound,
            bce.accel_lower_bound,
            be.instantaneous_energy_avg,
            be.instantaneous_energy_upper_bound,
            be.instantaneous_energy_lower_bound
        FROM binned_cumulative_energy bce 
        FULL OUTER JOIN binned_energy be ON 1 = 1
            AND bce.source_id = be.source_id
            AND bce.time_seconds_bin = be.time_seconds_bin
        ORDER BY time_seconds_bin ASC
        ;"""

    LEADERBOARD_CHART_AGG = """
        SELECT
            date AS query_date,
            source_id,
            energy_model_id,
            efficiency_meters_per_joules,
            efficiency_miles_per_gallon,
            throughput_per_hour
        FROM  leaderboard_chart
        ORDER BY query_date, source_id ASC
        ;"""
