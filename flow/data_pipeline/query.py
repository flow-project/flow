"""stores all the pre-defined query strings."""
from collections import defaultdict
from enum import Enum

# tags for different queries
prerequisites = {
    "TACOMA_FIT_DENOISED_ACCEL": (
        "fact_energy_trace", {"FACT_VEHICLE_TRACE"}
    ),
    "PRIUS_FIT_DENOISED_ACCEL": (
        "fact_energy_trace", {"FACT_VEHICLE_TRACE"}
    ),
    "FACT_SAFETY_METRICS_2D": (
        "fact_safety_metrics", {"FACT_VEHICLE_TRACE"}
    ),
    "FACT_SAFETY_METRICS_3D": (
        "fact_safety_metrics", {"FACT_VEHICLE_TRACE"}
    ),
    "FACT_NETWORK_THROUGHPUT_AGG": (
        "fact_network_throughput_agg", {"FACT_VEHICLE_TRACE"}
    ),
    "FACT_NETWORK_INFLOWS_OUTFLOWS": (
        "fact_network_inflows_outflows", {"FACT_VEHICLE_TRACE"}
    ),
    "FACT_NETWORK_SPEED": (
        "fact_network_speed", {"FACT_VEHICLE_TRACE"}
    ),
    "FACT_VEHICLE_COUNTS_BY_TIME": (
        "fact_vehicle_counts_by_time", {"FACT_VEHICLE_TRACE"}
    ),
    "FACT_VEHICLE_FUEL_EFFICIENCY_AGG": (
        "fact_vehicle_fuel_efficiency_agg", {"FACT_VEHICLE_TRACE",
                                             "TACOMA_FIT_DENOISED_ACCEL"}
    ),
    "FACT_NETWORK_METRICS_BY_DISTANCE_AGG": (
         "fact_network_metrics_by_distance_agg", {"FACT_VEHICLE_TRACE",
                                                  "TACOMA_FIT_DENOISED_ACCEL"}
    ),
    "FACT_NETWORK_METRICS_BY_TIME_AGG": (
         "fact_network_metrics_by_time_agg", {"FACT_VEHICLE_TRACE",
                                              "TACOMA_FIT_DENOISED_ACCEL"}
    ),
    "FACT_VEHICLE_FUEL_EFFICIENCY_BINNED": (
        "fact_vehicle_fuel_efficiency_binned", {"FACT_VEHICLE_FUEL_EFFICIENCY_AGG"}
    ),
    "FACT_NETWORK_FUEL_EFFICIENCY_AGG": (
        "fact_network_fuel_efficiency_agg", {"FACT_VEHICLE_FUEL_EFFICIENCY_AGG"}
    ),
    "FACT_SAFETY_METRICS_AGG": (
        "fact_safety_metrics_agg", {"FACT_SAFETY_METRICS_3D"}
    ),
    "FACT_SAFETY_METRICS_BINNED": (
        "fact_safety_metrics_binned", {"FACT_SAFETY_METRICS_3D"}
    ),
    "LEADERBOARD_CHART": (
        "leaderboard_chart", {"FACT_NETWORK_THROUGHPUT_AGG",
                              "FACT_NETWORK_SPEED",
                              "FACT_NETWORK_FUEL_EFFICIENCY_AGG",
                              "FACT_SAFETY_METRICS_AGG"}
    ),
    "LEADERBOARD_CHART_AGG": (
        "leaderboard_chart_agg", {"LEADERBOARD_CHART"}
    ),
    "FACT_TOP_SCORES": (
        "fact_top_scores", {"LEADERBOARD_CHART_AGG"}
    ),
}

triggers = [
    "FACT_VEHICLE_TRACE",
    "TACOMA_FIT_DENOISED_ACCEL",
    "FACT_VEHICLE_FUEL_EFFICIENCY_AGG",
    "FACT_SAFETY_METRICS_3D",
    "FACT_NETWORK_THROUGHPUT_AGG",
    "FACT_NETWORK_SPEED",
    "FACT_NETWORK_FUEL_EFFICIENCY_AGG",
    "FACT_SAFETY_METRICS_AGG",
    "LEADERBOARD_CHART",
    "LEADERBOARD_CHART_AGG"
]

tables = [
    "fact_vehicle_trace",
    "fact_energy_trace",
    "fact_vehicle_counts_by_time",
    "fact_safety_metrics",
    "fact_safety_metrics_agg",
    "fact_safety_metrics_binned",
    "fact_network_throughput_agg",
    "fact_network_inflows_outflows",
    "fact_network_speed",
    "fact_vehicle_fuel_efficiency_agg",
    "fact_vehicle_fuel_efficiency_binned",
    "fact_network_metrics_by_distance_agg",
    "fact_network_metrics_by_time_agg",
    "fact_network_fuel_efficiency_agg",
    "leaderboard_chart",
    "leaderboard_chart_agg",
    "fact_top_scores",
    "metadata_table"
]

summary_tables = ["leaderboard_chart_agg", "fact_top_scores"]

network_filters = defaultdict(lambda: {
        'loc_filter': "x BETWEEN 500 AND 2300",
        'warmup_steps': 500 * 3 * 0.4,
        'horizon_steps': 1000 * 3 * 0.4
    })
network_filters['I-210 without Ramps'] = {
        'loc_filter': "edge_id <> ALL (VALUES 'ghost0', '119257908#3')",
        'warmup_steps': 600 * 3 * 0.4,
        'horizon_steps': 1000 * 3 * 0.4
    }

max_decel = -1.0
leader_max_decel = -2.0

VEHICLE_POWER_DEMAND_TACOMA_FINAL_SELECT = """
    SELECT
        id,
        time_step,
        speed,
        acceleration,
        road_grade,
        GREATEST(0, 2041 * acceleration * speed +
            3405.5481762 +
            83.12392997 * speed +
            6.7650718327 * POW(speed,2) +
            0.7041355229 * POW(speed,3)
            ) + GREATEST(0, 4598.7155 * acceleration + 975.12719 * acceleration * speed) AS power,
        'TACOMA_FIT_DENOISED_ACCEL' AS energy_model_id,
        source_id
    FROM {}
    ORDER BY id, time_step
    """

VEHICLE_POWER_DEMAND_PRIUS_FINAL_SELECT = """
    , pmod_calculation AS (
        SELECT
            id,
            time_step,
            speed,
            acceleration,
            road_grade,
            GREATEST(1663 * acceleration * speed +
                1.046 +
                119.166 * speed +
                0.337 * POW(speed,2) +
                0.383 * POW(speed,3) +
                GREATEST(0, 296.66 * acceleration * speed)) AS p_mod,
            source_id
        FROM {}
    )
    SELECT
        id,
        time_step,
        speed,
        acceleration,
        road_grade,
        GREATEST(p_mod, 0.869 * p_mod, -2338 * speed) AS power,
        'PRIUS_FIT_DENOISED_ACCEL' AS energy_model_id,
        source_id
    FROM pmod_calculation
    ORDER BY id, time_step
    """

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
        {}"""


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

    TACOMA_FIT_DENOISED_ACCEL = \
        POWER_DEMAND_MODEL_DENOISED_ACCEL.format(VEHICLE_POWER_DEMAND_TACOMA_FINAL_SELECT.format('denoised_accel_cte'))

    PRIUS_FIT_DENOISED_ACCEL = \
        POWER_DEMAND_MODEL_DENOISED_ACCEL.format(VEHICLE_POWER_DEMAND_PRIUS_FINAL_SELECT.format('denoised_accel_cte'))

    FACT_SAFETY_METRICS_2D = """
        SELECT
            vt.id,
            vt.time_step,
            COALESCE((
                value_lower_left*(headway_upper-headway)*(rel_speed_upper-leader_rel_speed) +
                value_lower_right*(headway-headway_lower)*(rel_speed_upper-leader_rel_speed) +
                value_upper_left*(headway_upper-headway)*(leader_rel_speed-rel_speed_lower) +
                value_upper_right*(headway-headway_lower)*(leader_rel_speed-rel_speed_lower)
            ) / ((headway_upper-headway_lower)*(rel_speed_upper-rel_speed_lower)), 200.0) AS safety_value,
            'v2D_HJI' AS safety_model,
            vt.source_id
        FROM fact_vehicle_trace vt
        LEFT OUTER JOIN fact_safety_matrix sm ON 1 = 1
            AND vt.leader_rel_speed BETWEEN sm.rel_speed_lower AND sm.rel_speed_upper
            AND vt.headway BETWEEN sm.headway_lower AND sm.headway_upper
        WHERE 1 = 1
            AND vt.date = \'{date}\'
            AND vt.partition_name = \'{partition}\'
            AND vt.time_step >= {start_filter}
            AND vt.{loc_filter}
        ;
    """

    FACT_SAFETY_METRICS_3D = """
        SELECT
            id,
            time_step,
            headway + (CASE
                WHEN -speed/{max_decel} > -(speed+leader_rel_speed)/{leader_max_decel} THEN
                    -0.5*POW(leader_rel_speed, 2)/{leader_max_decel} +
                    -0.5*POW(speed,2)/{leader_max_decel} +
                    -speed*leader_rel_speed/{leader_max_decel} +
                    0.5*POW(speed,2)/{max_decel}
                ELSE
                    -leader_rel_speed*speed/{max_decel} +
                    0.5*POW(speed,2)*{leader_max_decel}/POW({max_decel},2) +
                    -0.5*POW(speed,2)/{max_decel}
                END) AS safety_value,
            'v3D' AS safety_model,
            source_id
        FROM fact_vehicle_trace
        WHERE 1 = 1
            AND date = \'{date}\'
            AND partition_name = \'{partition}\'
            AND leader_id IS NOT NULL
            AND time_step >= {start_filter}
            AND {loc_filter}
        ;
    """

    FACT_SAFETY_METRICS_AGG = """
        SELECT
            source_id,
            SUM(CASE WHEN safety_value > 0 THEN 1.0 ELSE 0.0 END) * 100.0 / COUNT() safety_rate,
            MIN(safety_value) AS safety_value_max
        FROM fact_safety_metrics
        WHERE 1 = 1
            AND date = \'{date}\'
            AND partition_name = \'{partition}_FACT_SAFETY_METRICS_3D\'
            AND safety_model = 'v3D'
        GROUP BY 1
        ;
    """

    FACT_SAFETY_METRICS_BINNED = """
        WITH unfilter_bins AS (
            SELECT
                ROW_NUMBER() OVER() - 51 AS lb,
                ROW_NUMBER() OVER() - 50 AS ub
            FROM fact_safety_matrix
        ), bins AS (
            SELECT
                lb,
                ub
            FROM unfilter_bins
            WHERE 1=1
                AND lb >= -5
                AND ub <= 15
        )
        SELECT
            CONCAT('[', CAST(bins.lb AS VARCHAR), ', ', CAST(bins.ub AS VARCHAR), ')') AS safety_value_bin,
            COUNT() AS count
        FROM bins
        LEFT JOIN fact_safety_metrics fsm ON 1 = 1
            AND fsm.date = \'{date}\'
            AND fsm.partition_name = \'{partition}_FACT_SAFETY_METRICS_3D\'
            AND fsm.safety_value >= bins.lb
            AND fsm.safety_value < bins.ub
            AND fsm.safety_model = 'v3D'
        GROUP BY 1
        ;
    """

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
                AND {loc_filter}
            GROUP BY 1, 2
        ), agg AS (
            SELECT
                source_id,
                COUNT(DISTINCT id) AS n_vehicles,
                MAX(enter_time) - MIN(enter_time) AS total_time_seconds
            FROM min_time
            WHERE 1 = 1
                AND enter_time >= {start_filter}
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
                AND e.partition_name = \'{partition}_TACOMA_FIT_DENOISED_ACCEL\'
                AND e.energy_model_id = 'TACOMA_FIT_DENOISED_ACCEL'
                AND e.time_step >= {start_filter}
            WHERE 1 = 1
                AND v.date = \'{date}\'
                AND v.partition_name = \'{partition}\'
                AND v.{loc_filter}
            GROUP BY 1, 2, 3
            HAVING 1 = 1
                AND MAX(distance) - MIN(distance) > 10
                AND COUNT(DISTINCT e.time_step) > 10
        )
        SELECT
            id,
            source_id,
            energy_model_id,
            distance_meters,
            power_watts * time_step_size_seconds AS energy_joules,
            distance_meters / (power_watts * time_step_size_seconds) AS efficiency_meters_per_joules,
            33554.13 * distance_meters / (power_watts * time_step_size_seconds) AS efficiency_miles_per_gallon
        FROM sub_fact_vehicle_trace
        WHERE 1 = 1
            AND power_watts * time_step_size_seconds != 0
        ;
    """

    FACT_VEHICLE_FUEL_EFFICIENCY_BINNED = """
        WITH unfilter_bins AS (
            SELECT
                ROW_NUMBER() OVER() - 1 AS lb,
                ROW_NUMBER() OVER() AS ub
            FROM fact_safety_matrix
        ), bins AS (
            SELECT
                lb,
                ub
            FROM unfilter_bins
            WHERE 1=1
                AND lb >= 0
                AND ub <= 60
        )
        SELECT
            CONCAT('[', CAST(bins.lb AS VARCHAR), ', ', CAST(bins.ub AS VARCHAR), ')') AS fuel_efficiency_bin,
            COUNT() AS count
        FROM bins
        LEFT JOIN fact_vehicle_fuel_efficiency_agg agg ON 1 = 1
            AND agg.date = \'{date}\'
            AND agg.partition_name = \'{partition}_FACT_VEHICLE_FUEL_EFFICIENCY_AGG\'
            AND agg.energy_model_id = 'TACOMA_FIT_DENOISED_ACCEL'
            AND agg.efficiency_miles_per_gallon >= bins.lb
            AND agg.efficiency_miles_per_gallon < bins.ub
        GROUP BY 1
        ;
    """

    FACT_NETWORK_FUEL_EFFICIENCY_AGG = """
        SELECT
            source_id,
            energy_model_id,
            SUM(distance_meters) AS distance_meters,
            SUM(energy_joules) AS energy_joules,
            SUM(distance_meters) / SUM(energy_joules) AS efficiency_meters_per_joules,
            33554.13 * SUM(distance_meters) / SUM(energy_joules) AS efficiency_miles_per_gallon
        FROM fact_vehicle_fuel_efficiency_agg
        WHERE 1 = 1
            AND date = \'{date}\'
            AND partition_name = \'{partition}_FACT_VEHICLE_FUEL_EFFICIENCY_AGG\'
            AND energy_model_id = 'TACOMA_FIT_DENOISED_ACCEL'
        GROUP BY 1, 2
        HAVING 1=1
            AND SUM(energy_joules) != 0
        ;"""

    FACT_NETWORK_SPEED = """
        WITH vehicle_agg AS (
            SELECT
                id,
                source_id,
                AVG(speed) AS vehicle_avg_speed,
                COUNT(DISTINCT time_step) AS n_steps,
                MAX(time_step) - MIN(time_step) AS time_delta,
                MAX(distance) - MIN(distance) AS distance_delta
            FROM fact_vehicle_trace
            WHERE 1 = 1
                AND date = \'{date}\'
                AND partition_name = \'{partition}\'
                AND {loc_filter}
                AND time_step >= {start_filter}
                AND time_step < {stop_filter}
            GROUP BY 1, 2
        )
        SELECT
            source_id,
            SUM(vehicle_avg_speed * n_steps) / SUM(n_steps) AS avg_instantaneous_speed,
            SUM(distance_delta) / SUM(time_delta) AS avg_network_speed
        FROM vehicle_agg
        GROUP BY 1
    ;"""

    LEADERBOARD_CHART = """
        SELECT
            nt.source_id,
            fe.energy_model_id,
            fe.efficiency_meters_per_joules,
            33554.13 * fe.efficiency_meters_per_joules AS efficiency_miles_per_gallon,
            nt.throughput_per_hour,
            ns.avg_instantaneous_speed,
            ns.avg_network_speed,
            sm.safety_rate,
            sm.safety_value_max
        FROM fact_network_throughput_agg AS nt
        JOIN fact_network_speed AS ns ON 1 = 1
            AND ns.date = \'{date}\'
            AND ns.partition_name = \'{partition}_FACT_NETWORK_SPEED\'
            AND nt.source_id = ns.source_id
        JOIN fact_network_fuel_efficiency_agg AS fe ON 1 = 1
            AND fe.date = \'{date}\'
            AND fe.partition_name = \'{partition}_FACT_NETWORK_FUEL_EFFICIENCY_AGG\'
            AND nt.source_id = fe.source_id
            AND fe.energy_model_id = 'TACOMA_FIT_DENOISED_ACCEL'
        JOIN fact_safety_metrics_agg AS sm ON 1 = 1
            AND sm.date = \'{date}\'
            AND sm.partition_name = \'{partition}_FACT_SAFETY_METRICS_AGG\'
            AND nt.source_id = sm.source_id
        WHERE 1 = 1
            AND nt.date = \'{date}\'
            AND nt.partition_name = \'{partition}_FACT_NETWORK_THROUGHPUT_AGG\'
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
                AND {loc_filter}
            GROUP BY 1, 2
        ), inflows AS (
            SELECT
                CAST(min_time_step / 60 AS INTEGER) * 60 AS time_step,
                source_id,
                60 * COUNT(DISTINCT id) AS inflow_rate
            FROM min_max_time_step
            WHERE 1 = 1
                AND min_time_step >= {start_filter}
                AND min_time_step < {stop_filter}
            GROUP BY 1, 2
        ), outflows AS (
            SELECT
                CAST(max_time_step / 60 AS INTEGER) * 60 AS time_step,
                source_id,
                60 * COUNT(DISTINCT id) AS outflow_rate
            FROM min_max_time_step
            WHERE 1 = 1
                AND max_time_step >= {start_filter}
                AND max_time_step < {stop_filter}
            GROUP BY 1, 2
        )
        SELECT
            COALESCE(i.time_step, o.time_step) - MIN(COALESCE(i.time_step, o.time_step))
                OVER (PARTITION BY COALESCE(i.source_id, o.source_id)
                ORDER BY COALESCE(i.time_step, o.time_step) ASC) AS time_step,
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
                    OVER (PARTITION BY vt.id, vt.source_id ORDER BY vt.time_step ASC) AS distance_meters,
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
                AND et.partition_name = \'{partition}_TACOMA_FIT_DENOISED_ACCEL\'
                AND vt.id = et.id
                AND vt.source_id = et.source_id
                AND vt.time_step = et.time_step
                AND et.energy_model_id = 'TACOMA_FIT_DENOISED_ACCEL'
            WHERE 1 = 1
                AND vt.date = \'{date}\'
                AND vt.partition_name = \'{partition}\'
                AND vt.{loc_filter}
                AND vt.time_step >= {start_filter}
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
                AVG(acceleration) AS accel_avg,
                AVG(acceleration) + STDDEV(acceleration) AS accel_upper_bound,
                AVG(acceleration) - STDDEV(acceleration) AS accel_lower_bound,
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
                    OVER (PARTITION BY id, CAST(distance_meters/10 AS INTEGER) * 10
                    ORDER BY time_step ASC) AS energy_start,
                LAST_VALUE(energy_joules)
                    OVER (PARTITION BY id, CAST(distance_meters/10 AS INTEGER) * 10
                    ORDER BY time_step ASC) AS energy_end
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
            bce.source_id AS source_id,
            bce.distance_meters_bin AS distance_meters_bin,
            bce.cumulative_energy_avg,
            bce.cumulative_energy_lower_bound,
            bce.cumulative_energy_upper_bound,
            bce.speed_avg,
            bce.speed_upper_bound,
            bce.speed_lower_bound,
            bce.accel_avg,
            bce.accel_upper_bound,
            bce.accel_lower_bound,
            COALESCE(be.instantaneous_energy_avg, 0) AS instantaneous_energy_avg,
            COALESCE(be.instantaneous_energy_upper_bound, 0) AS instantaneous_energy_upper_bound,
            COALESCE(be.instantaneous_energy_lower_bound, 0) AS instantaneous_energy_lower_bound
        FROM binned_cumulative_energy bce
        JOIN binned_energy be ON 1 = 1
            AND bce.source_id = be.source_id
            AND bce.distance_meters_bin = be.distance_meters_bin
        ORDER BY distance_meters_bin ASC
        ;"""

    FACT_NETWORK_METRICS_BY_TIME_AGG = """
        WITH joined_trace AS (
            SELECT
                vt.id,
                vt.source_id,
                vt.time_step - FIRST_VALUE(vt.time_step)
                    OVER (PARTITION BY vt.id, vt.source_id ORDER BY vt.time_step ASC) AS time_step,
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
                AND et.partition_name = \'{partition}_TACOMA_FIT_DENOISED_ACCEL\'
                AND vt.id = et.id
                AND vt.source_id = et.source_id
                AND vt.time_step = et.time_step
                AND et.energy_model_id = 'TACOMA_FIT_DENOISED_ACCEL'
            WHERE 1 = 1
                AND vt.date = \'{date}\'
                AND vt.partition_name = \'{partition}\'
                AND vt.{loc_filter}
                AND vt.time_step >= {start_filter}
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
                CAST(time_step/10 AS INTEGER) * 10 AS time_seconds_bin,
                AVG(speed) AS speed_avg,
                AVG(speed) + STDDEV(speed) AS speed_upper_bound,
                AVG(speed) - STDDEV(speed) AS speed_lower_bound,
                AVG(acceleration) AS accel_avg,
                AVG(acceleration) + STDDEV(acceleration) AS accel_upper_bound,
                AVG(acceleration) - STDDEV(acceleration) AS accel_lower_bound,
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
                CAST(time_step/10 AS INTEGER) * 10 AS time_seconds_bin,
                FIRST_VALUE(energy_joules)
                    OVER (PARTITION BY id, CAST(time_step/10 AS INTEGER) * 10
                    ORDER BY time_step ASC) AS energy_start,
                LAST_VALUE(energy_joules)
                    OVER (PARTITION BY id, CAST(time_step/10 AS INTEGER) * 10
                    ORDER BY time_step ASC) AS energy_end
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
            bce.source_id AS source_id,
            bce.time_seconds_bin AS time_seconds_bin,
            bce.cumulative_energy_avg,
            bce.cumulative_energy_lower_bound,
            bce.cumulative_energy_upper_bound,
            bce.speed_avg,
            bce.speed_upper_bound,
            bce.speed_lower_bound,
            bce.accel_avg,
            bce.accel_upper_bound,
            bce.accel_lower_bound,
            COALESCE(be.instantaneous_energy_avg, 0) AS instantaneous_energy_avg,
            COALESCE(be.instantaneous_energy_upper_bound, 0) AS instantaneous_energy_upper_bound,
            COALESCE(be.instantaneous_energy_lower_bound, 0) AS instantaneous_energy_lower_bound
        FROM binned_cumulative_energy bce
        JOIN binned_energy be ON 1 = 1
            AND bce.source_id = be.source_id
            AND bce.time_seconds_bin = be.time_seconds_bin
        ORDER BY time_seconds_bin ASC
        ;"""

    FACT_VEHICLE_COUNTS_BY_TIME = """
        WITH counts AS (
            SELECT
                vt.source_id,
                vt.time_step,
                COUNT(DISTINCT vt.id) AS vehicle_count
            FROM fact_vehicle_trace vt
            WHERE 1 = 1
                AND vt.date = \'{date}\'
                AND vt.partition_name = \'{partition}\'
                AND vt.{loc_filter}
                AND vt.time_step >= {start_filter}
            GROUP BY 1, 2
        )
        SELECT
            source_id,
            time_step - FIRST_VALUE(time_step)
                OVER (PARTITION BY source_id ORDER BY time_step ASC) AS time_step,
            vehicle_count
        FROM counts
    ;
    """

    LEADERBOARD_CHART_AGG = """
        WITH agg AS (
            SELECT
                l.date AS submission_date,
                m.submission_time,
                l.source_id,
                m.submitter_name,
                m.strategy,
                m.network,
                m.is_baseline,
                COALESCE (m.penetration_rate, 'x') AS penetration_rate,
                COALESCE (m.version, '2.0') AS version,
                COALESCE (m.road_grade, 'False') AS road_grade,
                COALESCE (m.on_ramp, 'False') AS on_ramp,
                l.energy_model_id,
                l.efficiency_meters_per_joules,
                l.efficiency_miles_per_gallon,
                l.throughput_per_hour,
                l.avg_instantaneous_speed,
                l.avg_network_speed,
                l.safety_rate,
                l.safety_value_max,
                b.source_id AS baseline_source_id
            FROM leaderboard_chart AS l, metadata_table AS m, baseline_table as b
            WHERE 1 = 1
                AND l.source_id = m.source_id
                AND m.network = b.network
                AND (m.is_baseline='False'
                     OR (m.is_baseline='True'
                         AND m.source_id = b.source_id))
        ), joined_cols AS (
            SELECT
                agg.submission_date,
                agg.source_id,
                agg.submitter_name,
                agg.strategy,
                agg.network || ';' ||
                    ' v' || agg.version || ';' ||
                    ' PR: ' || agg.penetration_rate || '%;' ||
                    CASE agg.on_ramp WHEN
                        'True' THEN ' with ramps;'
                        ELSE ' no ramps;' END ||
                    CASE agg.road_grade WHEN
                        'True' THEN ' with grade;'
                        ELSE ' no grade;' END AS network,
                agg.is_baseline,
                agg.energy_model_id,
                agg.efficiency_meters_per_joules,
                agg.efficiency_miles_per_gallon,
                100 * (1 - baseline.efficiency_miles_per_gallon / agg.efficiency_miles_per_gallon)
                    AS fuel_improvement,
                agg.throughput_per_hour,
                100 * (agg.throughput_per_hour - baseline.throughput_per_hour) / baseline.throughput_per_hour
                    AS throughput_change,
                agg.avg_network_speed,
                100 * (agg.avg_network_speed - baseline.avg_network_speed) / baseline.avg_network_speed
                    AS speed_change,
                agg.safety_rate,
                agg.safety_value_max
            FROM agg
            JOIN agg AS baseline ON 1 = 1
                AND agg.network = baseline.network
                AND agg.version = baseline.version
                AND agg.on_ramp = baseline.on_ramp
                AND agg.road_grade = baseline.road_grade
                AND baseline.is_baseline = 'True'
                AND agg.baseline_source_id = baseline.source_id
        )
        SELECT
            submission_date,
            source_id,
            submitter_name,
            strategy,
            network,
            is_baseline,
            energy_model_id,
            efficiency_miles_per_gallon,
            CAST (ROUND(efficiency_miles_per_gallon, 1) AS VARCHAR) ||
                ' (' || (CASE WHEN SIGN(fuel_improvement) = 1 THEN '+' ELSE '' END) ||
                CAST (ROUND(fuel_improvement, 1) AS VARCHAR) || '%)' AS efficiency,
            CAST (ROUND(throughput_per_hour, 1) AS VARCHAR) ||
                ' (' || (CASE WHEN SIGN(throughput_change) = 1 THEN '+' ELSE '' END) ||
                CAST (ROUND(throughput_change, 1) AS VARCHAR) || '%)' AS inflow,
            CAST (ROUND(avg_network_speed, 1) AS VARCHAR) ||
                ' (' || (CASE WHEN SIGN(speed_change) = 1 THEN '+' ELSE '' END) ||
                CAST (ROUND(speed_change, 1) AS VARCHAR) || '%)' AS speed,
            ROUND(safety_rate, 1) AS safety_rate,
            ROUND(safety_value_max, 1) AS safety_value_max
        FROM joined_cols
        ;"""

    FACT_TOP_SCORES = """
        WITH curr_max AS (
            SELECT
                network,
                submission_date,
                MAX(efficiency_miles_per_gallon)
                    OVER (PARTITION BY network ORDER BY submission_date ASC
                    ROWS BETWEEN UNBOUNDED PRECEDING and CURRENT ROW) AS max_score
            FROM leaderboard_chart_agg
            WHERE 1 = 1
                AND is_baseline = 'False'
        ), prev_max AS (
            SELECT
                network,
                submission_date,
                LAG(max_score, 1) OVER (PARTITION BY network ORDER BY submission_date ASC) AS max_score
            FROM curr_max
        ), unioned AS (
            SELECT * FROM curr_max
            UNION ALL
            SELECT * FROM prev_max
        )
        SELECT DISTINCT *
        FROM unioned
        WHERE 1 = 1
            AND max_score IS NOT NULL
        ORDER BY 1, 2, 3
        ;"""
