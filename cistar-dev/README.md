# CISTAR Documentation

## Getting Started

- TODO: Tutorial for visualization / plot generating scripts

Sumo Params:

- Port required, recommended set to 8873
- Timestep, recommended is 0.01, default is 1.0
- TODO: Same flags as SUMO Popen, make it more robust

Env Params:

- These will change based on the scenario
- Target Velocity

Net Params:

- For each environment, you should determine which net params are relevant. 
- Used in Generator files that are specific to each scenario?
- "length" : length of the track
- "lanes" : number of lanes
- "speed_limit"
- "resolution" : number of nodes per edge, affects how 'circular' the track appears when visualized but doesn't affect performance [sic] (e.g. if you have 4 edges for a circle and resolution=2 it will display as 12 lines in the gui)
- "net_path" : path for the folder where the net XML files will be saved: edg.xml, .netccfg, nod.xml, typ.xml
- Suggestion: Direct control of naming of XML files

Configuration (Cfg) Params:

- "start_time" : 0
- "end_time" : When the simulation ends, so pick a reasonably large number
- "cfg_path" : path for the folder where the cfg XML files will be saved: add.xml, gui.cfg, net.xml, rou.xml, sumo.cfg

Vehicle Params:

- Specifies number of cars for each type
- "Type" : (Number of cars, Car Following Model, Lane Changing Model)
- "rl": No car following model, action determined by RL algorithm
- All other types can have arbitrary names because their actions/updates are determined by the models (other functions in the tuple)
- Suggestion: instead of having "rl" be specific, we could have it such that None or a RL HOF are recognized as "rl vehicles"; Other suggestion: specifying controlled_vehicle_params and rl_params


### Vehicle Params

Implemented car following models:

- bcm-10
- bcm-15
- make_better_cfm: cfm with max acceleration
	- Needs v_des (destination velocity) specified

Lane changing models:

- Stochastic lane changer