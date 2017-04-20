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
- TODO(cathywu) what are the units of start/end time?
- "cfg_path" : path for the folder where the cfg XML files will be saved: add.xml, gui.cfg, net.xml, rou.xml, sumo.cfg

Vehicle Params:

- Dictionary of car type tag -> (count, car following controller, lane changing controller) assignments, where controller is a method.
- Specifies number of cars for each type
- "Type" : (Number of cars, Car Following Model, Lane Changing Model)
- "rl": No car following model, action determined by RL algorithm
- All other types can have arbitrary names because their actions/updates are determined by the models (other functions in the tuple)
- Suggestion: instead of having "rl" be specific, we could have it such that None or a RL HOF are recognized as "rl vehicles"; Other suggestion: specifying controlled_vehicle_params and rl_params
- TODO(cathywu) include an example here

### Vehicle Params

Implemented car following models:

- Basic Car Following Model
	- per [Horn 2013](http://ieeexplore.ieee.org/abstract/document/6728204/)
	- Only considers vehicle ahead.
	- Terms for desired velocity and headway gap
- Bilateral Control Model
	- per [Horn 2013](http://ieeexplore.ieee.org/abstract/document/6728204/)
	- Considers vehicle ahead and vehicle behind.
	- Term for desired velocity. Another term to place self halfway between car ahead and car behind.
- Optimal Vehicle Model
	- per [Jin & Gabor 2014](http://www-personal.umich.edu/~orosz/articles/CDC_2014_Jin.pdf)
	- Only considers vehicle ahead.
	- Desired velocity term is a function of headway. Also seeks to match velocity of car ahead.


Lane changing models:

- No lane changing
- Stochastic lane changer
