To run a simulation with output stored locally only:

    `python simulate.py EXP_CONFIG --gen_emission`
    
To run a simulation and upload output to pipeline:

    `python simulate.py EXP_CONFIG --to_aws`
    
To run a simulation, upload output to pipeline, and mark it as baseline:

    `python simulate.py EXP_CONFIG --to_aws --is_baseline`
    
