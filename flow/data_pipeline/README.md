To run a simulation with output and metadata stored locally only:

`python simulate.py EXP_CONFIG --gen_emission`

To run a sumulation and upload output and metadata to AWS:

`python simulate.py EXP_CONFIG --to_aws`

To run a simulation, upload output and metadata to AWS and mark it as baseline:

`python simulate.py EXP_CONFIG --to_aws --is_baseline`