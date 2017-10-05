# Setting up AWS


  - Pull `openai/rllab-distributed` repository and switch to production `sharedPolicyUpdate_release` branch
  ```bash
  git clone https://github.com/openai/rllab-distributed.git
  cd rllab-distributed
  git checkout sharedPolicyUpdate_release
  ```
  - Follow [rllab local setup instructions](https://rllab.readthedocs.io/en/latest/user/installation.html)
  - Follow [rllab cluster setup instructions](http://rllab.readthedocs.io/en/latest/user/cluster.html)
    - If prompted, region = us-west-1
    - Note: the current Docker image path is "evinitsky/flow-distributed". Your `rllab-distributed/rllab/config_personal.py` should reflect that.
    - (Optional): As desired, add to `config_personal.py` files and 
    directories that you do not need uploaded to EC2 for every 
    experiment by modifying `FAST_CODE_SYNC_IGNORES`.
  - Go to `Makefile.template` in `learning-traffic/flow_dev` and update
  the path to your rllab root directory (no trailing slash)
    - The `flow_dev` reference in the Makefile might need to be updated to `flow`
  - (See note below); Run `make prepare` 
  - Try an example! Run any experiment from `flow_dev/examples`, change
   mode to “ec2”
  - You can run it locally by changing the mode to local_docker. If this isn't working, make sure to check that your local docker image is the most current image. 
  - Log into AWS via: https://cathywu.signin.aws.amazon.com/console
  - If you don’t see the instance you just launched (give it a few 
  minutes) running on AWS, then make sure your region is correct (go to
   top right and change to `US West (N. California)` 

## Notes

- When we run experiments on AWS, we create a new instance for each
seed and use the Docker image I created as the VM. Built into the rllab 
script for running experiments in EC2 mode, we upload the rllab root 
directory to AWS. This way, the AWS instance has access to all files it
 might need to successfully run the experiment. Editing that code is
  pretty complicated, so Cathy and I have decided on the following 
  workflow:
  - All code modification will happen in the learning-traffic directory
  - Before each experiment, run the command `make prepare` , which will
   remove the flow_dev directory in rllab root and copy
   `learning-traffic/flow_dev-dev/flow_dev` into your rllab root directory
    - This means if you make modifications to flow_dev in the rllab
    directory, they may be lost
  - Before each experiment, always make sure you have a commit to that 
  exact snapshot of the `flow_dev` directory. This is because you may
   modify flow_dev later. When you want to run `visualizer.py` , which is
    our modified version of `sim_policy.py` , AKA when you want to 
    create rollout plots, you need the files in `rllab/flow_dev` to match
    the files that were there when you originally ran the experiment.
     So when you want to create rollout plots, you will checkout the
      commit that matches when you ran the experiment and run make 
      prepare, then you can create rollout plots in the rllab directory.
  - Ping me if there are issues!
- I recommend cleaning up your rllab directory, especially if you 
notice you’re uploading a large size to AWS (it will tell you how many
 MB you are uploading, should be < 5 MB). The command `make clean` 
 removes the debug directory (since so far we hardcoded that our SUMO
  files go into that directory, this should be changed in the future) 
  and also all XML files in rllab root directory.
