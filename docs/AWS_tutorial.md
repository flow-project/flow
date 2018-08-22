# Setting up AWS


  ```
  - Follow [rllab local setup instructions](https://rllab.readthedocs.io/en/latest/user/installation.html)
  - Follow [rllab cluster setup instructions](http://rllab.readthedocs.io/en/latest/user/cluster.html)
    - If prompted, region = us-west-1
    - Note: the current Docker image path is "evinitsky/flow". Your `rllab-multiagent/rllab/config_personal.py` should reflect that.
    - (Optional): As desired, add to `config_personal.py` files and 
    directories that you do not need uploaded to EC2 for every 
    experiment by modifying `FAST_CODE_SYNC_IGNORES`.
  the path to your rllab root directory (no trailing slash)
  - Try an example! Run any experiment from `flow/examples`, change
   mode to “ec2”
  - You can run it locally by changing the mode to local_docker. If this isn't working, make sure to check that your local docker image is the most current image. 
  - Log into your AWS
  - If you don’t see the instance you just launched (give it a few 
  minutes) running on AWS, then make sure your region is correct (go to
   top right and change to `US West (N. California)` 

## Notes
   - If you notice that the upload to AWS is taking a long time, make sure 
     you do not have a lot of XML files in your flow directory. 
     The entirety of flow is being uploaded to EC2. 

