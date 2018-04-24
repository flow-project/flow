# Setting up an rllib cluster
- cd into scripts
- If you haven't set up before, comment out file_mounts and setup_commands
- run ray create_or_update ray_autoscale.yaml
- Enter into the ~/.ssh and run  ssh-keygen -y -f <NAME OF RAY KEY.pem>
- Copy the resulting key and add it as a key to your github following these instructions: https://help.github.com/articles/adding-a-new-ssh-key-to-your-github-account/#platform-linux
- Uncomment out file mount and setup_commands
- Change tmp/foo to have your path and point to the desired branch
- Make sure that branch is pushed up to github
- Change tmp/foo2 to be the path to your ray key
- run ray create_or_update ray_autoscale.yaml
- Log into your cluster, change redis_address in ray.init(redis_address="ADDRESS") to the redis address output by create_or_update
- Run your code! Make sure to kill the cluster when you're done

## Notes

- If you want to update your branch across all workers, push it to github and
    rerun ray create_or_update
- *Save the instructions after ray create_or_update, they're useful!*
- If you see "fatal: not a tree", you need to push your branch to github
- If it fails because of "stash your changes", go to the node and stash your changes