
# Docker Tutorial

- Note: all the hard work is in making a good Dockerfile. You should be able to go into Dockerfile and just update version numbers from now forward. There are comments (denoted by `#` ) for the sections that involve SUMO, including the patch that happens before building. Don’t touch anything outside the SUMO section. Also, if you’re feeling ambitious, check what the rllab latest Dockerfile is and merge in any changes they’ve made.
  - TODO: Update link to sumo patch, must be correct before attempting build. I recommend using AWS S3. 
- Familiarize yourself with the docker tutorial at: https://docs.docker.com/get-started/
- Optional: ssh into lab machine
- Optional: install Docker on lab machine account
- Transfer `Dockerfile` , and `environment.yml` file from rllab into same directory (root)
  - Can use Dropbox public links (wget) or git or AWS
- Note: environment.yml should be exactly the same as the one in rllab, except we have two more dependencies:
  - `libxml2=2.9.4`
  - `libxslt=1.1.29` 
  - Takeaway: when updating `environment.yml` make sure to look at both the old version in cistar_dev and the newer ones in rllab, and don’t lose data
- In the same directory that contains the Dockerfile and environment file, run:
  - `docker build -t <name_of_image> .` 
  - Check it’s there: `docker images` 
  - Create a dockerhub account
  - `docker login` 
  - `docker tag <name> <username>/<name>:latest` 
  - `docker push <username>/<name>:latest` 
  - Now update your config_personal to point to the right image! `<username>/<name>` 
  - If you run out of space there are great tutorials online about removing floating containers and images
