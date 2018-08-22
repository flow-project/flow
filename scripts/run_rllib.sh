#!/bin/bash
echo "Setting up Rlib cluster"

# build the script
shutdown=0
background=0
CMD="ray2 submit ray_autoscale.yaml"
if [ $# -gt 0 ]
then
    while [ "$1" != "" ]; do
        case $1 in
            -f | --file )           shift
                                    filename=$1
                                    ;;
            -s | --shutdown )       shutdown=1
                                    ;;
            -b | --background )     background=1
                                    ;;
            -n | --new_cluster )    new_cluster=1
                                    ;;
        esac
        shift
    done
    if [ "$shutdown" = "1" ]; then
        echo $CMD
        CMD+=" --shutdown"
    fi
    if [ "$background" = "1" ]
    then
        CMD+=" --background"
    fi
    CMD+=" $filename"
    echo "the command is $CMD"
else
    echo "please pass the name of a script to run"
fi

# ray create_or_update ray_autoscale.yaml -y
# ray2 setup ray_autoscale.yaml
# eval $(ray2 login_cmd ray_autoscale.yaml)
# "cd learning-traffic && python setup.py develop"

eval $CMD

echo "Running script"