#!/bin/bash
INPUT=$1
OLDIFS=$IFS
IFS=','
[ ! -f $INPUT ] && { echo "$INPUT file not found"; exit 99; }
while read a b noise v0 T delta s0 acc dec sigma tau mg ms sf sd i hor maxAcc maxDec tV len lane sl edgNum sstep lres srad expNum 
do      
        echo  "================== Experiment $expNum ===================" 
        echo  "******* IDM Parameters ********"
	echo -e "\tAcceleration : $a"
	echo -e "\tDeceleration : $b"
	echo -e "\tNoise : $noise"
	echo -e "\tDesirable Velocity : $v0"
	echo -e "\tSafe Time Headway : $T"
	echo -e "\tAcceleration exponent : $delta"
	echo -e "\tLinear Jam Distance : $s0"
        echo "******* Car Following Parameters ********"
        echo -e "	accel : $acc"	
        echo -e "	decel : $dec"	
        echo -e "	sigma : $sigma"	
        echo -e "	tau : $tau"	
        echo -e "	min_gap : $mg"	
        echo -e "	max_speed : $ms"	
        echo -e "	speed_factor : $sf"	
        echo -e "	speed_dev : $sd"	
        echo -e "	impatience : $i"	
        echo "******* Environment Parameters ********"
        echo -e "\tHorizon : $hor"
        echo -e "\tMax Accel : $maxAcc"
        echo -e "\tMax Decel : $maxDec"
        echo -e "\tTarget Velocity : $tV"
        echo "******* Network Parameters ********"
        echo -e "\tLength : $len"
        echo -e "\tLanes : $lane"
        echo -e "\tSpeedLimit : $sl"
        echo -e "\tNumber Of Edges : $edgNum"
        echo "******* SUMO Parameters ********"
        echo -e "\tSim Step : $sstep"
        echo -e "\tLateral Resolution : $lres"
        echo -e "\tSight Radius : $srad"
        echo "Starting simulation with the given parameters ... " 
        echo "Experiment $expNum complete!"
        echo "==================================================="
	echo ""
done < $INPUT
IFS=$OLDIFS

#need to read data from shell script to python sim
#need to typecast in python
#need to create python file that creates these parameters
