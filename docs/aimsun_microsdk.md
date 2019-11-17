# Car-following Controller with Aimsun MicroSDK

This documentation walks you through applying car-following controllers (e.g., IDM car-following model) in Aimsun Next using its software development kit (microSDK). All microSDK programs should be coded in C++. This documentation is produced for Aimsun Next 8.4.1 Ubuntu version and doesnt apply to ealier versions of Aimsun Next such as 8.3.1. For further information visit the Aimsun Next help and manuals.

1. The microSDK files are available in the Aimsun Next installation folder:
        
        cd ~/Aimsun_Next_8_4_1/programming/Aimsun\ Next\ microSDK

2. The microSDK folder includes 4 subfolders: ext, include, lib, and samples. Make a copy of the samples folder and keep it for future references and go to the samples directory. Five examples are available in the samples folder including CarFollowingModel, CarFollowingAccelerationModel, LaneChangingModel, GiveWayModel, FullModel. In this document we are focusing on changing the car following models in the 02_CarFollowingFunctionsModel folder: 

        cp samples samples_copy
        cd samples/02_CarFollowingFunctionsModel

3. To change the car-following behavior, the behavioralModelParticular.cpp file needs to be edited using a text editor (e.g., vim):

        cd 02_CarFollowingFunctionsModel
        vim behavioralModelParticular.cpp 

4. The relative functions such as `computeCarFollowingAccelerationComponentSpeed()` and  `computeCarFollowingDecelerationComponentSpeed()` need to be modified. For instance, an example of IDM car-following model implementation would be:

        double behavioralModelParticular::computeCarFollowingAccelerationComponentSpeed(A2SimVehicle *vehicle,double speed,double v0, double sim_step)
        {
        double new_speed = min(v0, getIDMAccelerationSpeed((simVehicleParticular*)vehicle, speed, v0, sim_step));

        return new_speed;
        }

        double behavioralModelParticular::computeCarFollowingDecelerationComponentSpeed(A2SimVehicle *vehicle, double Shift, A2SimVehicle *vehicleLeader, double ShiftLeader, bool controlDecelMax, bool aside, int time)
        {
            double v0 = vehicle->getFreeFlowSpeed();
            double new_speed = min(v0, getIDMDecelerationSpeed((simVehicleParticular*)vehicle,Shift,(simVehicleParticular*)vehicleLeader,ShiftLeader));
 
            return new_speed;
        }

        double behavioralModelParticular::computeMinimumGap(A2SimVehicle *vehicle, A2SimVehicle *leader, bool VehicleIspVehDw, int time )
        {
            double gap_min = 0;
            double gap = 0;
            double x_up, speed_up, x_down, speed_down;
            double shift = 0,shift_leader = 0;
            gap = vehicle->getGap(shift,  leader, shift_leader, x_up, speed_up, x_down, speed_down, time);
            double a=vehicle->getAcceleration();
            double speed = vehicle->getSpeed(vehicle->isUpdated());
            double speed_leader = leader->getSpeed(leader->isUpdated());
            double desired_gap = getIDMMinimumGap((simVehicleParticular*)vehicle,(simVehicleParticular*)leader,speed, speed_leader, gap);

            return desired_gap;
        }

        double behavioralModelParticular::getIDMDecelerationSpeed(simVehicleParticular *vehicle,double Shift,simVehicleParticular *leader,double ShiftLeader)
        {
            double a = vehicle->getAcceleration();
            double v0 = vehicle->getFreeFlowSpeed();
            double sim_step = getSimStep();
            double speed, pos, speed_leader, pos_leader;
            double gap = vehicle->getGap(Shift,leader,ShiftLeader,pos,speed,pos_leader,speed_leader);
            double desired_gap = getIDMMinimumGap(vehicle,leader,speed,speed_leader, gap);
            double acc = a * (1 - pow((speed/v0),4) - pow((desired_gap/gap),2));
            double next_speed = max(0., speed + acc * getSimStep());

            return next_speed;
        }

        double behavioralModelParticular::getIDMMinimumGap(simVehicleParticular *vehicle,simVehicleParticular *leader,double speed,double speed_leader, double gap)
        {
            double a = vehicle->getAcceleration();
            double b = vehicle->getDeceleration();
            double s0 =  vehicle->getMinimumDistanceInterVeh();
            double T = vehicle->getMinimumHeadway();
            double desired_gap = s0 + max(0., speed*T + speed*(speed_leader-speed)/(2*sqrt(a*b)));

            return desired_gap;
        }

        double behavioralModelParticular::getIDMAccelerationSpeed(simVehicleParticular *vehicle,double speed,double v0, double sim_step)
        {
            double a = vehicle->getAcceleration();
            double acc = a * (1 - pow((speed/v0),4));
            double next_speed = max(0., speed + acc * sim_step);

            return next_speed;
        }
5. Define the new functions such as `getIDMDecelerationSpeed()` in the behavioralModelParticular.h header file, make sure to add them as follows:

        double getIDMAccelerationSpeed (simVehicleParticular *vehicle,double speed,double desired_speed, double sim_step);
        double getIDMDecelerationSpeed (simVehicleParticular *vehicle,double Shift,simVehicleParticular *leader,double ShiftLeader);
        double getIDMMinimumGap(simVehicleParticular *vehicle,simVehicleParticular *leader,double speed,double speed_leader, double gap);

5. After modifying the .cpp and .h files, create a make file and compile the program:

        qmake
        make
        
6.  Copy the xml file as well as the created shared object (.so) files into the plugins folder:
        
        cp 02_CarFollowingFunctionsModel.xml ~/Aimsun_Next_8_4_1/plugins/aimsun/models
        cp libCarFollowingFunctionsModel.so ~/Aimsun_Next_8_4_1/plugins/aimsun/models
        cp libCarFollowingFunctionsModel.so.1 ~/Aimsun_Next_8_4_1/plugins/aimsun/models
        cp libCarFollowingFunctionsModel.so.1.0 ~/Aimsun_Next_8_4_1/plugins/aimsun/models
        cp libCarFollowingFunctionsModel.so.1.0.0 ~/Aimsun_Next_8_4_1/plugins/aimsun/models


7. Enable Aimsun Next to load the plugin: 1) Open the network (for Flow scnearios we load a template in `path/to/flow/flow/utils/aimsun/Aimsun_Flow.ang`) , 2) double click on the Experiment in the Dynamic Scenario, 3) click on the Behaviour tab, 4) enable Activate External Behavioural Model
