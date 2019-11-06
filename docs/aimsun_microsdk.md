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

4. The relative functions such as `evaluateCarFollwoing()`, `computeCarFollowingAccelerationComponentSpeed()` and  `computeCarFollowingDecelerationComponentSpeed()` need to be modified. For instance, an example of IDM car-following model implementation would be:

        bool UseIDM=true;

        double behavioralModelParticular::computeCarFollowingAccelerationComponentSpeed( A2SimVehicle *vehicle, double VelActual, double VelDeseada, double RestoCiclo)
        {
            double speed = 0;
            speed = compute_IDM_acc_speed( (simVehicleParticular*)vehicle, VelActual, VelDeseada, RestoCiclo);

        return speed;
        }
        
        double behavioralModelParticular::computeCarFollowingDecelerationComponentSpeed (A2SimVehicle *vehicle,double Shift,A2SimVehicle *vehicleLeader,double ShiftLeader,bool controlDecelMax, bool aside,int time)
        {
            double speed=0;
            speed = compute_IDM_dec_speed((simVehicleParticular*)vehicle,Shift,(simVehicleParticular*)vehicleLeader,ShiftLeader);
           
            return speed;
        }
        
        double behavioralModelParticular::computeMinimumGap(A2SimVehicle *vehicleUp,A2SimVehicle *vehicleDown, bool VehicleIspVehDw, int time)
        {

            double GapMin=0;
            GapMin = get_IDM_minimum_gap((simVehicleParticular*)vehicleUp,(simVehicleParticular*)vehicleDown,Vup, Vdw, Gap);

            return GapMin;
        }
        
        double behavioralModelParticular::compute_IDM_acc_speed(simVehicleParticular *vehicle,double current_speed,double desired_speed, double sim_step)
        {
            double acceleration=max(vehicle->getDeceleration(),vehicle->getAcceleration()*(1.-pow(current_speed/desired_speed,4)));
            double speed = max(0., current_speed + acceleration * sim_step);

            return speed;
        }

        double behavioralModelParticular::compute_IDM_dec_speed(simVehicleParticular *vehicle,double Shift,simVehicleParticular *leader,double ShiftLeader)
        {
            double a=vehicle->getAcceleration();
            double VelAnterior,PosAnterior,VelAnteriorLeader,PosAnteriorLeader;
            double GapAnterior=vehicle->getGap(Shift,leader,ShiftLeader,PosAnterior,VelAnterior,PosAnteriorLeader,VelAnteriorLeader);
            double DesiredGap=getIDMMinimumGap(vehicle,leader,VelAnterior,VelAnteriorLeader, GapAnterior);
            double X=VelAnterior/vehicle->getFreeFlowSpeed();
            double acceleration=a*(1-pow(X,4)-(DesiredGap/GapAnterior)*(DesiredGap/GapAnterior));
            double speed=max(0.,VelAnterior+acceleration*getSimStep());

            return speed;
        }

        double behavioralModelParticular::get_IDM_minimum_gap(simVehicleParticular* pVehUp,simVehicleParticular* pVehDw,double VelAnterior,double VelAnteriorLeader,double GapAnterior)
        {
            double a=pVehUp->getAcceleration();
            double b=-pVehUp->getDeceleration();
            double DesiredGap=pVehUp->getMinimumDistanceInterVeh()+max(0.,VelAnterior*pVehUp->getMinimumHeadway()+VelAnterior*(VelAnteriorLeader-VelAnterior)/(2*sqrt(a*b)));

            return DesiredGap;
        }

5. After modifying the .cpp file, create a make file and compile the program:

        qmake
        make
        
6.  Copy the xml file as well as the created shared object (.so) files into the plugins folder:
        
        cp 02_CarFollowingFunctionsModel.xml ~/Aimsun_Next_8_4_1/plugins/aimsun/models
        cp libCarFollowingFunctionsModel.so ~/Aimsun_Next_8_4_1/plugins/aimsun/models
        cp libCarFollowingFunctionsModel.so.1 ~/Aimsun_Next_8_4_1/plugins/aimsun/models
        cp libCarFollowingFunctionsModel.so.1.0 ~/Aimsun_Next_8_4_1/plugins/aimsun/models
        cp libCarFollowingFunctionsModel.so.1.0.0 ~/Aimsun_Next_8_4_1/plugins/aimsun/models


7. Enable Aimsun Next to load the plugin: 1) Open the network (for Flow scnearios we load a template in `path/to/flow/flow/utils/aimsun/Aimsun_Flow.ang`) , 2) double click on the Experiment in the Dynamic Scenario, 3) click on the Behaviour tab, 4) enable Activate External Behavioural Model
