### SUMO Depart Time Issue:

Issue: When running an experiement using RLLab with  flow_dev, after a large number of steps, or specifiying a small timestep causes the following error when resetting the state of the vehicles:


1. Find your SUMO installation directory (should be in your bash config and be named something like: `sumo-0.27.1`)
2. Open `/src/traci-server/TraCIServerAPI_Vehicle.cpp` and go to line 1200
3. Change the line from: 
4. ```return server.writeErrorStatusCmd(CMD_SET_VEHICLE_VARIABLE, "Departure time in the past.";``` 
to:
```vehicleParams.depart = MSNet::getInstance()->getCurrentTimeStep();```
5. ./configure and make to rebuild

Essentially, we're forcing SUMO to add the car at the current time if it detects that you've specified a time in the past.