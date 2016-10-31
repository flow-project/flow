import traci

import time


PORT = 8873

def run():
	traci.init(PORT)
	step = 0

	while traci.simulation.getMinExpectedNumber() >0:
		t0 = time.time()
		traci.simulationStep()
		t1 = time.time()
		print("simulation time (ms): ", traci.simulation.getCurrentTime())
		print("simulation step: ", step)
		print("real time for sim step:", t1-t0)
		raw_input("Press enter for next step")
		step += 1
	traci.close()
if __name__ == "__main__":
	run()