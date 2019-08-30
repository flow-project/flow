# trainer.py
import os
import sys
import time
import ray



ray.init(redis_address=os.environ["ip_head"])

@ray.remote
def f():
  time.sleep(1)

# The following takes one second (assuming that ray was able to access all of the allocated nodes).
start = time.time()
num_cpus = int(sys.argv[1])
ray.get([f.remote() for _ in range(num_cpus)])
end = time.time()
print(end - start)
