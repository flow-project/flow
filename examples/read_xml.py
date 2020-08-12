import csv
from collections import defaultdict
import numpy as np

i=0
# data collected every 0.4s
# 2500 steps ie 3000 seconds (*3*0.4)

ids_per_time = defaultdict(list)

filename = './data/I-210_subnetwork_20200812-0117311597187851.349534-0_emission.csv'
with open(filename, mode='r') as infile:
    reader = csv.reader(infile)
    for row in reader:
        if i==0: i+=1; continue
        ids_per_time[float(row[0])].append(row[1])
        # print(row)
        i+=1
        # if i==1000000: break
print(f'csv lines: {i}')

print("time entries:", len(ids_per_time))

total_durations = []
current_ids = set()
durations = defaultdict(int)

times = sorted(ids_per_time)
print(f"times range from {np.min(times)}s to {np.max(times)}s")

for k in times:
    step_ids = set(ids_per_time[k])

    stayed_ids = current_ids & step_ids
    new_ids = step_ids - current_ids
    removed_ids = current_ids - step_ids
    
    for vid in stayed_ids:
        durations[vid] += 1
    for vid in new_ids:
        durations[vid] += 1

    for vid in removed_ids:
        total_durations.append(durations[vid])
        durations[vid] = 0

    current_ids = step_ids

print(len(total_durations), "duration entries")

total_durations = total_durations[1000:]

print(f"mean duration = {np.mean(total_durations)}, std = {np.std(total_durations)}    (steps)")
print(f"mean duration = {np.mean(total_durations)*0.4}, std = {np.std(total_durations)*0.4}    (seconds)")