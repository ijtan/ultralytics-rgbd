import wandb
import numpy as np

from collections import defaultdict
from tqdm import tqdm

ENTITY = 'ijtan'


# PROJECTS = ['YOLOv7', 'YOLOv7-AC']
# METRICS = ['metrics/mAP_0.5', 'metrics/mAP_0.5:0.95']

PROJECTS = ['YOLOv8']
METRICS = ['metrics/mAP50(B)', 'metrics/mAP50-95(B)']


state = "finished"



api = wandb.Api()

runs = []

for PROJECT in PROJECTS:
    proj_runs = api.runs(f"{ENTITY}/{PROJECT}")
    state_runs = [run for run in runs if run.state == state]
    runs.extend(proj_runs)


print(f"Found {len(runs)} runs")
for run in tqdm(runs):
    if run.state != state:
        continue
    
    for METRIC_NAME in METRICS:
        
        values = run.history()[METRIC_NAME].values
        values = values[~np.isnan(values)]

        run.summary[f"{METRIC_NAME}_max"] = np.max(values)
        # run.summary[f"{METRIC_NAME}_min"] = np.min(values)
        # run.summary[f"{METRIC_NAME}_std"] = np.std(values)

    run.summary.update()