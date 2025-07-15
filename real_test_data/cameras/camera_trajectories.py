import json

camera_trajectories = {}
for i in range(347):
    camera_trajectories[f"frame{i}"] = {"cam01": "[1 0 0 0] [0 1 0 0] [0 0 1 0] [3390 1000 240 1]"}

with open('camera_extrinsics.json', 'w') as f:
    json.dump(camera_trajectories, f, indent=4)
    
       

    
    