# automated script to test all the trained models
# Command

# python CSI_network_test.py ./dopplerTracesBinWise/ S2a_merge,S2b_merge,S2c_merge,S4a_merge,S4b_merge,S4c_merge,S6a_merge,S6b_merge,S6c_merge,S7a_merge,S7b_merge,S7c_merge 740 32 1 10 {numAntenna} S1a_merge,S1b_merge,S1c_merge A,B,C,D,F {groupLambda} {L1Lambda} 10 74 {antennaConfig} S1a_merge,S1b_merge,S1c_merge

import os

def test_models(antennaConfig, groupLambda, L1Lambda, groupSize, totalGroup, filePrefix):
    # Iterate through the antenna configurations
    for num_antennas, configs in antennaConfig.items():
        # ex: num_antennas = 1, configs = [0, 1, 2, 3]
        for config in configs:
            # Convert the antenna configuration to a comma-separated string
            
            if isinstance(config, list):
                antennasIndex = ','.join(map(str, config))
            else:
                antennasIndex = str(config)
                
            print(f"Training model for {num_antennas} antennas with antenna indices {antennasIndex}", flush=True)
            # Format the command string
            command = f"python CSI_network_test.py ./doppler_traces/ S2a_merge 2960 32 1 10 {num_antennas} {filePrefix} A,B,C,D,E,F,G,H,I,J,K,L {groupLambda} {L1Lambda} {groupSize} {totalGroup} {antennasIndex} S1a_merge"

            # Execute the command
            print(f"Executing: {command}", flush=True)
            os.system(command)

# Parameters
groupLambda = [0.01, 0.005, 0.001]
# groupLambda = 0.01
L1Lambda = [0.01, 0.005, 0.001]
# L1Lambda = 0.01
groupSize = 40
totalGroup = 74
filePrefix = "allActivities40_S1a_merge"

# Antenna configurations {number of antennas: [antenna indices]}
antennaConfig = {
    1: [0, 1, 2, 3],
    2: [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]],
    3: [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]],
    4: [[0, 1, 2, 3]]
}

Train the models
for i in range(3):
    test_models(antennaConfig, groupLambda[i], L1Lambda[i], groupSize, totalGroup, filePrefix)
# test_models(antennaConfig, groupLambda, L1Lambda, groupSize, totalGroup, filePrefix)