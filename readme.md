
This is the implementation of the paper SLIM-SENSE: A Resource Efficient WiFi Sensing Framework towards Integrated
Sensing and Communication. <br> The repository shares both the datasets and the source code of SLIM-SENSE.

### Raw dataset 
We utilized three public dataset and one our own dataset given below: 
1. [Exposing_CSI](https://github.com/ansresearch/exposing-the-csi)
2. [SHARPax](https://ieee-dataport.org/documents/ieee-80211ax-csi-dataset-human-activity-recognition)
3. [SimWiSense](https://github.com/kfoysalhaque/SiMWiSense)
4. HeadGest - we will share the link soon. 

### Preprocess and Create the Doppler phase vector for each subchannels/RUs and create input dataset for training and testing. 
To create the input dataset for training for each dataset use Python scripts. The scripts are:-
1. Exposing_CSI: `SLIM-SENSE/Exposing_CSI` 
2. SHARPax: `SLIM-SENSE/SHARPax`
3. SimWiSense: `SLIM-SENSE/SimWiSense`
4. HeadGest: `SLIM-SENSE/HeadGest`

### Resource selection through SLIM-SENSE 

If you want to perfom the our HRL model to train and select optimal resources for Sensing and rest for Communication. <br>
The scripts are:- <br>
### 1. SLIM-SENSE's HRL model for resource selection through SHARP model <br>
Found in `SLIM-SENSE/hrl`
```bash
 python RL_model.py <'directory of the input dataset specific to raw input dataset'> <'sub-directories of training scenarios, comma-separated'> <'length along the feature dimension (height)'> <'length along the time dimension (width)'> <'number of channels'> <'number of samples in a batch'> <'Number of antenna * number of spatial streams'><'name prefix for the files'> <'activities to be considered, comma-separated'> <'group-size specific to size of Doppler vector'> <'total number of sub-channels/RUs'> <'sub-directories of testing scenarios, comma-separated'>
```

e.g., python RL_model.py ./doppler_traces_ofdm/ S1a40_merge 320 256 1 30 tempsingle_ant A,B,C,D,F 40 8 S2a40_merge,S4a40_merge

### 2. SLIM-SENSE with `75%`, `45%` and `30%` baselines
In this baselines 
```bash
  python lambdaTest.py
```
Note: Thsi code will utilize the script mention as: 
```bash
python CSI_network.py <'directory of the input dataset specific to raw input dataset'> <'sub-directories of training scenarios, comma-separated'> <'length along the feature dimension (height)'> <'length along the time dimension (width)'> <'number of channels'>  <'number of samples in a batch'> <'name prefix for the files'> <'activities to be considered, comma-separated'> <'group_lambda'> <'l1_lambda'> <'group_size'> <'total_group'> <--bandwidth 'bandwidth'> <--sub-band 'index of the sub-band to consider (for 20 MHz and 40 MHz)'> 
```
e.g., python CSI_network.py ./doppler_traces/ S1a_merge 2960 32 1 10 4 single_ant A,B,C,D,F 0.001 0.001 40 74 0,1,2,3

Then run the followig script to test with testing scenarios the trained model with specific antennas combination and L1 lambda and group lambda as: 

```bash
python CSI_network_test.py <'directory of the input dataset specific to raw input dataset'> <'sub-directories of training scenarios, comma-separated'> <'length along the feature dimension (height)'> <'length along the time dimension (width)'> <'number of channels'> <'number of samples in a batch'> <'name prefix for the files'> <'activities to be considered, comma-separated'> <'group_lambda'> <'l1_lambda'> <'group_size'> <'total_group'> <'sub-directories of testing scenarios, comma-separated'> <--bandwidth 'bandwidth'> <--sub-band 'index of the sub-band to consider (for 20 MHz and 40 MHz)'>  
```
  e.g., python CSI_network_test.py ./doppler_traces/ S2a_merge 2960 32 1 30 4 S1a_merge_singleAnt A,B,C,D,F 0.001 0.001 40 74

  ### 3. WiImg as baselines

  
