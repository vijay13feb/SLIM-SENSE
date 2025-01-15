1. Preprocess the exposing CSI dataset. 
```bash
python CSI_phase_sanitization_signal_preprocessing.py <'directory of the input data'> <'process all the files in subdirectories (1) or not (0)'> <'name of the file to process (only if 0 in the previous field)'> <'number of spatial streams'> <'number of cores'> <'index where to start the processing for each stream'> 
```
for example: 
python CSI_phase_sanitization_signal_preprocessing.py ../input_files/S1a/ 1 - 1 4 0

```bash
python CSI_phase_sanitization_H_estimation.py <'directory of the input data'> <'process all the files in subdirectories (1) or not (0)'> <'name of the file to process (only if 0 in the previous field)'> <'number of spatial streams'> <'number of cores'> <'index where to start the processing for each stream'> <'index where to stop the processing for each stream'> 
```
e.g., python CSI_phase_sanitization_H_estimation.py ../input_files/S1a/ 0 S1a_E 1 4 0 -1

```bash
python CSI_phase_sanitization_signal_reconstruction.py <'directory of the processed data'> <'directory to save the reconstructed data'> <'number of spatial streams'> <'number of cores'> <'index where to start the processing for each stream'> <'index where to stop the processing for each stream'> 
```
e.g., python CSI_phase_sanitization_signal_reconstruction.py ./phase_processing/ ./processed_phase/ 1 4 0 -1

Doppler computation for each individual RU 
```bash 
python dopplerVecScript.py 

in the dopplerVecScript.py code use the following command. 
e.g., python CSI_doppler_computation_final_grouping.py ./processed_phase/ S1a ./doppler_traces/ 400 400 31 1 -1.2 RU{} {} {} 10
after computing doppler for each RUs. Merge all the RUs doppler and create single merged doppler. 

python CSI_merging_file.py ./doppler_traces/ S1a 74

```


### Dataset creation
```bash
python CSI_merging_file.py <'directory of the Doppler data'> <'sub-directories'>  <'Total RUS'>
```
e.g., python CSI_merging_file.py ./doppler_traces/ S1a 74

- Create the datasets for training and validation
```bash
python CSI_doppler_create_dataset_train.py <'directory of the Doppler data'> <'sub-directories, comma-separated'> <'number of packets in a sample'> <'number of packets for sliding operations'> <'number of samples per window'> <'number of samples for window sliding'> <'labels of the activities to be considered'> <'number of streams * number of antennas'>
```
  e.g., python CSI_doppler_create_dataset_train.py ./doppler_traces/ S1a_merge 31 1 32 10 A,B,C,D,F 4


  - Create the datasets for test
```bash
python CSI_doppler_create_dataset_test.py <'directory of the Doppler data'> <'sub-directories, comma-separated'> <'number of packets in a sample'> <'number of packets for sliding operations'> <'number of samples per window'> <'number of samples for window sliding'> <'labels of the activities to be considered'> <'number of streams * number of antennas'>
```
  e.g., python CSI_doppler_create_dataset_test.py ./doppler_traces/ S2a_merge 31 1 32 10 A,B,C,D,F 4

  ### Train the learning algorithm for HAR
```bash
python CSI_network.py <'directory of the datasets'> <'sub-directories, comma-separated'> <'length along the feature dimension (height)'> <'length along the time dimension (width)'> <'number of channels'>  <'number of samples in a batch'> <'name prefix for the files'> <'activities to be considered, comma-separated'> <'group_lambda'> <'l1_lambda'> <'group_size'> <'total_group'> <--bandwidth 'bandwidth'> <--sub-band 'index of the sub-band to consider (for 20 MHz and 40 MHz)'> 
```
e.g., python CSI_network.py ./doppler_traces/ S1a_merge 2960 32 1 10 4 single_ant A,B,C,D,F 0.001 0.001 40 74 0,1,2,3

- Run the algorithm with the test data 
```bash
python CSI_network_test.py <'directory of the datasets'> <'sub-directories, comma-separated'> <'length along the feature dimension (height)'> <'length along the time dimension (width)'> <'number of channels'> <'number of samples in a batch'> <'name prefix for the files'> <'activities to be considered, comma-separated'> <'group_lambda'> <'l1_lambda'> <'group_size'> <'total_group'> <--bandwidth 'bandwidth'> <--sub-band 'index of the sub-band to consider (for 20 MHz and 40 MHz)'> 
```
  e.g., python CSI_network_test.py ./doppler_traces/ S2a_merge 2960 32 1 30 4 S1a_merge_singleAnt A,B,C,D,F 0.001 0.001 40 74


  ```bash
   training the model with different antennas configurations 


  python CSI_network.py <'directory of the datasets'> <'sub-directories, comma-separated'> <'length along the feature dimension (height)'> <'length along the time dimension (width)'> <'number of channels'> <'number of samples in a batch'> <'Number of antenna * number of spatial streams'><'name prefix for the files'> <'activities to be considered, comma-separated'> <'group_lambda'> <'l1_lambda'> <'group_size'> <'total_group'><'antennas_index'> <--bandwidth 'bandwidth'> <--sub-band 'index of the sub-band to consider (for 20 MHz and 40 MHz)'> 
  total_antennas      atennas_index
  1                   0   1   2   3
  2                   0,1   0,2   0,3   1,2  1,3  2,3 
  3                   0,1,2  0,1,3  0,2,3   1,2,3
  4                   0,1,2,3
   
   
  ```

  e.g., python CSI_network.py ./doppler_traces/ S1a_merge 2960 32 1 10 2 tempsingle_ant A,B,C,D,F 0.001 0.001 40 74 0,2


  e.g., python CSI_network_test.py ./doppler_traces/ S2a_merge 2960 32 1 10 4 S1a_merge_singleAnt A,B,C,D,F 0.001 0.001 40 74 0,1,2,3 S1a_merge


