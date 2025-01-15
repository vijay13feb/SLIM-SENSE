

### Preprocessing 
Convert the raw CSI values into complex number. 
```bash
python convert_complex.py
```
Combine CSI files participants-wise and store them into ```input_combine```

```bash
python combine.py
```
### Denoising- Phase saniization

```bash
python preprocessing.py
python CSI_signal_Hestimation.py
python CSI_signal_reconstruction.py
```
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



```bash
python CSI_doppler_create_dataset_train_test.py ./ <'setup> <Number of packets in a sample> <Number of packet for sliding operations> <Number of samples per window> <Number of samples to stride> <labels_activities> <Number of streams * number of antennas>
```
e.g., python CSI_doppler_create_dataset_train_test.py ./ S4 31 1 256 30 Fo,LD,LL,LR,LU,No,Sh 1


Fo - Looking Forward, LD - Looking Down, LL - Looking Left, LR - Looking Right, LU - Looking Up No - Nodding, Sh - Shaking
### Training and Evaluating the SHARP model 
```bash
python CSI_network.py ./ <setup> <feature_length-set by doppler_computation.py code> <Length along the time dimension (width)> <Number of channel> <Number of samples in a batch> <Number of antenna * number of spatial streams> <Name base> <labels_activities>
```
e.g., python CSI_network.py ./ S4 100 340 1 16  1 temp  Fo,LD,LL,LR,LU,No,Sh
## Python and relevant libraries version
Python >= 3.7.7

TensorFlow >= 2.6.0

Numpy >= 1.19.5

Scipy = 1.4.1

Scikit-learn = 0.23.2

OSQP >= 0.6.1





