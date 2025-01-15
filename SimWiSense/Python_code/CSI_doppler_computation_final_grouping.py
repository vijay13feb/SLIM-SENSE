"""
    Copyright (C) 2022 Marco Cominelli <marco.cominelli@unibs.it>
    Copyright (C) 2022 Francesca Meneghello
    contact: meneghello@dei.unipd.it
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import argparse
import numpy as np
import scipy.io as sio
import math as mt
from scipy.fftpack import fft
from scipy.fftpack import fftshift
from scipy.signal.windows import hann
import pickle
import os
# the code compute doppler traces for given index related to RU for example:
# python CSI_doppler_computation.py ./processed_phase/ S1a ./doppler_traces/ 400 400 31 1 -1.2 RU1 1 26
##############################################################################################RU name and its starting and ending index. 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dir', help='Directory of data') # ./processed_phase/ 
    parser.add_argument('subdirs', help='Sub-directories') # S1as
    
    parser.add_argument('dir_doppler', help='Directory to save the Doppler data') # ./doppler_traces/
    parser.add_argument('start', help='Start processing', type=int) # 400
    parser.add_argument('end', help='End processing (samples from the end)', type=int) # 400
    parser.add_argument('sample_length', help='Number of packet in a sample', type=int) # 31
    parser.add_argument('sliding', help='Number of packet for sliding operations', type=int) # 1
    parser.add_argument('noise_level', help='Level for the noise to be removed', type=float) # -1.2
    parser.add_argument('RU', help='RU for doppler creation')
    parser.add_argument('start_index', help="tells the starting index of the RU tones", type=int)
    parser.add_argument('last_index', help="last index of RU tones", type=int)
    parser.add_argument('doppler_bins', help="doppler_bins", type=int)
    parser.add_argument('--bandwidth', help='Bandwidth in [MHz] to select the subcarriers, can be 20, 40, 80 '
                                            '(default 80)', default=1, required=False, type=int)
    parser.add_argument('--sub_band', help='Sub_band idx in [1, 2, 3, 4] for 20 MHz, [1, 2] for 40 MHz '
                                           '(default 1)', default=1, required=False, type=int)

    # python CSI_doppler_computation.py ./processed_phase/ S1a ./doppler_traces/ 400 400 31 1 -1.2
    args = parser.parse_args()

    num_symbols = args.sample_length  # 31 # Number of packet in a sample'
    middle = int(mt.floor(num_symbols / 2)) # 15
    doppler_bins = args.doppler_bins
    Tc = 6e-3 # sampling of collecting csi data. 6 miliseconds
    fc = 5e9 #central frequency = 5GHz 
    v_light = 3e8 # velocity of light = 3X10^8
    delta_v = round(v_light / (Tc * fc * num_symbols), 3) # equation number 22. 

    sliding = args.sliding # 1
    noise_lev = args.noise_level # -1.2
    bandwidth = args.bandwidth # 80 
    sub_band = args.sub_band # 1
    start_index = args.start_index
    last_index = args.last_index
    list_subdir = args.subdirs # S1a
    RU = args.RU
    last_index = last_index+1
 
    
    for subdir in list_subdir.split(','):
        main_dir= args.dir_doppler+subdir+str(doppler_bins)
        if not os.path.exists(main_dir):
            os.mkdir(main_dir)
        path_doppler = main_dir+'/' + subdir+RU
        print(path_doppler)
        if not os.path.exists(path_doppler):
            os.mkdir(path_doppler)

        exp_dir = args.dir + subdir + '/'

        names = []
        all_files = os.listdir(exp_dir)
        for i in range(len(all_files)):
            names.append(all_files[i][:-4])
        # print(len(names))

        for name in names:
            path_doppler_name = path_doppler + '/' + name + '.txt'
            if os.path.exists(path_doppler_name):
                print("already done")
                continue

            print(path_doppler_name) # like ./doppler_traces/S1a/S1a_C_stream_1.txt
            name_file = exp_dir + name + '.txt'
            # mdic = sio.loadmat(name_file)
            # print("read mat file")
            # csi_matrix_processed = mdic['csi_matrix_processed'] # input size rows = 12000 , column = 2025 , real and imaginary= 2 
            with open(name_file, 'rb') as f:
               csi_matrix_processed = pickle.load(f)

            csi_matrix_processed = csi_matrix_processed[args.start:-args.end, :, :]

            csi_matrix_processed[:, :, 0] = csi_matrix_processed[:, :, 0] / np.mean(csi_matrix_processed[:, :, 0],
                                                                                    axis=1,  keepdims=True)
            # processed all real value and then divide with mean of the values of the real part                                

            csi_matrix_complete = csi_matrix_processed[:, :, 0]*np.exp(1j*csi_matrix_processed[:, :, 1]) # part is of all real and seonnd part is imagin
            print(csi_matrix_complete.shape)

            if bandwidth == 40:
                if sub_band == 1:
                    selected_subcarriers_idxs = np.arange(0, 117, 1)
                elif sub_band == 2:
                    selected_subcarriers_idxs = np.arange(128, 245, 1)
                num_selected_subcarriers = selected_subcarriers_idxs.shape[0]
                csi_matrix_complete = csi_matrix_complete[:, selected_subcarriers_idxs]
            elif bandwidth == 20:
                if sub_band == 1:
                    selected_subcarriers_idxs = np.arange(0, 57, 1)
                elif sub_band == 2:
                    selected_subcarriers_idxs = np.arange(60, 117, 1)
                elif sub_band == 3:
                    selected_subcarriers_idxs = np.arange(128, 185, 1)
                elif sub_band == 4:
                    selected_subcarriers_idxs = np.arange(188, 245, 1)
                num_selected_subcarriers = selected_subcarriers_idxs.shape[0]
                csi_matrix_complete = csi_matrix_complete[:, selected_subcarriers_idxs]
            elif bandwidth == 1:
                # Skip sub_band selection
                
                selected_subcarriers_idxs = np.arange(start_index, last_index, 1)
                num_selected_subcarriers = selected_subcarriers_idxs.shape[0]
                csi_matrix_complete = csi_matrix_complete[:, selected_subcarriers_idxs]

            print(csi_matrix_complete.shape)

            csi_d_profile_list = []
            for i in range(0, csi_matrix_complete.shape[0]-num_symbols, sliding):
                # print(num_symbols)
                csi_matrix_cut = csi_matrix_complete[i:i+num_symbols, :]
                # print("csi_matrix_cut", csi_matrix_cut.shape)
                csi_matrix_cut = np.nan_to_num(csi_matrix_cut)
                # print("csi_matrix_cut", csi_matrix_cut.shape)

                hann_window = np.expand_dims(hann(num_symbols), axis=-1)
                # print('hann_window', hann_window.shape)
                csi_matrix_wind = np.multiply(csi_matrix_cut, hann_window)
                # print('csi_matrix_wind', csi_matrix_wind.shape)

                csi_doppler_prof = fft(csi_matrix_wind, n=doppler_bins, axis=0)   # doppler velocity bins
                # print('csi_d_profile_array', csi_doppler_prof.shape )
                csi_doppler_prof = fftshift(csi_doppler_prof, axes=0)
                # print('csi_d_profile_array', csi_doppler_prof.shape )

                csi_d_map = np.abs(csi_doppler_prof * np.conj(csi_doppler_prof))

                csi_d_map = np.sum(csi_d_map, axis=1)
                # print('csi_d_map', csi_d_map)
                csi_d_profile_list.append(csi_d_map) # 1 x 100
                
            csi_d_profile_array = np.asarray(csi_d_profile_list)
            csi_d_profile_array_max = np.max(csi_d_profile_array, axis=1, keepdims=True)
            csi_d_profile_array = csi_d_profile_array/csi_d_profile_array_max
            print('csi_d_profile_array', csi_d_profile_array)
            csi_d_profile_array[csi_d_profile_array < mt.pow(10, noise_lev)] = mt.pow(10, noise_lev) # (11169, 100)

            with open(path_doppler_name, "wb") as fp:  # Pickling
                pickle.dump(csi_d_profile_array, fp)
