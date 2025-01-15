
"""
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
# this is complete and final no doubt in this
import argparse
import numpy as np
import scipy.io as sio
from os import listdir
import pickle
from os import path


def hampel_filter(input_matrix, window_size, n_sigmas=3):
    n = input_matrix.shape[1]
    new_matrix = np.zeros_like(input_matrix)
    k = 1.4826  # scale factor for Gaussian distribution

    for ti in range(n):
        start_time = max(0, ti - window_size)
        end_time = min(n, ti + window_size)
        x0 = np.nanmedian(input_matrix[:, start_time:end_time], axis=1, keepdims=True)
        s0 = k * np.nanmedian(np.abs(input_matrix[:, start_time:end_time] - x0), axis=1)
        mask = (np.abs(input_matrix[:, ti] - x0[:, 0]) > n_sigmas * s0)
        new_matrix[:, ti] = mask*x0[:, 0] + (1 - mask)*input_matrix[:, ti]

    return new_matrix


if __name__ == '__main__':
   
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dir', help='Directory of data')
    parser.add_argument('all_dir', help='All the files in the directory, default no', type=int, default=0)
    parser.add_argument('name', help='Name of experiment file')
    parser.add_argument('nss', help='Number of spatial streams', type=int) # 1
    parser.add_argument('ncore', help='Number of cores', type=int) # 4
    parser.add_argument('start_idx', help='Idx where start processing for each stream', type=int) # 1
    args = parser.parse_args()

    exp_dir = args.dir
    names = []


    if args.all_dir:
        all_files = listdir(exp_dir)
        print("ALL_FILES IS",all_files)
        mat_files = []
        for i in range(len(all_files)):
            if all_files[i].endswith('.mat'):
                names.append(all_files[i][:-4])
    else:
        names.append(args.name)

    for name in names:
        name_file = './phase_processing/signal_' + name + '.txt'
        # if path.exists(name_file):
        #     print('Already processed')
        #     continue

        csi_buff_file = exp_dir + name + ".mat"
        # print(csi_buff_file)
        csi_buff = sio.loadmat(csi_buff_file)
        csi_buff = (csi_buff['csi']) #changed key from csi_buff to csi # 12000 X 2048 X 4 
        length = csi_buff.shape[0]
        print(length) # 12000
        length2 = csi_buff.shape[1]
        print(length2) # features 2048
        a = np.zeros((csi_buff.shape[0]*4, csi_buff.shape[1]), dtype=complex)
        
        
        rows = a.shape[0]
        cols = a.shape[1]
        
        for i in range(0, rows, 4):
            for j in range(0, length2):
                a[i][j] = csi_buff[i//4][j][0]
        
        for i in range(1, rows, 4):
            for j in range(0, length2):
                a[i][j] = csi_buff[i//4][j][1]
                
        for i in range(2, rows, 4):
            for j in range(0, length2):
                a[i][j] = csi_buff[i//4][j][2]
                
        for i in range(3, rows, 4):
            for j in range(0, length2):
                a[i][j] = csi_buff[i//4][j][3]
        
        print(a.shape)
  
        csi_buff = a
        print(csi_buff.shape)

        delete_idxs = np.argwhere(np.sum(csi_buff, axis=1) == 0)[:, 0]
        print(delete_idxs)
        csi_buff = np.delete(csi_buff, delete_idxs, axis=0)
        print("this is after deleting", csi_buff.shape)

        # t1= np.arange(0,12)
        # t2= np.arange(510,515)
        # t3= np.arange(1012, 1035)
        # t4 = np.arange(1533, 1538)
        # t5= np.arange(2037, 2048)
        # t1= np.arange(0,12) # 0-11
        # t2= np.arange(509,515) # 509 to 514
        # t3= np.arange(1013, 1036) # 1013 - 1035
        # t4 = np.arange(1535, 1540) 
        # t5= np.arange(2036, 2048)

        # delete_idxs =  np.concatenate([t1,t2,t3,t4,t5], dtype=int)
        delete_idxs = np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 509, 510, 511, 512, 513, 514, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1032, 1033, 1034, 1035, 1535, 1536, 1537, 1538, 1539, 2036, 2037, 2038, 2039, 2040, 2041, 2042, 2043, 2044, 2045, 2046, 2047], dtype=int)

        n_ss = args.nss
        n_core = args.ncore
        n_tot = n_ss * n_core
       
        start = args.start_idx  # 1000
        end = int(np.floor(csi_buff.shape[0]/n_tot))
        print(end)
        print(end)
        signal_complete = np.zeros((csi_buff.shape[1] - delete_idxs.shape[0], end-start, n_tot), dtype=complex)
        # print(f"signal_complteted{signal_complete.shape}")

        for stream in range(0, n_tot):
            signal_stream = csi_buff[stream:end*n_tot + 1:n_tot, :][start:end, :]
            signal_stream[:, 64:] = - signal_stream[:, 64:]

            signal_stream = np.delete(signal_stream, delete_idxs, axis=1)
            mean_signal = np.mean(np.abs(signal_stream), axis=1, keepdims=True)
            H_m = signal_stream/mean_signal
            print(H_m.T.shape)

            signal_complete[:, :, stream] = H_m.T

        name_file = './phase_processing/signal_' + name + '.txt'
        with open(name_file, "wb") as fp:  # Pickling
            pickle.dump(signal_complete, fp)
