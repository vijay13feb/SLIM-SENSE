
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
from optimization_utility import *
from os import listdir
import pickle
from plots_utility import * 
# recommnet it to plot the graph. 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dir', help='Directory of data')
    parser.add_argument('all_dir', help='All the files in the directory, default no', type=int, default=0)
    parser.add_argument('name', help='Name of experiment file')
    parser.add_argument('nss', help='Number of spatial streams', type=int)
    parser.add_argument('ncore', help='Number of cores', type=int)
    parser.add_argument('start_r', help='Start processing', type=int)
    parser.add_argument('end_r', help='End processing', type=int)
    args = parser.parse_args()

    exp_save_dir = args.dir
    names = []

    if args.all_dir:
        all_files = listdir(exp_save_dir)
        mat_files = []
        for i in range(len(all_files)):
            if all_files[i].endswith('.mat'):
                names.append(all_files[i][:-4])
    else:
        names.append(args.name)

    for name in names:
        name_file = './phase_processing/signal_' + name + '.txt'
        with open(name_file, "rb") as fp:  # Pickling
            signal_complete = pickle.load(fp)

        t1= np.arange(0,12)
        t2= np.arange(510,515)
        t3= np.arange(1012, 1035)
        t4 = np.arange(1533, 1538)
        t5= np.arange(2037, 2048)

        delete_idxs =  np.concatenate([t1,t2,t3,t4,t5], dtype=int)

        # pilot_subcarriers = [25, 53, 89, 117, 139, 167, 203, 231]
        pilot_subcarriers = [31,99, 165, 233, 273, 341, 407,475, 518,586, 652, 720, 828, 894, 962, 1015, 1083,1217,1257, 1325, 1391,1459, 1497,1565,1631, 1699,1807,1873,1941,1994]
        subcarriers_space = 2 # defines real and imginary
        delta_t = 1E-7 # symbol period or time interval between consecutive symbols in the communication system
        delta_t_refined = 5E-9 # refined or more precise symbol period
        range_refined_up = 2.5E-7 # refined range or time interval used in synchronization or error correction processes
        range_refined_down = 2E-7 # time interval used for synchronization, always less tha "range_refined_up"

        start_r = args.start_r
        if args.end_r != -1:
            end_r = args.end_r
        else:
            end_r = signal_complete.shape[1]

        F_frequency = 2048 # in 11ax total sub carrier are 2048
        delta_f = 78.125E3 ## sub carrier spacing. 
        frequency_vector_complete = np.zeros(F_frequency, )
        F_frequency_2 = F_frequency // 2
        for row in range(F_frequency_2):
            freq_n = delta_f * (row - F_frequency / 2)
            frequency_vector_complete[row] = freq_n
            freq_p = delta_f * row
            frequency_vector_complete[row + F_frequency_2] = freq_p
        frequency_vector = np.delete(frequency_vector_complete, delete_idxs)

        T = 1/delta_f #1/78125 1.28e-5 1 microsecond
        t_min = -3E-7
        t_max = 5E-7
        #These values represent minimum and maximum time limits, set to -3E-7 seconds 
        #(-300 nanoseconds) and 5E-7 seconds (500 nanoseconds), respectively. They could be used to define time 
        #intervals or windows within which signal synchronization and timing corrections are performed.

        T_matrix, time_matrix = build_T_matrix(frequency_vector, delta_t, t_min, t_max)
        r_length = int((t_max - t_min) / delta_t_refined) # # yaha se aaya 160 this is total paths 

        start_subcarrier = 0
        end_subcarrier = frequency_vector.shape[0]
        select_subcarriers = np.arange(start_subcarrier, end_subcarrier, subcarriers_space)

        n_ss = args.nss
        n_core = args.ncore
        n_tot = n_ss * n_core

        # Auxiliary data for first step
        row_T = int(T_matrix.shape[0] / subcarriers_space)
        col_T = T_matrix.shape[1]
        m = 2 * row_T
        n = 2 * col_T
        In = scipy.sparse.eye(n)
        Im = scipy.sparse.eye(m)
        On = scipy.sparse.csc_matrix((n, n))
        Onm = scipy.sparse.csc_matrix((n, m))
        P = scipy.sparse.block_diag([On, Im, On], format='csc')
        q = np.zeros(2 * n + m)
        A2 = scipy.sparse.hstack([In, Onm, -In])
        A3 = scipy.sparse.hstack([In, Onm, In])
        ones_n_matr = np.ones(n)
        zeros_n_matr = np.zeros(n)
        zeros_nm_matr = np.zeros(n + m)

        for stream in range(0, 4):
            name_file = './phase_processing/r_vector' + name + '_stream_' + str(stream) + '.txt'
            signal_considered = signal_complete[:, start_r:end_r, stream]
            r_optim = np.zeros((r_length, end_r - start_r), dtype=complex)
            Tr_matrix = np.zeros((frequency_vector_complete.shape[0], end_r - start_r), dtype=complex)

            for time_step in range(end_r - start_r):
                signal_time = signal_considered[:, time_step]
                complex_opt_r = lasso_regression_osqp_fast(signal_time, T_matrix, select_subcarriers, row_T, col_T,
                                                           Im, Onm, P, q, A2, A3, ones_n_matr, zeros_n_matr,
                                                           zeros_nm_matr)

                position_max_r = np.argmax(abs(complex_opt_r))
                time_max_r = time_matrix[position_max_r]

                T_matrix_refined, time_matrix_refined = build_T_matrix(frequency_vector, delta_t_refined,
                                                                       max(time_max_r - range_refined_down, t_min),
                                                                       min(time_max_r + range_refined_up, t_max))

                # Auxiliary data for second step
                col_T_refined = T_matrix_refined.shape[1]
                n_refined = 2 * col_T_refined
                In_refined = scipy.sparse.eye(n_refined)
                On_refined = scipy.sparse.csc_matrix((n_refined, n_refined))
                Onm_refined = scipy.sparse.csc_matrix((n_refined, m))
                P_refined = scipy.sparse.block_diag([On_refined, Im, On_refined], format='csc')
                q_refined = np.zeros(2 * n_refined + m)
                A2_refined = scipy.sparse.hstack([In_refined, Onm_refined, -In_refined])
                A3_refined = scipy.sparse.hstack([In_refined, Onm_refined, In_refined])
                ones_n_matr_refined = np.ones(n_refined)
                zeros_n_matr_refined = np.zeros(n_refined)
                zeros_nm_matr_refined = np.zeros(n_refined + m)

                complex_opt_r_refined = lasso_regression_osqp_fast(signal_time, T_matrix_refined, select_subcarriers,
                                                                   row_T, col_T_refined, Im, Onm_refined, P_refined,
                                                                   q_refined, A2_refined, A3_refined,
                                                                   ones_n_matr_refined, zeros_n_matr_refined,
                                                                   zeros_nm_matr_refined)

                position_max_r_refined = np.argmax(abs(complex_opt_r_refined))

                T_matrix_refined, time_matrix_refined = build_T_matrix(frequency_vector_complete, delta_t_refined,
                                                                       max(time_max_r - range_refined_down, t_min),
                                                                       min(time_max_r + range_refined_up, t_max))

                Tr = np.multiply(T_matrix_refined, complex_opt_r_refined)

                Tr_sum = np.sum(Tr, axis=1)

                Trr = np.multiply(Tr, np.conj(Tr[:, position_max_r_refined:position_max_r_refined + 1]))
                Trr_sum = np.sum(Trr, axis=1)

                Tr_matrix[:, time_step] = Trr_sum
                time_max_r = time_matrix_refined[position_max_r_refined]

                start_r_opt = int((time_matrix_refined[0] - t_min)/delta_t_refined)
                end_r_opt = start_r_opt + complex_opt_r_refined.shape[0]
                r_optim[start_r_opt:end_r_opt, time_step] = complex_opt_r_refined

            name_file = './phase_processing/r_vector_' + name + '_stream_' + str(stream) + '.txt'
            with open(name_file, "wb") as fp:  # Pickling
                pickle.dump(r_optim, fp)

            name_file = './phase_processing/Tr_vector_' + name + '_stream_' + str(stream) + '.txt'
            with open(name_file, "wb") as fp:  # Pickling
                pickle.dump(Tr_matrix, fp)
