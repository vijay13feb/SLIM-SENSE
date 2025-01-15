import numpy as np
import argparse
import os
import pickle
import sys
# if not os.path.exists(path_doppler):
            #os.mkdir(path_doppler)
# python CSI_merging_file.py ./doppler_traces/ S1a 74

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dir', help='Directory of data') #  ./doppler_traces/
    parser.add_argument('subdirs', help='Sub-directories') # S1a
    parser.add_argument('no_rus', help='Sub-directories', type=int) 
    args = parser.parse_args()

    exdir= args.dir
    subdir= args.subdirs
    no_rus= args.no_rus
    ru_length = no_rus*48
    path_doppler = exdir+subdir+"_merge"
    if not os.path.exists(path_doppler):
        os.mkdir(path_doppler)
    
    all_ = os.listdir(exdir)
    all_paths=[]
    for i in all_:
        if subdir in i and subdir+"_merge" not in i:
            all_paths.append(i)
        else:
            pass

    

    def numeric_order(item):
        return int(''.join(filter(str.isdigit, item)))
    all_paths  = sorted(all_paths , key=numeric_order)
    
    print(all_paths)
    all_paths_RU =[]
    for i in all_paths:
    
        if "S1a_merge" in i:
            pass
        else:
            
            path_all = os.listdir(exdir+i)
           
            all_paths_RU.append(path_all)
            
   
    all_paths_RU[0]  = sorted(all_paths_RU[0] , key=numeric_order)
  
    # with open(name_file, "rb") as fp:  # Unpickling
    #             stft_sum_1 = pickle.load(fp)
    # print("All", all_paths_RU)
    arr_list=[]
    for i in all_paths_RU[0]:
        # print(exdir)
       
        path_temp = exdir+subdir+"/"+i
        # print("path_temp",path_temp)
        complete_dir = os.listdir(path_temp)
        complete_dir.sort()
        # print("complete path: ",path_temp )
        # print(complete_dir)
        
        for j in complete_dir:
        
            if j.startswith(subdir[:-4]): # change here
                path_temp=exdir+subdir+'/'+i+'/'+j
                # print(path_temp)
                with open(path_temp, "rb") as fp:
                    arr = pickle.load(fp)
                arr_list.append(arr)
    print(len(arr_list))
    print(all_paths_RU[0])
    # path_temp = exdir+"/"+all_paths_RU[0][0]
    # complete_dir = os.listdir(path_temp)
    # complete_dir = all_paths_RU[0]
    # complete_dir.sort()
    # print("here is the list ", complete_dir)
    # print(arr_list[0].shape)
    # print(arr_list[40].shape)
    # print(len(arr_list))
    # print(all_paths_RU)
    print("complete_dir",complete_dir)
    for j in range(0,48,1):
        name_f= exdir+subdir+"_merge"+"/"+complete_dir[j]
        # print(name_f)
        merge_list =[]
        if os.path.exists(name_f):
            print("Already Exist")
            pass
        else:
            for i in range(j, ru_length, 48):
                merge_list.append(arr_list[i])
            merge_arr = np.concatenate(merge_list, axis=1)
            print(merge_arr.shape)
            with open(name_f, "wb") as fp:  # Pickling
                pickle.dump(merge_arr, fp)
    

     