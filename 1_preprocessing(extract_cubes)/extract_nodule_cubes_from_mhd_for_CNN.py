# extract nodule cubes for CNN

import SimpleITK as sitk
import numpy as np
import pandas as pd
import sys
import os
from glob import glob
import pickle

from normalization import *

# ------------------------------- #
cube_generation_dir = "/global/home/hpc4535/LNDbChallenge/LNDb_preprocessing_generate_all_cubes/"
mhd_path = "/global/project/hpcg1553/LNDbChallenge/LNDb_original_data/testPadding/"
# mhd_path = "/global/project/hpcg1553/LNDbChallenge/LNDb_original_data/data0to4/"
realcube_annotation_path = "/global/home/hpc4535/LNDbChallenge/LNDb_preprocessing_generate_all_cubes/realcube_w_coord.csv"
fakecube_annotation_path = "/global/home/hpc4535/LNDbChallenge/LNDb_preprocessing_generate_all_cubes/fakecube_w_coord.csv"
# plot_output_path = "/global/home/hpc4535/LNDbChallenge/LNDb_preprocessing_generate_all_cubes/cubes_for_plotting/"

sm_cube_output_path = "/global/home/hpc4535/LNDbChallenge/LNDb_preprocessing_generate_all_cubes/cubes_for_CNN/all_cubes_6x6x6/"
med_cube_output_path = "/global/home/hpc4535/LNDbChallenge/LNDb_preprocessing_generate_all_cubes/cubes_for_CNN/all_cubes_10x10x10/"
lg_cube_output_path = "/global/home/hpc4535/LNDbChallenge/LNDb_preprocessing_generate_all_cubes/cubes_for_CNN/all_cubes_20x20x20/"

# ------------------------------- #

def extract_nodule_cubes_from_mhd_for_CNN(mhd_path,\
                                          realOrFake,\
                                          annotation_file,\
                                          sm_cube_output_path,\
                                          med_cube_output_path,\
                                          lg_cube_output_path):
    '''
      this function extracts all nodules,
      use different annotation files and specify real or fake with realOrFake arg
      
      @param: mhd_path :        the path contains all mhd file
      @param: annotation_file:  the annatation csv file, contains every nodules' coordinate
      @param: output_path(s):   the save path of extracted cubes of size 
                                6x6x6, 10x10x10, 20x20x20 npy files,
                                every nodule end up withs three size,
                                each size saved into their own respective directories
    '''    
    file_list = glob(mhd_path + "*.mhd")
    annot_df = pd.read_csv(annotation_file)              
    
    centroid_dict = {}
    
    for img_file in file_list:
    #     print(img_file)                                          # /global/project/hpcg1553/LNDbChallenge/LNDb_original_data/testPadding/LNDb-0304.mhd
        case_name_ext = os.path.basename(img_file)                 # get e.g. LNDb-0304.mhd
        case_name = os.path.splitext(case_name_ext)[0]             # get rid of .mhd
        case_df = annot_df[annot_df["LNDb_ID_0padded"] == case_name] # get all nodules associate with file

        if case_df.shape[0] > 0:                        # some files may not have a nodule--skipping those { not in our case }

            # load the data once
            itk_img = sitk.ReadImage(img_file)
            img_array = sitk.GetArrayFromImage(itk_img) # indexes are z,y,x (notice the ordering)
            
            num_z, height, width = img_array.shape      # height*width constitute the transverse plane (e.g. (316 512 634))
            origin = np.array(itk_img.GetOrigin())      # x,y,z  origin in world coord. (mm) (e.g. [ -177.48591592  -272.72653342 -1232.31600019])
            spacing = np.array(itk_img.GetSpacing())    # spacing of voxels in world coord. (mm) (e.g. [0.54186326 0.54186279 1. ])

            # go through all nodes
            print("begin to process case %s, all %s nodules..." % (str(case_name_ext), str(realOrFake)))
            img_array = img_array.transpose(2,1,0)      # for sequence of axis of v_center later on, transfer to x,y,z (e.g. (634, 512, 316))

#             truncate_hu(img_array)
            img_array_norm = normalization(img_array)

            for node_idx, cur_row in case_df.iterrows():      # node_idx = 746, 747, etc, cur_row is that entire row of df
                print(node_idx)
                node_x = cur_row["x"]
                node_y = cur_row["y"]
                node_z = cur_row["z"]
                nodule_pos_str = str(node_x) + "," + str(node_y) + "," + str(node_z)      # e.g. -61.892730526620475_-98.24648952720491_-989.3160000637768

                # every nodules saved into size of 6x6x6, 10x10x10, 20x20x20
                imgs1 = np.ndarray([6,6,6],dtype=np.float32)
                imgs2 = np.ndarray([10,10,10],dtype=np.float32)
                imgs3 = np.ndarray([20,20,20],dtype=np.float32)
                center = np.array([node_x, node_y, node_z])          # nodule center
                v_center = np.rint((center - origin) / spacing)      # nodule center in voxel coord (still x,y,z ordering) (e.g. [213. 322. 243.])

                centroid_dict[cur_row["CombinedID"]] = v_center
                
                try:
                    # these following are the standard data as input of CNN { just truncates and normalizes }
                    imgs1[:,:,:]=img_array_norm[int(v_center[0]-3):int(v_center[0]+3),int(v_center[1]-3):int(v_center[1]+3),int(v_center[2]-3):int(v_center[2]+3)]
                    imgs2[:,:,:]=img_array_norm[int(v_center[0]-5):int(v_center[0]+5),int(v_center[1]-5):int(v_center[1]+5),int(v_center[2]-5):int(v_center[2]+5)]
                    imgs3[:,:,:]=img_array_norm[int(v_center[0]-10):int(v_center[0]+10),int(v_center[1]-10):int(v_center[1]+10),int(v_center[2]-10):int(v_center[2]+10)]
                    np.save(os.path.join(sm_cube_output_path, "%s_images_%s_%d_pos_(%s)_6x6x6_norm.npy" % (str(realOrFake), str(case_name), node_idx, nodule_pos_str)), imgs1)
                    np.save(os.path.join(med_cube_output_path, "%s_images_%s_%d_pos_(%s)_10x10x10_norm.npy" % (str(realOrFake), str(case_name), node_idx, nodule_pos_str)), imgs2)
                    np.save(os.path.join(lg_cube_output_path, "%s_images_%s_%d_pos_(%s)_20x20x20_norm.npy" % (str(realOrFake), str(case_name), node_idx, nodule_pos_str)), imgs3)

                except Exception as e:
                    print(" process images %s error..." % str(case_name))
                    print(Exception,":", e)
                    traceback.print_exc()

    return centroid_dict

# ------------------------------- #
# extract real cubes for CNN
real_centroid = extract_nodule_cubes_from_mhd_for_CNN(mhd_path=mhd_path,\
                                                      realOrFake="real",\
                                                      annotation_file=realcube_annotation_path,\
                                                      sm_cube_output_path=sm_cube_output_path,\
                                                      med_cube_output_path=med_cube_output_path,\
                                                      lg_cube_output_path=lg_cube_output_path)

# ------------------------------- #
# extract fake cubes for CNN
fake_centroid = extract_nodule_cubes_from_mhd_for_CNN(mhd_path=mhd_path,\
                                                      realOrFake="fake",\
                                                      annotation_file=fakecube_annotation_path,\
                                                      sm_cube_output_path=sm_cube_output_path,\
                                                      med_cube_output_path=med_cube_output_path,\
                                                      lg_cube_output_path=lg_cube_output_path)

# ------------------------------- #
# saving dictionary as text file
def SaveDictionary(dictionary, File):
    with open(File, "wb") as myFile:
        pickle.dump(dictionary, myFile)
        myFile.close()

SaveDictionary(real_centroid, os.path.join(cube_generation_dir, "real_centroid_dict.txt"))
SaveDictionary(fake_centroid, os.path.join(cube_generation_dir, "fake_centroid_dict.txt"))

# ------------------------------- #
# reload dictionary from text file
def LoadDictionary(File):
    with open(File, "rb") as myFile:
        dict = pickle.load(myFile)
        myFile.close()
        return dict

# LoadDictionary(os.path.join(cube_generation_dir, "real_centroid_dict.txt"))
# LoadDictionary(os.path.join(cube_generation_dir, "fake_centroid_dict.txt"))

# ------------------------------- #
# EOF
