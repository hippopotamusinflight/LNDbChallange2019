import SimpleITK as sitk
import numpy as np
from glob import glob
import pandas as pd
import os
import traceback
import pickle

from data_preparing import *

'''
mhd_path is the path containing all original image data

'''
mhd_path = "/global/home/hpc4535/LNDbChallenge/LNDb_original_data/data0to4/"
realcube_annotation_path = "/global/home/hpc4535/LNDbChallenge/LNDb_preprocessing_generate_all_cubes/realcube_w_coord.csv"
fakecube_annotation_path = "/global/home/hpc4535/LNDbChallenge/LNDb_preprocessing_generate_all_cubes/fakecube_w_coord.csv"

sm_cube_output_path = "/global/home/hpc4535/LNDbChallenge/LNDb_preprocessing_generate_all_cubes/cubes_for_CNN/all_cubes_6x6x6/"
med_cube_output_path = "/global/home/hpc4535/LNDbChallenge/LNDb_preprocessing_generate_all_cubes/cubes_for_CNN/all_cubes_10x10x10/"
lg_cube_output_path = "/global/home/hpc4535/LNDbChallenge/LNDb_preprocessing_generate_all_cubes/cubes_for_CNN/all_cubes_20x20x20/"

'''
This function takes a dataframe containing nodule information that 
correspond to the image passed in and calculate the nodule centroid in voxel spacing
@input: df,dataframe contains all nodule information from annotation file
@input: origin, the origin of corresponding image
@input: spacing, the voxel spacing information for corresponding image
@output: centroid_list, a list of voxel coordinate of nodules 
'''
def nodule_centroid_loc(df, origin, spacing):

    centroid_list = []
 
    for node_idx, cur_row in df.iterrows():
        node_x = cur_row["x"]
        node_y = cur_row["y"]
        node_z = cur_row["z"]

        center = np.array([node_x, node_y, node_z])          # nodule center
        v_center = np.rint((center - origin) / spacing)      # nodule center in voxel coord (still x,y,z ordering)
        centroid_list.append(v_center)

    return centroid_list

'''
This function takes an image object and extract background cubes from the image with input size and number
Uses nodule_centroid_loc() function, export the extracted cube to target folder
@input: mhd_path, path containing all image data
@input: fakecube_annotation_path, path containing non-nodule annotation 
@input: realcube_annotation_path, path containing nodule annotation 
@input: cube_size, the size of background cube you want to generate-(6,10,20)
@input: num_to_extract, the number of background cube to be extracted per image
no output
'''
def extract_background(mhd_path,fakecube_annotation_path,realcube_annotation_path,cube_size,num_to_extract):           # takes one img object
    
    file_list = glob(mhd_path + "*.mhd")
    
    df_fakes = pd.read_csv(fakecube_annotation_path) 
    df_reals = pd.read_csv(realcube_annotation_path)
    df_all = pd.concat([df_fakes, df_reals])
    
    half_size = int(cube_size / 2)
    for img_file in file_list: #for each image 
        file_name_ext = os.path.basename(img_file)                 # get e.g. LNDb-0304.mhd
        file_name = os.path.splitext(file_name_ext)[0]             # get rid of .mhd
        case_df = df_all[df_all["LNDb_ID_0padded"] == file_name]   # subset df_all if match file_name
        
        if case_df.shape[0] > 0: #if this case contains at least one nodule 

            itk_img = sitk.ReadImage(img_file)
            img_array = sitk.GetArrayFromImage(itk_img) # indexes are z,y,x (notice the ordering)

            num_z, height, width = img_array.shape      # height*width constitute the transverse plane
            origin = np.array(itk_img.GetOrigin())      # x,y,z  origin in world coord. (mm)
            spacing = np.array(itk_img.GetSpacing())    # spacing of voxels in world coord. (mm)
            dim = np.array(itk_img.GetSize())
            
            nodule_centroid_list = nodule_centroid_loc(case_df, origin, spacing)    # list containing nodule centers    
            img_array = img_array.transpose(2,1,0) #transpose to match the index
            norm_array = normalization(img_array) # normalize whole image
            
            # make a new array with same dimension as original image, mark with 1
            new_array = np.full(dim, 1)
            new_array = new_array.transpose(2,1,0)
            
            # go through all the noduls 
            for nodule_centroid in nodule_centroid_list: 
                [x,y,z] = nodule_centroid
                [x,y,z] = [int(v) for v in [x,y,z]]
                # assume all nodules are size 30x30x30
                x -= 15
                y -= 15        
                z -= 15         
            
                # for each nodule in the list, create a small array with largest dimension(10*10*10) mark with 0
                nodule_array = np.full((cube_size, cube_size, cube_size), 0)
                # go through the list and change all the nodule to 0 at where they correspond to the original image
                new_array[z:z+cube_size, y:y+cube_size, x:x+cube_size] = nodule_array  #transpose(2,1,0) of the centroid index to match array
         
            i = j = k = 100
            bkgd_cube_num = 0
            
            # slide through the created array and find a cube where there is no 0 inside
            # sliding by 10 each time so they look more different 
            # once finds such cube, use this index to extract from the original image 
            
            while bkgd_cube_num < num_to_extract and (i + cube_size) < dim[0]:
                while bkgd_cube_num < num_to_extract and (j + cube_size) < dim[1]:
                    while bkgd_cube_num < num_to_extract and (k + cube_size) < dim[2]:
                        
                        extract_cube = new_array[i:i+cube_size, j:j+cube_size, k:k+cube_size]           # new_array just for checking if actual nodules is hit

                        if 0 not in extract_cube:
                            bkgd_cube = norm_array[i:i+cube_size, j:j+cube_size, k:k+cube_size]          # img_array with actual intensities

                            nodule_pos_str = str(i+half_size) + "," + str(j+half_size) + "," + str(k+half_size)       #                

                            if cube_size == 6: #export to different folder for different sizes
                                np.save(os.path.join(sm_cube_output_path, "bkgd_images_%s_%d_pos_(%s)_6x6x6_Normd.npy" % (str(file_name), bkgd_cube_num+1, nodule_pos_str)), bkgd_cube)                           
                            elif cube_size == 10:
                                np.save(os.path.join(med_cube_output_path, "bkgd_images_%s_%d_pos_(%s)_10x10x10_Normd.npy" % (str(file_name), bkgd_cube_num+1, nodule_pos_str)), bkgd_cube)                           
                            else:
                                np.save(os.path.join(lg_cube_output_path, "bkgd_images_%s_%d_pos_(%s)_20x20x20_Normd.npy" % (str(file_name), bkgd_cube_num+1, nodule_pos_str)), bkgd_cube)                           
                            
                            bkgd_cube_num += 1
                            
                        k += 10 #stride
                    j += 10
                i += 10
        else:
            nodule_list = []



extract_background(mhd_path,fakecube_annotation_path,realcube_annotation_path,6,2)
extract_background(mhd_path,fakecube_annotation_path,realcube_annotation_path,10,2)
extract_background(mhd_path,fakecube_annotation_path,realcube_annotation_path,20,2)
