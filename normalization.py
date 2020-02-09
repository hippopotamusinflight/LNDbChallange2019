

import numpy as np
from sklearn.preprocessing import MinMaxScaler
'''
This function takes an image array, normalize the intensities 
and return the normalized array
'''
def normalization(img_arr):
    x, y, z = img_arr.shape
    img_arr_reshaped = np.reshape(img_arr, newshape=(-1, z))
    scaler = MinMaxScaler()
    scaler.fit(img_arr_reshaped)
    img_arr_norm = scaler.transform(img_arr_reshaped)
    img_arr_norm_reshaped = np.reshape(img_arr_norm, newshape=(x, y, z))
    return img_arr_norm_reshaped


# EOF
