# configurations

cube_size = 20
case_name = "LNDb-298"
prediction_dir = "/global/home/hpc4535/LNDbChallenge/LNDb_prediction_try1/"

cube_dir = "/global/home/hpc4535/LNDbChallenge/LNDb298_all_cubes/"
# cube_dir = "/global/home/hpc4535/LNDbChallenge/LNDb_prediction_try1/"
model_name = "arch3_20s_Adam_LR0_00001_BS128_c3_128filters_xavierInitializer_500epochs"
model_path = "/global/home/hpc4535/LNDbChallenge/LNDb_CNNmodel_test/output/"
print(model_path + model_name + '.meta')

# Import libraries
import tensorflow as tf
import csv
import numpy as np
import pandas as pd
import os
from glob import glob
import re
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# ---------------------------------------------- #

label_dict = {0: 'nodule',
              1: 'non-nodule',
              2: 'background'}

file_list = glob(cube_dir + "*.npy")
# print(file_list[0:4])

# get coordinate list
coord_list = []
for file in file_list:
    m = re.search('\(.+?\)', file)
    if m:
        found = m.group(0)
        coord_list.append(found)
# print(coord_list[0:4])

# ---------------------------------------------- #
# get all cubes (.npy files) from a case_dir
# convert .npy to np.arrays
# create a list of all cube arrays
# change list of cube arrays to one np.array and reshape

all_cubes_list = []
for npy in file_list:
    arr = np.load(npy)
    arr = arr.transpose(2, 1, 0)      # switch from x,y,z to z,y,x for numpy and tensorflow    
    all_cubes_list.append(arr)

all_cubes_arr = np.array(all_cubes_list).reshape(-1, cube_size, cube_size, cube_size, 1)
print(all_cubes_arr.shape)

# ---------------------------------------------- #

graph = tf.get_default_graph()                                                  # clear any existing graphs just in case
with tf.Session() as sess:                                                      # start a new session
    print("prediction session started...")
    new_saver = tf.train.import_meta_graph(model_path + model_name + '.meta')   # import the graph
    new_saver.restore(sess, model_path + model_name)                            # import the weights
    pred = graph.get_tensor_by_name('output_layer:0')                           # need to restate all the placeholders again
    x = graph.get_tensor_by_name('x_input:0')                                     # easier to just get it from the graph
    keep_prob = graph.get_tensor_by_name('keep_prob:0')
    predict = sess.run(pred, feed_dict = {x:all_cubes_arr, keep_prob:1.0})          # run prediction

    
prediction_probability_list = predict.tolist()
print(prediction_probability_list[0:4])

prediction_list = []
for i in range(len(predict)):
    prediction_list.append(label_dict[np.argmax(predict[i])])
print(prediction_list[0:4])

zippedList =  list(zip(coord_list,
                       prediction_probability_list,
                       prediction_list))
prediction_df = pd.DataFrame(zippedList, columns = ['coordinate_(x,y,z)',
                                                    'pred_prob_(class0,1,2)', 
                                                    'pred_name_(nodule/non-nodule/background)'])
# print(prediction_df)

prediction_filename = case_name + "_predictions_df.csv"
prediction_df.to_csv(os.path.join(prediction_dir, prediction_filename),
                     index=False, quoting = csv.QUOTE_NONE, escapechar = ' ')

# data = pd.read_csv(os.path.join(prediction_dir, prediction_filename))
# data.head()
print("prediction complete for case", case_name, "!!!")
