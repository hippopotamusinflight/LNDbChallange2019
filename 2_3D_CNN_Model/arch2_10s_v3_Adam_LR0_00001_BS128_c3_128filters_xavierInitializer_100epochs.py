# arch2_10s_v3_Adam_LR0_00001_BS128_c3_128filters_xavierInitializer_100epochs 

# hyperparameters
epochs = 100
batch_size = 128           # powers of 2
learning_rate = 0.00001

cube_dir = '/global/home/hpc4535/LNDbChallenge/LNDb_preprocessing_generate_all_cubes/cubes_for_CNN/all_cubes_10x10x10/'
# cube_dir = '/global/home/hpc4535/LNDbChallenge/LNDb_preprocessing_generate_all_cubes/cubes_for_CNN_testing/all_cubes_10x10x10/'

# Import libraries
import tensorflow as tf
import sklearn.model_selection as sk
import numpy as np
import pandas as pd
import os
import time
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# ------------------------------- #
# preprocessing all cubes of 10x10x10
# if "real_" append [1, 0, 0] as label
# if "fake_" append [0, 1, 0] as label
# if "bkgd_" append [0, 0, 1] as label

all_images = []
all_labels = []
all_cubes = os.listdir(cube_dir)

for npy in all_cubes:
    arr = np.load(os.path.join(cube_dir, npy))
    arr = arr.transpose(2, 1, 0)  # switch from x,y,z to z,y,x for numpy and tensorflow    
    all_images.append(arr)
    if 'real_' in npy.split("/")[-1]:
        all_labels.append([1, 0, 0])
    elif 'fake_' in npy.split("/")[-1]:
        all_labels.append([0, 1, 0])
    elif 'bkgd_' in npy.split("/")[-1]:
        all_labels.append([0, 0, 1])

all_images_arr = np.array(all_images).reshape(-1,10,10,10,1)
all_labels_arr = np.array(all_labels)

# ------------------------------- #
print(all_images_arr.shape)
print(all_labels_arr.shape)

# ------------------------------- #
# sklearn train test split
train_imgs, val_imgs, train_labels, val_labels = \
sk.train_test_split(all_images_arr, all_labels_arr, \
                    test_size = 0.20, random_state = 42)

print(train_imgs.shape, train_labels.shape, val_imgs.shape, val_labels.shape)

# ------------------------------------------------------------------------- #
### design model

# tf.reset_default_graph()

# define network parameters
n_classes = 3                           # may change to 3 or 4 for background/other tissue cubes

# all placeholders are of type float32
x = tf.placeholder(tf.float32, [None,10,10,10,1], name="x_input")   # input placeholder
y = tf.placeholder(tf.float32, [None,n_classes], name="y_output")   # output placeholder
keep_prob = tf.placeholder(tf.float32, name="keep_prob")            # keep drop out prob placeholder
global_step = tf.Variable(0, trainable=False, name="global_step")   # counts optimization steps

# define some conv and maxpool wrappers
def conv3d(x, W, b, name, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv3d(x, W, strides=[1, strides, strides, strides, 1], padding='SAME', name=name)
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool3d(x, name):
    return tf.nn.max_pool3d(x, ksize=[1, 1, 2, 2, 1], \
                            strides=[1, 1, 2, 2, 1], padding='SAME', name=name)

# ------------------------------- #
# define weights and biases dict
weights = {
    'wc1': tf.get_variable('W0', shape=(3,5,5,1,64), initializer=tf.contrib.layers.xavier_initializer()), 
    'wc2': tf.get_variable('W1', shape=(2,2,2,64,64), initializer=tf.contrib.layers.xavier_initializer()), 
    'wc3': tf.get_variable('W2', shape=(3,5,5,64,128), initializer=tf.contrib.layers.xavier_initializer()), 
    # flattening to dense layer for fc layer
    # only 1 max_pool ksize=1, stride=1, goes from 10x10x10 --> 10x5x5
    'wd1': tf.get_variable('W3', shape=(10*5*5*128,250), initializer=tf.contrib.layers.xavier_initializer()),     
    'out': tf.get_variable('W4', shape=(250,n_classes), initializer=tf.contrib.layers.xavier_initializer()), 
}
biases = {
    'bc1': tf.get_variable('B0', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
    'bc2': tf.get_variable('B1', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
    'bc3': tf.get_variable('B2', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
    'bd1': tf.get_variable('B3', shape=(250), initializer=tf.contrib.layers.xavier_initializer()),
    'out': tf.get_variable('B4', shape=(n_classes), initializer=tf.contrib.layers.xavier_initializer()),
}

# ------------------------------- #

def conv_net(x, weights, biases, keep_prob):
    conv1 = conv3d(x, weights['wc1'], biases['bc1'], name="conv1_layer")
    drop1 = tf.nn.dropout(conv1, keep_prob = keep_prob, name="drop1_layer")
    pool1 = maxpool3d(drop1, name="pool1_layer")

    conv2 = conv3d(pool1, weights['wc2'], biases['bc2'], name="conv2_layer")

    conv3 = conv3d(conv2, weights['wc3'], biases['bc3'], name="conv3_layer")

    fc1 = tf.reshape(conv3, [-1, weights['wd1'].get_shape().as_list()[0]], name="flattened_layer")     # flattening
    # connect flattened conv3 neurons with **each and every** neuron in fc layer
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'], name="fc1_layer")     
    fc1 = tf.nn.relu(fc1)
    drop2 = tf.nn.dropout(fc1, keep_prob = keep_prob, name="drop2_layer")
    
    # Output, class prediction
    out = tf.add(tf.matmul(drop2, weights['out']), biases['out'], name="output_layer")
    
    return out

# ------------------------------- #
# pred = output of conv_net, which is the last output layer of neurons (dim = 3)
pred = conv_net(x, weights, biases, keep_prob)

# logits = final output layer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

# initial_learning_rate = 0.7    # standard
# learning_rate = tf.train.exponential_decay(initial_learning_rate,
#                                           global_step=global_step,
#                                           decay_steps=5000, decay_rate=0.95)

# train_step = tf.train.MomentumOptimizer(learning_rate, momentum=0.9, name="MomentumOptTrainer").minimize(cost)

# might try later
train_step = tf.train.AdamOptimizer(learning_rate=learning_rate, name="AdamTrainer").minimize(cost)

# calculate accuracy across all the given images and average them out. 
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# ------------------------------- #

saver = tf.train.Saver(keep_checkpoint_every_n_hours=2)

# tf.reset_default_graph()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())  # all variables (weights, bias) are "trainable" and will be updated
    
    summary_writer = tf.summary.FileWriter('../output/', sess.graph)
    
    for i in range(epochs):
        epoch_start =time.time()
        for batch in range(len(train_imgs)//batch_size):
            batch_x = train_imgs[batch*batch_size:min((batch+1)*batch_size,len(train_imgs))]
            batch_y = train_labels[batch*batch_size:min((batch+1)*batch_size,len(train_labels))]    

            _, loss = sess.run([train_step, cost], feed_dict={x: batch_x, y: batch_y, keep_prob:0.8})

            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y, keep_prob:0.8})
            
#        print("Iter " + str(i) + \
#              ", Loss= " + "{:.6f}".format(loss) + \
#              ", Training Accuracy= " + "{:.5f}".format(acc))      # prints the loss and training accuracy after each epoch

        # Calculate accuracy for validation
        valid_acc = sess.run(accuracy, feed_dict={x: val_imgs, y: val_labels, keep_prob:1.0})

#        print("Optimization Finished!  Validation Accuracy:","{:.5f}".format(valid_acc))

        print("Iter " + str(i+1) + ", Loss = " + "{:.6f}".format(loss) + ", Training Accuracy = " + "{:.5f}".format(acc) + ", Validation Accuracy = " + "{:.5f}".format(valid_acc))

        epoch_end =time.time()
#        print("job took: %2d minutes and %4.2f seconds\n" % ((epoch_end-epoch_start)//60, (epoch_end-epoch_start)%60))

        
    saved_path = saver.save(sess, '../output/arch2_10s_Adam_LR0_00001_BS128_c3_128filters_xavierInitializer_100epochs')    

    summary_writer.close()


# ------------------------------- #
# EOF
