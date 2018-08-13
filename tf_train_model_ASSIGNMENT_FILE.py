import tensorflow as tf
import numpy as np
import os
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

class build_train:
    def __init__(self):
        self.rootPath = os.path.abspath(os.path.join(os.getcwd(), os.pardir))   # DO NOT EDIT
        self.save_dir = self.rootPath + '/tf_model'                             # DO NOT EDIT

    def build_train_network(self, network):

        ############### MNIST DATA #########################################
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)      # DO NOT EDIT
        ############### END OF MNIST DATA ##################################

        ############### CONSTRUCT NEURAL NETWORK MODEL HERE ################

        # Holds array of images
        x = tf.placeholder(tf.float32, [None, 784], name='ph_x')

        # Holds array of labels
        y_ = tf.placeholder(tf.float32, [None, 10], name='ph_y_')

        # We can now implement our first layer. It will consist of convolution, 
        # followed by max pooling. The convolution will compute 32 features for 
        # each 5x5 patch. Its weight tensor will have a shape of [5, 5, 1, 32]. 
        # The first two dimensions are the patch size, the next is the number of 
        # input channels, and the last is the number of output channels. We will 
        # also have a bias vector with a component for each output channel.
        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])

        # To apply the layer, we first reshape x to a 4d tensor, with the second 
        # and third dimensions corresponding to image width and height, and the 
        # final dimension corresponding to the number of color channels.
        x_image = tf.reshape(x, [-1,28,28,1])

        # We then convolve x_image with the weight tensor, add the bias, apply 
        # the ReLU function, and finally max pool. The max_pool_2x2 method will 
        # reduce the image size to 14x14.
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

        # In order to build a deep network, we stack several layers of this type. 
        # The second layer will have 64 features for each 5x5 patch.
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

        # Now that the image size has been reduced to 7x7, we add a fully-connected 
        # layer with 1024 neurons to allow processing on the entire image. We reshape 
        # the tensor from the pooling layer into a batch of vectors, multiply by a
        # weight matrix, add a bias, and apply a ReLU.
        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        # To reduce overfitting, we will apply dropout before the readout layer. We 
        # create a placeholder for the probability that a neuron's output is kept 
        # during dropout. This allows us to turn dropout on during training, and turn 
        # it off during testing. TensorFlow's tf.nn.dropout op automatically handles 
        # scaling neuron outputs in addition to masking them, so dropout just works 
        # without any additional scaling.
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        # Finally, we add a layer, just like for the one layer softmax regression above.
        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])
        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        # Calculating accuracy
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='op_accuracy')
        
        ############# END OF NEURAL NETWORK MODEL ##########################

        ############# CONSTRUCT TRAINING FUNCTION ##########################

        # Creating trainer function with the more sophisticated ADAM optimizer.
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

        ############# END OF TRAINING FUNCTION #############################


        ############# CONSTRUCT TRAINING SESSION ###########################

        saver = tf.train.Saver()                                            # DO NOT EDIT
        sess = tf.InteractiveSession()                                      # DO NOT EDIT
        sess.run(tf.global_variables_initializer())                         # DO NOT EDIT

        # Sizes of batches to train and check accuracy
        batchSize = 50

        # List to store saved data
        plotData =  []

        for i in range(10000):

            # Training on data (accounting for noise to avoid
            # overfitting?)
            batch_xs, batch_ys = mnist.train.next_batch(batchSize)
            sess.run(train_step, feed_dict={ x: batch_xs, y_: batch_ys, keep_prob: 0.5 })

            # Plot every 10 iterations
            iter = i + 1
            if iter % 100 == 0:

                # Calculating train accuracy
                print('Iteration ' + str(iter))
                train = sess.run(accuracy, { x: batch_xs, y_: batch_ys, keep_prob: 1.0 })
                print('Train accuracy: ' + str(train))

                # Calculating validation accuracy
                vxs, vys = mnist.validation.next_batch(batchSize)
                validation = sess.run(accuracy, { x: vxs, y_: vys, keep_prob: 1.0 })
                print('Validation accuracy: ' + str(validation))

                # Calculating test accuracy
                txs, tys = mnist.test.next_batch(batchSize)
                test = sess.run(accuracy, { x: txs, y_: tys, keep_prob: 1.0 })
                print('Test accuracy: ' + str(test) + '\n')

                # Adding data to plotData
                plotData.append({
                    'iteration': iter,
                    'train': train,
                    'validation': validation,
                    'test': test
                })

        ############# END OF TRAINING SESSION ##############################

        ############# SAVE MODEL ###########################################

        saver.save(sess, save_path=self.save_dir, global_step=network)      # DO NOT EDIT
        print('Model Saved')                                                # DO NOT EDIT
        sess.close()                                                        # DO NOT EDIT

        ############# END OF SAVE MODEL ####################################

        ############# OUTPUT ACCURACY PLOT ################################     

        # Plotting train accuracy
        plt.plot(
            [i['iteration'] for i in plotData],
            [i['train'] for i in plotData],
            label='Train'
        )

        # Plotting validation accuracy
        plt.plot(
            [i['iteration'] for i in plotData],
            [i['validation'] for i in plotData],
            label='Validation'
        )

        # Plotting test accuracy
        plt.plot(
            [i['iteration'] for i in plotData],
            [i['test'] for i in plotData],
            label='Test'
        )

        # Adding plot info
        plt.ylabel('Accuracy')
        plt.xlabel('Iterations')
        plt.legend(loc='lower right')

        # Plot
        plt.show()

        ############# END OF ACCURACY PLOT ################################

def weight_variable(shape):
    '''
    Convenience fucntion that lets us create
    weight variables
    '''
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    '''
    Conveneice fucntion that lets us create
    bias variables
    '''
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    '''
    Convenience function that lets us create
    convolutions using consistent strides and
    padding
    '''
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    '''
    Conveneice functions that lets us create
    pooling that always does max pooling over
    2x2 blocks
    '''
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
