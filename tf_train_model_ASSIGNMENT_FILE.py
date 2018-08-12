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

        # MODEL
        # INPUT MUST BE 784 array in order be able to train on MNIST
        # INPUT PLACEHOLDERS MUST BE NAME AS name='ph_x' AND name='ph_y_'
        '''
        Follow following format for defining placeholders:
        x = tf.placeholder(data_type, array_shape, name='ph_x')
        y_ = tf.placeholder(data_type, array_shape, name='ph_y_')
        '''
        # OUTPUT VECTOR y MUST BE LENGTH 10, EACH OUTPUT NEURON CORRESPONDS TO A DIGIT 0-9
        x = tf.placeholder(tf.float32, [None, 784])
        y_ = tf.placeholder(tf.float32, [None, 10])
        W = tf.Variable(tf.zeros([784, 10]))
        b = tf.Variable(tf.zeros([10]))
        y = tf.matmul(x, W) + b

        # LOSS FUNCTION, PREDICTION FUNCTION, ACCURACY FUNCTIONS
        # MAKE SURE ACCURCY FUNCTION IS NAMED ---name='op_accuracy'----
        '''
        EXAMPLE OF NAMING ACCURACY FUNCTION:
        accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32), name='op_accuracy')
        '''

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        ############# END OF NEURAL NETWORK MODEL ##########################

        ############# CONSTRUCT TRAINING FUNCTION ##########################

        # TRAINING FUNCTION SHOULD USE YOUR LOSS FUNCTION TO OPTIMIZE THE MODEL PARAMETERS
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)), reduction_indices=[1]))
        train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

        ############# END OF TRAINING FUNCTION #############################


        ############# CONSTRUCT TRAINING SESSION ###########################
        saver = tf.train.Saver()                                            # DO NOT EDIT
        sess = tf.InteractiveSession()                                      # DO NOT EDIT
        sess.run(tf.global_variables_initializer())                         # DO NOT EDIT

        plotData =  []

        for i in range(1000):

            batch_xs, batch_ys = mnist.train.next_batch(100)
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

            iter = i + 1
            if iter % 100 == 0:

                print('Iteration ' + str(iter))
                train = sess.run(accuracy, { x: mnist.train.images, y_: mnist.train.labels })
                print('Train accuracy: ' + str(train))
                validation = sess.run(accuracy, { x: mnist.validation.images, y_: mnist.validation.labels })
                print('Validation accuracy: ' + str(validation))
                test = sess.run(accuracy, { x: mnist.test.images, y_: mnist.test.labels })
                print('Test accuracy: ' + str(test) + '\n')

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

        plt.plot(
            [i['iteration'] for i in plotData],
            [i['train'] for i in plotData],
            label='Train'
        )
        plt.plot(
            [i['iteration'] for i in plotData],
            [i['validation'] for i in plotData],
            label='Validation'
        )
        plt.plot(
            [i['iteration'] for i in plotData],
            [i['test'] for i in plotData],
            label='Test'
        )
        plt.ylabel('Accuracy')
        plt.xlabel('Iterations')
        plt.legend(loc='lower right')
        plt.show()

        ############# END OF ACCURACY PLOT ################################
