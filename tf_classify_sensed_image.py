import tensorflow as tf

from tensorflow.python.framework import tensor_util

from pprint import pprint

import os
from os.path import expanduser
# from tensorflow.examples.tutorials.mnist import input_data

class evaluate_model:
    def __init__(self):
        home = expanduser("~/MODELS")
	#self.rootPath = os.path.abspath(os.path.join(os.getcwd(), os.pardir))   # DO NOT EDIT
        self.rootPath = os.path.abspath(home)
        self.save_dir = self.rootPath + '/tf_model'                             # DO NOT EDIT

    def evaluate_model(self,model_version,input):

        # mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

        tf.reset_default_graph()

        filename = self.save_dir + '-' + str(model_version) + '.meta'           # DO NOT EDIT
        filename2 = self.save_dir + '-' + str(model_version)                    # DO NOT EDIT
        print('Opening Model: ' + str(filename))    
                            # DO NOT EDIT
        print("Model 2:",str(filename2))
        #saver = tf.train.import_meta_graph(filename)                            # DO NOT EDIT
        sess = tf.InteractiveSession()   
                                       # DO NOT EDIT
        # INITIALIZE GLOBAL VARIABLES
        init = [tf.global_variables_initializer(),tf.local_variables_initializer()]
        #GLOBAL OR LOCAL??
        sess.run(init)                             # DO NOT EDIT
        
        saver = tf.train.import_meta_graph(filename)  
        saver.restore(sess, filename2)
        #print(sess.run('W1'))
        #DO NOT EDIT
        graph = tf.get_default_graph()                                          # DO NOT EDIT

        #graph_nodes = None
        
        ##DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING
        ##DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING
        ##DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING
        #graph_nodes = [n for n in graph.node]
        #for v in tf.global_variables():
            #print(sess.run(v))
        '''
        op = sess.graph.get_operations()
        for m in op:
            print("VAL",(m.values()))
            #print("NAME",m)
        '''
        #my_feed_dict = None
        
        '''
        wts = [n for n in op if op=='Const']
        for n in wts:
            print ("Name of node - %s" % n.name)
            print ("Value - ")
            #print (tensor_util.MakeNdarray(n.attr['value'].tensor))
            print (n.value())
        '''
        #with tf.variable_scope('layer2', reuse=True):
            #print(sess.run(b1))
        #tvars = tf.trainable_variables()
        #tvars_vals = sess.run(tvars)
        #for var, val in zip(tvars, tvars_vals):
            #print(var.name,val)
        # accuracy = graph.get_tensor_by_name('op_accuracy:0')
        ##DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING
        ##DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING
        ##DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING
        
        pprint([v.name for v in tf.global_variables()])
        #x = graph.get_tensor_by_name('ph_x:0')
        #print("X TENSOR:",sess.run('ph_x:0'))
        #y_ = graph.get_tensor_by_name('ph_y_:0')
        #y = graph.get_tensor_by_name('op_y:0')
        
        ##DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING
        ##DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING
        ##DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING
        # print("Y TENSOR:",sess.run('op_y:0'),feed_dict = {x:input})
        #y_ = graph.get_tensor_by_name('ph_y_:0')
        #Translation weights 784-10
        #W1= tf.get_variable('W1',[784,10])        
        #Translation offsets 784-10
        #print("GOT W:",(W1))
        #b1 = tf.get_variable('b1',[784,10])        
        #print("GOT B:",(b1))
        #y2 = tf.nn.softmax(tf.matmul(x,W1)+b1, name = "op_y2")
        ##DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING
        ##DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING
        ##DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING
        
        print ("ALL VARS:", [v for v in tf.trainable_variables()])
        b1 = None
        W1 = None
        v0 = None
        v1 = None
        v2 = None
        v3 = None
        v4 = None
        v5 = None
        v6 = None
        v7 = None
        h_fc1_drop = None
        W_fc2 = None
        b_fc2 = None
        for v in tf.global_variables():
           if v.name=='Variable:0':
               print(sess.run(v))
               v0 = sess.run(v)
           if v.name=='Variable_1:0':
                print(sess.run(v))
                v1 = sess.run(v)
           if v.name=='Variable_2:0':
                print(sess.run(v))
                v2 = sess.run(v)
           if v.name=='Variable_3:0':
                print(sess.run(v))
                v3 = sess.run(v)
           if v.name=='Variable_4:0':
                print(sess.run(v))
                v4 = sess.run(v)
           if v.name=='Variable_5:0':
                print(sess.run(v))
                v5 = sess.run(v)
           if v.name=='Variable_6:0':
                print(sess.run(v))
                v6 = sess.run(v)
           if v.name=='Variable_7:0':
                print(sess.run(v))
                v7 = sess.run(v)
        
        if(v1 is None or v2 is None):
            print("MUST FIND VARS GIVEN, Make sure have correct var specification EX, vars may be [W1:0,b1_0] -> [W1_1:0,b1_1:0] ->.....[Wn_n:0,bn_n:0]")
            exit(-1)
            #NO DATA, VARS NOT FOUND
        
        x = tf.placeholder(tf.float32, [None, 784])
        y_ = tf.placeholder(tf.float32, [None, 10])

        x_image = tf.reshape(x, [-1,28,28,1])
        h_conv1 = tf.nn.relu(self.conv2d(x_image,v0)+v1)
        h_pool1 = self.max_pool_2x2(h_conv1)
        
        h_conv2 = tf.nn.relu(self.conv2d(h_pool1,v2)+v3)
        h_pool2 = self.max_pool_2x2(h_conv2)
        
        h_pool2_flat = tf.reshape(h_pool2, [-1,7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,v4)+v5)
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob, name = 'h_fc1_drop2')
        
        
        y= tf.nn.softmax(tf.matmul(h_fc1_drop,v6)+v7,name = 'y_conv2')
        
	
	
        	#SET VECTOR GIVEN WEIGHTS (0) for 1 layer, and offsets! b1!!!!
        print("Y TENSOR:",y)
        output = tf.arg_max(y,1)
        
        ##DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING
        ##DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING
        #prediction = graph.get_tensor_by_name('op_y:0')
        #result = tf.nn.softmax(prediction,name='softmax')
        #print("PREDICTOR:",prediction)
        ##DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING
        ##DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING
        
        #x = None
        
        g = sess.run(y,feed_dict = {x:input, keep_prob:1.0})
        print("RESULT:",g)
        #pprint([out for op in tf.get_default_graph().get_operations() if op.type != 'Placeholder' for out in op.values() if out.dtype == tf.float32])
        print("OUTPUT:",output)
        print("INPUT:",input)
        print('Classifying Image...')
        output2 = sess.run(output, feed_dict={x:input,keep_prob:1.0})
        
        ##DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING
        #out = (sess.run([out for op in tf.get_default_graph().get_operations() if op.type != 'Placeholder' for out in op.values() if out.dtype == tf.float32 and out.shape==[1,10]],feed_dict={x:input}))
        ##DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING DEBUGGING
        
        sess.close()
        #Return output, the classifier
        return output2

    def initialize_uninitialized(self,sess):
        global_vars = tf.global_variables()
        is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
        not_initialized_vars = [v for (v,f) in zip(global_vars, is_not_initialized) if not f]
        #Test only
        for i in not_initialized_vars:
            print(i.name)
        if len(not_initialized_vars):
            sess.run(tf.variables_initializer(not_initialized_vars))
            
    def weight_variable(self,shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self,shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(self,x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self,x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
              
