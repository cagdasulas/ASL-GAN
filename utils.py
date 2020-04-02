# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 14:06:53 2018

@author: cagdas
"""


import numpy as np
import random
import scipy.io as sc
import h5py
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def select_train_test_subjects(dataset, subject_ind):
    subject_no = len(dataset)
    test_ind = subject_ind-1;
    train_ind = np.setdiff1d(np.arange(subject_no), test_ind)
    train_subjects = [dataset[i] for i in train_ind]
    test_subject = dataset[test_ind]
    return train_subjects, test_subject



def normalize_data(data):
    """  Normalize N-dimensional data (including time series)
    """
    arr = np.reshape(data, [-1])
    mns = np.mean(arr)
    sstd = np.std(arr)
    normalized = (data-mns)/(sstd)
    return normalized


def load_data(filename):
    """ Load a mat file which contains a Matlab struct
    """
    return sc.loadmat(filename, squeeze_me=True, struct_as_record=False)



def expand_vol_4d(input,dim):
    """ Expand 2D data to 4D
    """

    expanded = np.tile(np.expand_dims(np.expand_dims(input, 0), 0), dim)
    return expanded



def expand_vol_1d(input,dim):
    """ Expand data from 3D to 4D
    """

    expanded = np.tile(np.expand_dims(input, 3), dim)
    return expanded



def load_big_data(filename, key):
    """ Load a large .mat file in Python
    """

    f = h5py.File(filename, 'r')
    data = np.transpose(np.array(f.get(key)))
    return data


	
def load_several_keys(filename, key1, key2, key3):
    """ Load only three fields (variables) of a struct in .mat file
    """
    f = h5py.File(filename, 'r')
    data1 = np.transpose(np.array(f.get(key1)))
    data2 = np.transpose(np.array(f.get(key2)))
    data3 = np.transpose(np.array(f.get(key3)))
    return data1, data2, data3


def load_variables(params):
    """ Load ASL parameters into a dictionary object
    """

    var = {}
    ## Assign each variables
    var['alpha'] = tf.constant(params['param'].alpha, dtype=np.float32)
    var['lambda_blood'] = tf.constant(params['param'].lambda_blood, dtype=np.float32)
    var['T1blood'] = tf.constant(params['param'].T1blood, dtype=np.float32)
    var['T1tissue'] = tf.constant(params['param'].T1tissue, dtype=np.float32)
    var['PLDs'] = tf.constant(params['param'].PLDs, np.float32)
    var['tao'] = tf.constant(params['param'].tao, np.float32)
    var['scalar'] = tf.constant(params['param'].scalar_constant, dtype=np.float32)
    return var


def prepare_train_variables(train_subjects):
    
    n_subject = len(train_subjects)
    
    [nb, kx, ky, nt] = np.shape(train_subjects[0].data['xtrain'])
    
    
    xtrain = train_subjects[0].data['xtrain']
    ylabel = train_subjects[0].data['ylabel'] + train_subjects[0].data['yund']
    gt_pwi = train_subjects[0].data['sigData']
    m0     = train_subjects[0].data['M0']
    
    for i in range(n_subject-1):
        xtrain = np.concatenate((xtrain, train_subjects[i+1].data['xtrain']), axis=0)
        yfull = train_subjects[i+1].data['ylabel'] + train_subjects[i+1].data['yund']
        ylabel = np.concatenate((ylabel, yfull), axis=0)
        gt_pwi = np.concatenate((gt_pwi, train_subjects[i+1].data['sigData']), axis=0)
        m0 = np.concatenate((m0, train_subjects[i+1].data['M0']), axis=0)
        
    m0 = expand_vol_1d(m0, [1, 1, 1, nt])
    
    return  xtrain, ylabel, gt_pwi, m0


def arrange_test_variables(test_subject):
    
    Xtest = test_subject.data['xtrain']
    Ytest = test_subject.data['ylabel'] + test_subject.data['yund']
    Yund= test_subject.data['yund'] 
    
    return Xtest, Ytest, Yund




def synthetic_prepare_variables(gt_pwi, est_maps, ylabel, M0):
    """ Stack the ASL variables in a 5D array to be given as input to the network
    """
    [nb, nx, ny, nt] = np.shape(gt_pwi)
    
    nparam = np.shape(est_maps)[-1]
    
    Ytrain = np.zeros((nb, 4, nx, ny, nt))
    
    Ytrain[:,0,:,:,:] = np.concatenate((ylabel, np.zeros((nb,nx,ny,nt-nparam))), axis=-1)
    Ytrain[:,1,:,:,:] = np.concatenate((est_maps, np.zeros((nb,nx,ny,nt-nparam))), axis=-1)
    Ytrain[:,2,:,:,:] = gt_pwi
    Ytrain[:,3,:,:,:] = expand_vol_1d(M0, [1, 1, 1, nt]) # Expand 3D M0 to 4D
    
    return Ytrain 


def split_train_validation(data_interval, val_rate, consecutive_order = False, patch_size_per_im = 4):
    """ Function that returns the randomly selected data indexes to split data into train and validation.
        If the data samples are ordered consecutively (in patch-wise case), then the random selection of the
        indexes are also done from those consecutive index groups to make sure that each selected group can form up the
        entire image.
    """
    if consecutive_order:
        number_of_images = np.size(data_interval)/patch_size_per_im
        index_splits = np.split(data_interval, number_of_images)
        nval = int(np.round(val_rate*len(index_splits)))
        rand_ind_val = np.random.choice(np.arange(len(index_splits)), nval, replace=False)
        rand_ind_train = np.setdiff1d(np.arange(len(index_splits)), rand_ind_val)
        
        index_splits = np.array(index_splits)
        train_ind = np.hstack(index_splits[rand_ind_train,:])
        val_ind = np.hstack(index_splits[rand_ind_val,:])
        
    else:
        rand_ind = np.random.permutation(data_interval)
        nval = int(np.round(val_rate*np.size(data_interval)))
        train_ind = rand_ind[:-nval]
        val_ind =  rand_ind[-nval:]
    
    return train_ind, val_ind



def pick_random_test_samples(Xdata, Ylabel, Yund, sample_size = 2):
    """ Function to randomly pick N samples from test data for validation
    """
    
    idx = np.random.randint(0, Xdata.shape[0], sample_size)
    source_img = Xdata[idx]; y_label = Ylabel[idx]; y_und = Yund[idx]
    
    return source_img, y_label, y_und


def generate_batches(Xdata, Ylabel, Sig, M0, batch_size):
    """ Generate batch of input and output data to be used every iteration of GAN during training
    """

    shapedata=Xdata.shape
    
    #Shuffle data
    idx_rnd=np.random.choice(shapedata[0], shapedata[0], replace=False)
    
    xdata_b=Xdata[idx_rnd,...]
    ylabel_b=Ylabel[idx_rnd,...]
    # yund_b = Yund[idx_rnd,...]
    sig_b = Sig[idx_rnd,...]
    M0_b  = M0[idx_rnd,...]
    
    modulo=np.mod(shapedata[0], batch_size)
    # The number of samples will always be a multiple of batch size
    if modulo!=0:
        to_add=batch_size-modulo
        inds_toadd=np.random.randint(0,xdata_b.shape[0],to_add)
        
        X = arrange_generator(xdata_b, to_add, inds_toadd)
        y = arrange_generator(ylabel_b, to_add, inds_toadd)
        # yund = arrange_generator(yund_b, to_add, inds_toadd)
        sig = arrange_generator(sig_b, to_add, inds_toadd)   
        m0 = arrange_generator(M0_b, to_add, inds_toadd)          
        
    else:
        X=np.copy(xdata_b)                
        y=np.copy(ylabel_b)
        # yund=np.copy(yund_b)
        sig=np.copy(sig_b)
        m0=np.copy(M0_b)

    X=X.astype(np.float32)
    y=y.astype(np.float32) # + yund.astype(np.float32)
    # yund = yund.astype(np.float32)
    sig = sig.astype(np.float32)
    m0 = m0.astype(np.float32)
    
    
    while True:
        for i_batch in range(int(X.shape[0]/batch_size)):
            rng = np.arange(i_batch*batch_size, (i_batch+1)*batch_size)
            yield (X[rng,...],  y[rng,...], sig[rng,...], m0[rng,...])
        
        


def arrange_generator(data, to_add, inds_toadd):
    arranged=np.zeros((data.shape[0]+to_add, data.shape[1], data.shape[2], data.shape[3]))
    arranged[:data.shape[0],...]=data
    arranged[data.shape[0]:,...]=data[inds_toadd]
    
    return arranged


############ FUNCTION AT BELOW ARE NOT USED AT THE MOMENT. HOWEVER, THEY ARE STILL KEPT FOR POSSIBLE FUTURE USE ################


# Define our tensorflow version PRelu
def prelu_tf(inputs, name='Prelu'):
    with tf.variable_scope(name):
        alphas = tf.get_variable('alpha', inputs.get_shape()[-1], initializer=tf.zeros_initializer(), dtype=tf.float32)
    pos = tf.nn.relu(inputs)
    neg = alphas * (inputs - abs(inputs)) * 0.5
    return pos + neg


def conv_op(input_op, name, kw, kh, n_out, dw, dh,wd,padding,activation=True):
    n_in = input_op.get_shape()[-1].value
    shape=[kh, kw, n_in, n_out]
    with tf.variable_scope(name):
        kernel=_variable_with_weight_decay("w", shape, wd)
        conv = tf.nn.conv2d(input_op, kernel, (1, dh, dw, 1), padding=padding)
        bias_init_val = tf.constant(0.0, shape=[n_out], dtype=tf.float32)
        biases = tf.get_variable(initializer=bias_init_val, trainable=True, name='b')
        z = tf.nn.bias_add(conv, biases)
        if activation:
          z=tf.nn.relu(z, name='Activation')
        return z

def conv_op_bn(input_op, name, kw, kh, n_out, dw, dh, wd, padding,train_phase):
    n_in = input_op.get_shape()[-1].value
    shape=[kh, kw, n_in, n_out]
    scope_bn=name+'_bn'
    with tf.variable_scope(name):
        kernel=_variable_with_weight_decay("w", shape, wd)
        conv = tf.nn.conv2d(input_op, kernel, (1, dh, dw, 1), padding=padding)
        bias_init_val = tf.constant(0.0, shape=[n_out], dtype=tf.float32)
        biases = tf.get_variable(initializer=bias_init_val, trainable=True, name='b')
        out_conv = tf.nn.bias_add(conv, biases)
        z=batch_norm_layer(out_conv,train_phase,scope_bn)
        #activation = tf.nn.relu(z, name='Activation')
        return z

def mpool_op(input_op, name, kh, kw, dh, dw):
    return tf.nn.max_pool(input_op,
                          ksize=[1, kh, kw, 1],
                          strides=[1, dh, dw, 1],
                          padding='SAME',
                          name=name)



def concatenate_op(input_op1,input_op2,name):
    return tf.concat(3,[input_op1,input_op2],name=name)


def upsample_op(input_op,name):
    height=input_op.get_shape()[1].value
    width=input_op.get_shape()[2].value
    return tf.image.resize_nearest_neighbor(input_op, size=[2*height, 2*width],name=name)


def fullyconnected_op(input_op, name, n_out, wd, activation=True):
    im_shape = input_op.get_shape().as_list()
    assert len(im_shape) > 1, "Input Tensor shape must be at least 2-D: batch, ninputs"
    n_inputs = int(np.prod(im_shape[1:])) #units at lower layer
    shape=[n_inputs, n_out]
    with tf.variable_scope(name):
        W=_variable_with_weight_decay("w", shape, wd)
        # print(W.name)
        bias_init_val = tf.constant(0.0, shape=[n_out], dtype=tf.float32)
        biases = tf.get_variable(initializer=bias_init_val, trainable=True, name ='b')
        if len(im_shape) > 2: #we have to flatten it then
            x = tf.reshape(input_op, [-1, n_inputs])
        else:
            x=input_op
        z = tf.matmul(x, W)+biases
        if activation:
            z = tf.nn.relu(z)

    return z



def _variable_with_weight_decay(name, shape, wd):
  """Helper to create an initialized Variable with weight decay.
  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.
  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
  Returns:
    Variable Tensor
  """
  #tf.contrib.layers.xavier_initializer_conv2d()
  [fan_in, fan_out]=get_fans(shape)
  initializer=xavier_init(fan_in, fan_out)
  
  var=tf.get_variable(name, shape=shape,dtype=tf.float32, initializer=initializer)
  weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
  tf.add_to_collection('losses', weight_decay)

  return var


def get_fans(shape):
    receptive_field_size = np.prod(shape[:-2])
    fan_in = shape[-2] * receptive_field_size
    fan_out = shape[-1] * receptive_field_size
        
    return fan_in, fan_out


def xavier_init(n_inputs, n_outputs, uniform=True):

    """Set the parameter initialization using the method described.
    This method is designed to keep the scale of the gradients roughly the same
    in all layers.
    Xavier Glorot and Yoshua Bengio (2010):
           Understanding the difficulty of training deep feedforward neural
           networks. International conference on artificial intelligence and
           statistics.
    Args:
    n_inputs: fan_in
    n_outputs: fan_out
    uniform: If true use a uniform distribution, otherwise use a normal.
    Returns:
    An initializer.
    """
    if uniform:
        # 6 was used in the paper.
        init_range = np.sqrt(6.0 / (n_inputs + n_outputs))
        return tf.random_uniform_initializer(-init_range, init_range)
    else:
        # 3 gives us approximately the same limits as above since this repicks
        # values greater than 2 standard deviations from the mean.
        stddev = np.sqrt(3.0 / (n_inputs + n_outputs))
        return tf.truncated_normal_initializer(stddev=stddev)



def batch_norm_layer(x,train_phase,scope_bn):
    outputs = batch_norm(x, is_training=train_phase, center=False, scale=False, activation_fn=tf.nn.relu, updates_collections=None, scope='batch_norm')
    return outputs