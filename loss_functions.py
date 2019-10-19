# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 16:13:03 2018

@author: cagdas
"""

import tensorflow as tf



def lp_loss(generated, gt, l_num, batch_size_tf):
    """
    Calculates the sum of lp losses between the predicted and ground truth images.
    @param ct_generated: The predicted ct
    @param gt_ct: The ground truth ct
    @param l_num: 1 or 2 for l1 and l2 loss, respectively).
    @return: The lp loss.
    """
    lp_loss=tf.reduce_sum(tf.abs(generated - gt)**l_num)/(2*tf.cast(batch_size_tf,tf.float32))
    tf.add_to_collection('losses', lp_loss)

    loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
    return loss



def asl_loss(generated, sig_real, M0, l_num=2, params, batch_size_tf):
""" Calculate the ASL loss based on l1 or l2 loss between the real and 
    estimated (based on signal model) signals.
"""

    y_est = generated
    
    alpha = params['alpha']
    lambda_blood = params['lambda_blood']
    T1blood = params['T1blood']
    T1tissue = params['T1tissue']
    PLD = params['PLDs']
    tao = params['tao']
    scalar_constant = params['scalar']
    
    sig_est = asl_signal_model(y_est, M0, alpha, lambda_blood, T1blood, T1tissue, PLD, tao, scalar_constant)
    
    asl_loss = tf.reduce_sum(tf.abs(sig_est - sig_real)**l_num)/(2*tf.cast(batch_size_tf,tf.float32))    
    
    return asl_loss
    


def asl_signal_model(y_est, M0, alpha, lambda_blood, T1blood, T1tissue, pld, tao, scalar_constant):
""" Calculate the perfusion signal given the ASL parameters/variables using 
    Buxton model equation for multi-PLD data.

    VARIABLES:
    M0 = tf.convert_to_tensor(param.M0, np.float32)   # M0 proton density image
    pld = tf.constant(param.pld)  # Post-labeling delay
    tao = tf.constant(param.tao)    # Labelling duration
    alpha = tf.constant(0.85)      # Tagging efficient for pCASL
    T1_blood = tf.constant(1650)   # Longitudinal relaxation rate of blood
    lambda_blood = tf.constant(0.9)  # Blood-brain partition coefficient
"""
    
    nt = tf.size(pld)
    
    Ndim = tf.shape(y_est[:,:,:,0])
    
    Nb = Ndim[0];  Nx = Ndim[1]; Ny = Ndim[2]
    
    CBF = expand_vol_1d(y_est[:,:,:,0], [1, 1, 1, nt]) # Cerebral blood flow parameter
    ATT = expand_vol_1d(y_est[:,:,:,1], [1, 1, 1, nt]) # Arterial transit time paremter
    
    PLD = expand_vol_4d(pld, [Nb, Nx, Ny, 1])
    TAO = expand_vol_4d(tao, [Nb, Nx, Ny, 1])
    
    # Estimate the 4D perfusion signal based on the ASL parameters according to the Buxton model
    sig_est = (2*alpha*M0*CBF*T1tissue*tf.exp(-(ATT/T1blood))) * (tf.exp(-tf.maximum(PLD-ATT, 0)/T1tissue) - \
               tf.exp(-tf.maximum(TAO+PLD-ATT, 0)/T1tissue)) / (scalar_constant*lambda_blood)

    return sig_est


def cross_entropy_Discriminator(logits_D,gt_D):
""" Calculate binary cross entropy loss for discriminator

    logits_D is the output of the discriminator [batch_size,1]
    gt_D should be all ones for real data, and all zeros for fake-
    generated (output of generator) data[batch_size,1]
"""

    bce=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits_D, gt_D))
    return bce



def expand_vol_1d(input,dim):
""" Expand 3D data (spatial volumes) to 4D (time dimension added)
"""

    expanded = tf.tile(tf.expand_dims(input, 3), dim)
    return expanded



def expand_vol_4d(input,dim):
""" Expand 1D data (time series) to 4D (spatial dimensions added)
"""

    expanded = tf.tile(tf.expand_dims(tf.expand_dims(tf.expand_dims(input, 0), 0), 0), dim)
    return expanded


