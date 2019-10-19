# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 14:01:11 2018

@author: cagdas
"""

from __future__ import division
import time as tt
import tensorflow as tf
import numpy as np

import utils as ut
from loss_functions import lp_loss, asl_loss
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # Remove this when running on a machine without any GPU.
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt




class ASLGAN(object):
    def __init__(self, sess, batch_size = 10, Xdata = None, Ylabel = None, Sig = None, Yund = None, M0 = None, Params = None):
        """ Initialize the object
        """

        self.sess = sess
        # Input shape
        self.height_MR = 70
        self.width_MR = 70
        self.channels = 10  # Number of PLDs or time-points
        self.output_dim = 2  # number of ASL parameters to estimate (CBF, ATT)
        
        # Data
        self.Xdata = Xdata  # 
        self.Ylabel = Ylabel
        self.Yund = Yund
        
        # Number of residual blocks in the generator
        self.n_residual_blocks = 8
        
        # Depth of dual paths in the generator
        self.depth_dual = 4
    
        # Weight decay
        self.wd = 0.0005 
        self.df = 64
        
        # Batch normalization flag
        self.bn_g = False # For generator
        self.bn_d = True  # For discriminator
        
        # Lambda value for loss terms
        self.lam_lp = 1.0
        self.lam_asl = 3.0 # 3.0 is the optimal
        self.lam_adv = 0.5
        
        print ("[ LP loss coef: %f, ASL loss coef: %f, Adversarial loss coef: %f ]" % (self.lam_lp, self.lam_asl, self.lam_adv))
        
        # Learning rate
        self.learning_rate = 1e-3
        self.batch_size = batch_size
        
        # Norm of reconstruction term
        self.l_num = 1
        
        # Get the generic ASL parameters
        self.common_vars = ut.load_variables(Params)
        
        # Call the batch generation for every iteration of GAN
        self.data_generator = ut.generate_batches(Xdata, Ylabel, Yund, Sig, M0, self.batch_size)
        
        # Build GAN model
        self.build_model()
        
        
    def build_model(self):
       """ Build GAN model based on the given variables, GAN networks and loss functions
       """
        
        # Create TF placeholder for every variable used in GAN
        self.inputMR=tf.placeholder(tf.float32, shape=[None, self.height_MR, self.width_MR, self.channels])
        self.param_GT=tf.placeholder(tf.float32, shape=[None, self.height_MR, self.width_MR, self.output_dim])
        self.sig_real = tf.placeholder(tf.float32, shape=[None, self.height_MR, self.width_MR, self.channels])
        self.M0 = tf.placeholder(tf.float32, shape=[None, self.height_MR, self.width_MR, self.channels])
        self.train_phase = tf.placeholder(tf.bool, name='phase_train')
        
        batch_size_tf = tf.shape(self.inputMR)[0]  

        # Invoke generator network
        self.G, self.layer = self.generator(self.inputMR)     
        # Invoke discriminator network with Ground truth ASL parameters
        self.D, self.D_logits = self.discriminator_conv_blocks(self.param_GT)   # GT parameters
        # Invoke discriminator network with fake generated parameters
        self.D_, self.D_logits_ = self.discriminator_conv_blocks(self.G, reuse=True) # Fake generated parameter estimate
        
        # Calculate cross entropy loss for true samples
        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits, labels=tf.ones_like(self.D)))
        # Calculae cross entropy loss for fake samples
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.zeros_like(self.D_)))
        # Overal discriminator loss
        self.d_loss=self.d_loss_real + self.d_loss_fake
        
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        
        # Combined generator loss
        self.g_loss, self.lpterm, self.aslterm, self.bceterm=self.combined_loss_G(self.batch_size)
        
        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        # Define the optimizer type for both D and G networks.
        with tf.variable_scope(tf.get_variable_scope(),reuse=False)
            self.d_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5) \
                                  .minimize(self.d_loss, var_list=self.d_vars)
            self.g_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5) \
                                  .minimize(self.g_loss, var_list=self.g_vars, global_step=self.global_step)
                                  
        print('Shape output G = %s' %self.G.get_shape())
        print ('Shape output D = %s' %self.D.get_shape())
        print ('Learning rate = %s' %self.learning_rate)
        
        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter("./summaries", self.sess.graph)
        self.saver = tf.train.Saver()
        
        
    def generator(self, inputMR, reuse = False):
      """ Generator network architecture which is fully convolutional (involving all convolutional layers) 
          with PRELU activation. - Used in our MICCAI ASL paper
      """

        with tf.variable_scope('generator_unit', reuse=reuse):
            if (reuse):
                tf.get_variable_scope().reuse_variables()
                   
            ## Fully convolutional network ## 
            print('G input shape = %s' %inputMR.get_shape()) 
            
            print('Generator batch normalization is %s' %self.bn_g)
                        
            d = tf.layers.conv2d(inputMR, filters = 64, kernel_size=3, dilation_rate = 1,  padding='same', name='g_conv_1')
            d = ut.prelu_tf(d, name='prelu_1')
            d = tf.layers.batch_normalization(inputs=d, name='g_conv_1_bn') if self.bn_g else d
                    
        
         
            d = tf.layers.conv2d(d, filters = 64, kernel_size=3, dilation_rate = 1,  padding='same', name='g_conv_2')
            d = ut.prelu_tf(d, name='prelu_2')
            d = tf.layers.batch_normalization(inputs=d, name='g_conv_2_bn') if self.bn_g else d
            
            d = tf.layers.conv2d(d, filters = 64, kernel_size=3, dilation_rate = 1,  padding='same', name='g_conv_3')
            d = ut.prelu_tf(d, name='prelu_3')
            d = tf.layers.batch_normalization(inputs=d, name='g_conv_3_bn') if self.bn_g else d
        
        
            d = tf.layers.conv2d(d, filters = 64, kernel_size=3, dilation_rate = 1,  padding='same', name='g_conv_4')
            d = ut.prelu_tf(d, name='prelu_4')
            d = tf.layers.batch_normalization(inputs=d, name='g_conv_4_bn') if self.bn_g else d
     
            d = tf.layers.conv2d(d, filters = 64, kernel_size=3, dilation_rate = 1,  padding='same', name='g_conv_5')
            d = ut.prelu_tf(d, name='prelu_5')
            d = tf.layers.batch_normalization(inputs=d, name='g_conv_5_bn') if self.bn_g else d
            
            d = tf.layers.conv2d(d, filters = 64, kernel_size=3, dilation_rate = 1,  padding='same', name='g_conv_6')
            d = ut.prelu_tf(d, name='prelu_6')
            d = tf.layers.batch_normalization(inputs=d, name='g_conv_6_bn') if self.bn_g else d
        
            d = tf.layers.conv2d(d, filters = 64, kernel_size=3, dilation_rate = 1,  padding='same', name='g_conv_7')
            d = ut.prelu_tf(d, name='prelu_7')
            d = tf.layers.batch_normalization(inputs=d, name='g_conv_7_bn') if self.bn_g else d
        
            d = tf.layers.conv2d(d, filters = 64, kernel_size=3, dilation_rate = 1,  padding='same', name='g_conv_8')
            d = ut.prelu_tf(d, name='prelu_8')
            d = tf.layers.batch_normalization(inputs=d, name='g_conv_8_bn') if self.bn_g else d
        
            out = tf.layers.conv2d(d, filters = self.output_dim, kernel_size=3, dilation_rate = 1,  padding='same', name='g_conv_9')
            
            print('G output shape = %s' %out.get_shape())

            return out, out
        
        
    def generator_residual_blocks(self, inputMR, reuse = False):
       """ Generator network architecture used in SRGAN (C.Ledig et al CVPR 2017) based on residual block and skip connection.
       """
        with tf.variable_scope('generator_unit', reuse=reuse):
            if (reuse):
                tf.get_variable_scope().reuse_variables()
                
            print('G input shape = %s' %inputMR.get_shape())
            print('Generator batch normalization is %s' %self.bn_g)

            def residual_block(layer_input, fsize, strd = 1, bn = self.bn_g, res_block=1):
                nm = 'g_res_block_' + str(res_block)
                with tf.variable_scope(nm):
                    net = tf.layers.conv2d(layer_input, filters=fsize, kernel_size=3, strides=strd, padding='same', activation = tf.nn.relu, name='conv1')
                    net = tf.layers.batch_normalization(inputs=net, name ='bn1') if bn else net
                
                    net = tf.layers.conv2d(net, filters=fsize, kernel_size=3, strides=strd, padding='same', name='conv2')
                    net = tf.layers.batch_normalization(inputs=net, name = 'bn2') if bn else net
                    net = net + layer_input
                return net
            
                 # The input layer
            with tf.variable_scope('input_stage'):
                net = tf.layers.conv2d(inputMR, filters=64, kernel_size=5, strides=1, padding='same',  activation = tf.nn.relu, name='conv')
                # net = ut.prelu_tf(net)
                
            stage1_output = net
                
                #The residual block parts
            for i in range(1, self.n_residual_blocks+1, 1):
                net = residual_block(net, fsize=64, strd=1, res_block=i)
                
            with tf.variable_scope('resblock_output'):
                net = tf.layers.conv2d(net, filters= 64, kernel_size=3, strides=1, padding='same', name='conv')
                net = tf.layers.batch_normalization(inputs=net, name='bn') if self.bn_g else net
                    
            net = net + stage1_output
                
            with tf.variable_scope('post_conv_block1'):
                net = tf.layers.conv2d(net, filters=128, kernel_size=3, strides=1, padding='same', activation = tf.nn.relu, name='conv')
                # net = ut.prelu_tf(net)

            with tf.variable_scope('post_conv_block2'):
                net = tf.layers.conv2d(net, filters=128, kernel_size=3, strides=1, padding='same', activation = tf.nn.relu, name='conv')
                # net = ut.prelu_tf(net)

            with tf.variable_scope('output_stage'):
                out =  tf.layers.conv2d(net, filters=self.output_dim, kernel_size=5, strides=1, padding='same', name='conv')
            
            print('G output shape = %s' %out.get_shape())
             
            return out, out


    def discriminator(self, inputPR, reuse=False)
     """ A typical classification (discriminator) network architecture with convolutional layers, batch normalization,
         2D max pooling and dense layer at the end.
     """
        with tf.variable_scope('discriminator_unit', reuse=reuse):
            if reuse:
                tf.get_variable_scope().reuse_variables()
                
            print('D input shape = %s' %inputPR.get_shape()) 
            print('Discriminator batch normalization is %s' %self.bn_d)

        
            def cnn_block(layer_input, fsize, strd=1, bn=self.bn_d, conv_no=1):
                """Typical cnn layer"""
                nm = 'd_conv_' + str(conv_no)
                d = tf.layers.conv2d(layer_input, filters=fsize, kernel_size=3, strides=strd, padding='same', activation= tf.nn.relu, name = nm)
                d = tf.layers.batch_normalization(inputs=d, name = nm + '_bn') if bn else d
                d = tf.layers.max_pooling2d(inputs=d, pool_size=2, strides = 2,  name = nm + '_mp')
                return d
            
           
        
            d1 = cnn_block(inputPR, int(self.df/2), conv_no=1)
            d2 = cnn_block(d1, self.df, conv_no=2)
            d3 = cnn_block(d2, self.df*2, conv_no=3)
            d4 = tf.layers.conv2d(d3, self.df*4, kernel_size=3, padding='same', name = 'd_conv_4')
        
            d5 = tf.layers.dense(inputs=d4, units=self.df*8, activation=tf.nn.relu, name = 'd_fc1')
            d6 = tf.layers.dense(inputs=d5, units=self.df*2, activation=tf.nn.relu, name = 'd_fc2')

            out = tf.layers.dense(inputs=d6, units=self.output_dim, name = 'd_fc3')
                        
            print('D output shape = %s' %out.get_shape())

            return tf.nn.sigmoid(out), out
        
    
    def discriminator_conv_blocks(self, inputPR, reuse=False):
       """ Classification (Discriminator) network architecture with the combination of convolutional layers and
       fully convolutinal (dense) layers. Sigmoid activation function is used at the end.
       """

        with tf.variable_scope('discriminator_unit', reuse=reuse):
            if (reuse):
                tf.get_variable_scope().reuse_variables()
                
            print('D input shape = %s' %inputPR.get_shape())
            print('Discriminator batch normalization is %s' %self.bn_d)
            
            def discriminator_block(layer_input, fsize, ksize, strd, scope, bn = self.bn_d):
                net = tf.layers.conv2d(layer_input, filters=fsize, kernel_size=ksize, strides=strd, padding='same', activation = tf.nn.leaky_relu, name= scope + '_conv')
                net = tf.layers.batch_normalization(inputs=net, name = scope + '_bn') if bn else net
                return net
            
           
            net = tf.layers.conv2d(inputPR, filters=64, kernel_size=3, strides=1, padding='same', activation = tf.nn.leaky_relu, name='d_conv_1')
                
            # The discriminator block part
            # block 1
            net = discriminator_block(net, 64, 3, 2, scope = 'disblock_1')

            # block 2
            net = discriminator_block(net, 128, 3, 1, scope = 'disblock_2')

            # block 3
            net = discriminator_block(net, 128, 3, 2, scope = 'disblock_3')

            # block 4
            net = discriminator_block(net, 256, 3, 1, scope = 'disblock_4')

            # block 5
            net = discriminator_block(net, 256, 3, 2, scope = 'disblock_5')

            # block 6
            net = discriminator_block(net, 512, 3, 1, scope = 'disblock_6')

            # block_7
            net = discriminator_block(net, 512, 3, 2, scope = 'disblock_7')
            
             
            net = tf.layers.dense(inputs=net, units=1024, activation = tf.nn.leaky_relu, name='d_fc1')
            
            
            out = tf.layers.dense(inputs=net, units=self.output_dim, name='d_fc2')
                
            print('D output shape = %s' %out.get_shape())

            return tf.nn.sigmoid(out), out
    
    
    
    
    def train(self, epochs = 2000, save_interval = 2):
        """ Train the GAN for N epochs and validate it on sampled test images.
        """

        for v in tf.trainable_variables():
            print(v.name)
        
        self.sess.run(tf.global_variables_initializer())

        self.sess.graph.finalize()
        
        start = self.global_step.eval(session = self.sess) # get last global_step
        print("Start from:", start)
        
        # For every epoch train D and G networks alternatively
        for epoch in range(epochs):
            X, y, yund, sig, m0 = next(self.data_generator)
            
            
            # Update D network
            _, loss_eval_D, = self.sess.run([self.d_optim, self.d_loss],
                        feed_dict={ self.inputMR: X, self.param_GT:y, self.train_phase: True })

            # Update G network
            _, loss_eval_G, lp_eval, asl_eval, bce_eval, layer_out_eval = self.sess.run([self.g_optim, 
                                    self.g_loss, self.lpterm, self.aslterm, self.bceterm, self.layer],
                                    feed_dict={self.inputMR: X, self.param_GT:y, 
                                                self.sig_real: sig, self.M0: m0, self.train_phase: True})

           
            # Plot the progress
            print ("Epoch %d => [D loss: %f] [Total G loss: %f, LP loss: %f, ASL loss: %f, BCE loss: %f]" % (epoch, loss_eval_D, loss_eval_G, lp_eval, asl_eval, bce_eval))
            
            # For every Nth epoch sample test images and produce/save the associated parameter maps predicted by the generator
            if epoch % save_interval == 0:
                self.test_MR_source, self.test_ylabel, self.test_yund = ut.pick_random_test_samples(self.Xdata, self.Ylabel, self.Yund, sample_size = 2)
                self.test_sample_images(epoch)
    
    
    
    def evaluate(self, inputASL):
        """ Return the parameters maps for an input multi-PLD ASL data and 
        the trained generator network. (TESTING)
        """

        param_pred, MR16_eval= self.sess.run([self.G, self.layer],
                        feed_dict={ self.inputMR: inputASL, self.train_phase: False})

      
        return param_pred
    
    
    
            
    def test_sample_images(self, epoch):
        """ Predict parameter maps from the input test samples and save them in a figure.
        """
        r = np.shape(self.test_MR_source)[0]
        c = 3

        # Estimate the parameter maps from input ASL data
        gen_param = self.evaluate(self.test_MR_source)
        
       # Get either CBF or ATT maps - 0:CBF, 1:ATT
        est_param = gen_param[:,:,:,1]
        gt_param =  self.test_yund[:,:,:,1] + self.test_ylabel[:,:,:,1]
        y_und = self.test_yund[:,:,:,1]

        # Save generated images and the high resolution originals
        titles = ['Initial', 'Generated', 'Original']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for row in range(r):
             for col, image in enumerate([y_und, est_param, gt_param]):
                 axs[row, col].imshow(image[row,:,:],cmap='gray')
                 axs[row, col].set_title(titles[col])
                 axs[row, col].axis('off')
             cnt += 1
        fig.savefig("att_images/att_%d.png" % epoch)
        plt.close()
    
    
    

    def save(self, checkpoint_dir, step):
    """ Save the trained model 
     """
	 
        model_name = "ASL_GAN_Synthetic.model"
        
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
    """ Load the training model
    """
	
        print(" [*] Reading checkpoints...")

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            return True
        else:
            return False   
        
        
    def combined_loss_G(self, batch_size_tf):
        """
        Calculates the sum of the combined adversarial, LP and ASL losses in the given proportion. Used
        for training the generative model.
        @param gen_frames: A list of tensors of the generated frames at each scale.
        @param gt_frames: A list of tensors of the ground truth frames at each scale.
        @param d_preds: A list of tensors of the classifications made by the discriminator model at each
                        scale.
        @param lam_adv: The percentage of the adversarial loss to use in the combined loss.
        @param lam_lp: The percentage of the lp loss to use in the combined loss.
        @param lam_asl: The percentage of the ASL loss to use in the combined loss.
        @param l_num: 1 or 2 for l1 and l2 loss, respectively).
        @param alpha: The power to which each gradient term is raised in GDL loss.
        @return: The combined adversarial, lp and GDL losses.
        """


        lpterm=lp_loss(self.G, self.param_GT, self.l_num, batch_size_tf)
        aslterm=asl_loss(self.G, self.sig_real, self.M0, self.l_num, self.common_vars, batch_size_tf)
        bceterm=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.ones_like(self.D_)))
        loss_=self.lam_lp*lpterm + self.lam_asl*aslterm + self.lam_adv*bceterm
        
        tf.add_to_collection('losses', loss_)
        loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

        return loss, lpterm, aslterm, bceterm        



if __name__ == '__main__':
""" Main script for running synthetic dataset with GAN """

    # Load the synthetic dataset
    gt_pwi = ut.load_big_data('synthetic_net_sub_data_gt_pwi.mat', 'gt_pwi')
    xtrain = ut.load_big_data('synthetic_net_sub_data_in_pwi.mat', 'in_pwi')
    print(np.shape(xtrain))

    # Load the parameter maps and M0 data 
    est_maps, ylabel, m0 = ut.load_several_keys('synthetic_net_sub_data_maps.mat', 'est_maps', 'ylabel', 'M0')
    # Load the common ASL parameters for synthetic data
    params = ut.load_data('synthetic_ni_common_vars.mat')
    
    nt = np.shape(gt_pwi)[-1]
    
    # Expand 3D M0 data along time to make it 4D
    m0 = ut.expand_vol_1d(m0, [1, 1, 1, nt])

    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        # Initialize the ASLGAN object
        aslg = ASLGAN(sess, batch_size = 100, Xdata = xtrain, Ylabel = ylabel, Sig = gt_pwi, 
                                         Yund = est_maps, M0 = m0, Params = params)

        print('GAN training is starting..')
        start_time = tt.time()
        # Start training
        aslg.train(epochs=30000, save_interval=100)
        end_time = tt.time()
        
        # Display the elapsed time
        elapsed_time = end_time-start_time    
        print('GAN fitting finished..')
        print('Elapsed time is %.2f seconds..' %elapsed_time)


