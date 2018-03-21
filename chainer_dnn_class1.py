#coding: utf-8

#######################################################
#Description:
# load julius dictation-kit-v4.4 dnn model data
# and calculate dnn.
# This is based on dnnclient.py in dictation-kit-v4.4.zip of
#
# Copyright (c) 1991-2016 Kawahara Lab., Kyoto University
# Copyright (c) 2005-2016 Lee Lab., Nagoya Institute of Technology
#
# License: see LICENSE-Julius Dictation Kit.txt
#
#Date 2018-03-19
#By Shun
#########################################################

import os
import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F



# Check version
#  Python 2.7.12 on win32 (Windows version)
#  numpy (1.14.0)


class Class_DNN(chainer.Chain):
	def __init__(self, IN_DIR0='model/dnn'):
		super(Class_DNN,self).__init__()
		
		
		
		self.W, self.b = self.load_init_DNN_Wb(IN_DIR=IN_DIR0)
		self.num_states= self.W[7].shape[1]  # output dimension
		self.state_prior=self.get_prior( IN_DIR=IN_DIR0)
		
		# F.linear is fixed parameter function, not learnable
		self.l1=F.linear(x, self.W[0] * -1.0, self.b[0])  # w[0].T hituyoukamo???
		self.l2=F.linear(x, self.W[1] * -1.0, self.b[1])
		self.l3=F.linear(x, self.W[2] * -1.0, self.b[2])
		self.l4=F.linear(x, self.W[3] * -1.0, self.b[3])
		self.l5=F.linear(x, self.W[4] * -1.0, self.b[4])
		self.l6=F.linear(x, self.W[5] * -1.0, self.b[5])
		self.l7=F.linear(x, self.W[6] * -1.0, self.b[6])
		self.l8=F.linear(x, self.W[7] , self.b[7])

	def __call__(self,x0):
		# input, x0, should be 2D matrix,  if just a vector, then reshape(xxx,1)
		# output, tmp, shoud be 2D matrix
		if np.ndim(x0) == 1:
			x0=x0.reshape( x0.shape[0],1)
		
		x1 = F.sigmoid( self.l1(x0))
		x2 = F.sigmoid( self.l2(x1))
		x3 = F.sigmoid( self.l3(x2))
		x4 = F.sigmoid( self.l4(x3))
		x5 = F.sigmoid( self.l5(x4))
		x6 = F.sigmoid( self.l6(x5))
		x7 = F.sigmoid( self.l7(x6))
		tmp = F.exp(self.l8(x7))
		tmp /= F.sum(tmp, axis=0)
		tmp /= self.state_prior
		tmp= F.log10(tmp)
		return tmp


	@property
	def fc1_W(self):
		return self.W[0]
	@property
	def fc1_b(self):
		return self.b[0]
	@property
	def fc2_W(self):
		return self.W[1]
	@property
	def fc2_b(self):
		return self.b[1]
	@property
	def fc3_W(self):
		return self.W[2]
	@property
	def fc3_b(self):
		return self.b[2]
	@property
	def fc4_W(self):
		return self.W[3]
	@property
	def fc4_b(self):
		return self.b[3]
	@property
	def fc5_W(self):
		return self.W[4]
	@property
	def fc5_b(self):
		return self.b[4]
	@property
	def fc6_W(self):
		return self.W[5]
	@property
	def fc6_b(self):
		return self.b[5]
	@property
	def fc7_W(self):
		return self.W[6]
	@property
	def fc7_b(self):
		return self.b[6]
	@property
	def fc8_W(self):
		return self.W[7]
	@property
	def fc8_b(self):
		return self.b[7]


	def load_init_DNN_Wb(self,IN_DIR='model/dnn'):
		w_filename = ["W_l1_f4.npy", "W_l2_f4.npy", "W_l3_f4.npy", "W_l4_f4.npy", "W_l5_f4.npy", "W_l6_f4.npy", "W_l7_f4.npy", "W_output_f4.npy"]
		b_filename = ["bias_l1_f4.npy", "bias_l2_f4.npy", "bias_l3_f4.npy", "bias_l4_f4.npy", "bias_l5_f4.npy", "bias_l6_f4.npy", "bias_l7_f4.npy", "bias_output_f4.npy"]
		listw=['fc1_W','fc2_W','fc3_W','fc4_W','fc5_W','fc6_W','fc7_W','fc8_W']
		listb=['fc1_b','fc2_b','fc3_b','fc4_b','fc5_b','fc6_b','fc7_b','fc8_b']
		for i, listx in enumerate(w_filename):
			f=os.path.join(IN_DIR,listx)
			if os.path.exists(f):
				listw[i] = np.load(f)
				print (' load of ', f)
			else:
				listw[i] = None
				print (' no file of ', f)
		for i, listx in enumerate(b_filename):
			f=os.path.join(IN_DIR,listx)
			if os.path.exists(f):
				listb[i] = np.load(f)
				print (' load of ', f)
			else:
				listb[i] = None
				print (' no file of ', f)
		return listw,listb


	def get_prior(self,IN_DIR='model/dnn'):
		prior_filename = 'prior'
		f=os.path.join(IN_DIR,prior_filename)
		assert os.path.exists(f), 'error, no file of ' + f
		print (' open of ', f)
		state_prior = np.zeros(self.num_states)
		prior_factor = 1.0
		for line in open(f):
			state_id, state_p = line[:-1].split(' ')
			state_id = int(state_id)
			state_p = float(state_p) * prior_factor
			state_prior[state_id] = state_p
		
		return state_prior.reshape(self.num_states ,1)   # reshape for 2D x 1D


