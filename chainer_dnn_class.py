#coding: utf-8

#######################################################
#Description:
# load julius dictation-kit-v4.4 dnn model data
# and calculate dnn by deep learning framework chainer
# This is based on dnnclient.py in dictation-kit-v4.4.zip of
#
# Copyright (c) 1991-2016 Kawahara Lab., Kyoto University
# Copyright (c) 2005-2016 Lee Lab., Nagoya Institute of Technology
#
# License: see LICENSE-Julius Dictation Kit.txt
#
#Date 2018-03-20
#By Shun
#########################################################

import os
import numpy as np
import chainer
from chainer import cuda, Variable
from chainer import Link, Chain, ChainList
import chainer.links as L
import chainer.functions as F



# Check version
#  Python 2.7.12 on win32 (Windows version)
#  numpy (1.14.0)
#  chainer (1.22.0)


class Class_chainer_DNN(Chain):
	def __init__(self, in_size, mid_size, out_size, net=None, state_prior=None):
		super(Class_chainer_DNN, self).__init__(
		
		l1 = L.Linear(in_size,mid_size,initialW=net.fc1_W if net else None ,initial_bias=net.fc1_b if net else None),
		l2 = L.Linear(mid_size,mid_size,initialW=net.fc2_W if net else None ,initial_bias=net.fc2_b if net else None),
		l3 = L.Linear(mid_size,mid_size,initialW=net.fc3_W if net else None ,initial_bias=net.fc3_b if net else None),
		l4 = L.Linear(mid_size,mid_size,initialW=net.fc4_W if net else None ,initial_bias=net.fc4_b if net else None),
		l5 = L.Linear(mid_size,mid_size,initialW=net.fc5_W if net else None ,initial_bias=net.fc5_b if net else None),
		l6 = L.Linear(mid_size,mid_size,initialW=net.fc6_W if net else None ,initial_bias=net.fc6_b if net else None),
		l7 = L.Linear(mid_size,mid_size,initialW=net.fc7_W if net else None ,initial_bias=net.fc7_b if net else None),
		l8 = L.Linear(mid_size,out_size,initialW=net.fc8_W if net else None ,initial_bias=net.fc8_b if net else None),
		)
		self.state_prior=state_prior
		
		
	def __call__(self,x):
		# dnn
		x1 = F.sigmoid( self.l1(x))
		x2 = F.sigmoid( self.l2(x1))
		x3 = F.sigmoid( self.l3(x2))
		x4 = F.sigmoid( self.l4(x3))
		x5 = F.sigmoid( self.l5(x4))
		x6 = F.sigmoid( self.l6(x5))
		x7 = F.sigmoid( self.l7(x6))
		# discriminator
		tmp = F.softmax(self.l8(x7))
		
		# apply state prior probability and log10
		if self.state_prior is not None:
			tmp /= self.state_prior
			tmp= F.log10(tmp)
		
		return tmp


class Class_net(object):
	def __init__(self, IN_DIR='model/dnn'):
		self.IN_DIR= IN_DIR
		self.W, self.b = self.load_init_DNN_Wb()
		self.num_states= self.W[7].shape[1]  # output dimension
		self.num_input= self.W[0].shape[0]   # input dimension
		self.num_mid= self.W[0].shape[1]     # middle dimension
		self.prior = self.get_prior()
	@property
	def fc1_W(self):
		return self.W[0].T
	@property
	def fc1_b(self):
		return self.b[0].reshape(self.b[0].size)
	@property
	def fc2_W(self):
		return self.W[1].T
	@property
	def fc2_b(self):
		return self.b[1].reshape(self.b[1].size)
	@property
	def fc3_W(self):
		return self.W[2] .T
	@property
	def fc3_b(self):
		return self.b[2].reshape(self.b[2].size)
	@property
	def fc4_W(self):
		return self.W[3].T
	@property
	def fc4_b(self):
		return self.b[3].reshape(self.b[3].size)
	@property
	def fc5_W(self):
		return self.W[4].T
	@property
	def fc5_b(self):
		return self.b[4].reshape(self.b[4].size)
	@property
	def fc6_W(self):
		return self.W[5].T
	@property
	def fc6_b(self):
		return self.b[5].reshape(self.b[5].size)
	@property
	def fc7_W(self):
		return self.W[6].T
	@property
	def fc7_b(self):
		return self.b[6].reshape(self.b[6].size)
	@property
	def fc8_W(self):
		return self.W[7].T
	@property
	def fc8_b(self):
		return self.b[7].reshape(self.b[7].size)


	def load_init_DNN_Wb(self,):
		w_filename = ["W_l1_f4.npy", "W_l2_f4.npy", "W_l3_f4.npy", "W_l4_f4.npy", "W_l5_f4.npy", "W_l6_f4.npy", "W_l7_f4.npy", "W_output_f4.npy"]
		b_filename = ["bias_l1_f4.npy", "bias_l2_f4.npy", "bias_l3_f4.npy", "bias_l4_f4.npy", "bias_l5_f4.npy", "bias_l6_f4.npy", "bias_l7_f4.npy", "bias_output_f4.npy"]
		listw=['fc1_W','fc2_W','fc3_W','fc4_W','fc5_W','fc6_W','fc7_W','fc8_W']
		listb=['fc1_b','fc2_b','fc3_b','fc4_b','fc5_b','fc6_b','fc7_b','fc8_b']
		for i, listx in enumerate(w_filename):
			f=os.path.join(self.IN_DIR,listx)
			if os.path.exists(f):
				listw[i] = np.load(f)
				print (' load of ', f)
			else:
				listw[i] = None
				print (' no file of ', f)
		for i, listx in enumerate(b_filename):
			f=os.path.join(self.IN_DIR,listx)
			if os.path.exists(f):
				listb[i] = np.load(f)
				print (' load of ', f)
			else:
				listb[i] = None
				print (' no file of ', f)
		return listw,listb


	def get_prior(self,):
		prior_filename = 'prior'
		f=os.path.join(self.IN_DIR,prior_filename)
		assert os.path.exists(f), 'error, no file of ' + f
		print (' open of ', f)
		state_prior = np.zeros(self.num_states)
		prior_factor = 1.0
		for line in open(f):
			state_id, state_p = line[:-1].split(' ')
			state_id = int(state_id)
			state_p = float(state_p) * prior_factor
			state_prior[state_id] = state_p
		
		return state_prior


