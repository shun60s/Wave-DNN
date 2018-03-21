#coding: utf-8

#######################################################
#Description:
# This use julius dictation-kit-v4.4 dnn model data.
# Please see LICENSE-Julius Dictation Kit.txt
#
#Date 2018-03-19
#By Shun
#########################################################

import os
import numpy as np

from cmvn_class import *
from get_fbank import *
from dnn_class import *


# Check version
#  Python 2.7.12 on win32 (Windows version)
#  numpy (1.14.0)




if __name__ == '__main__':

	# set mel-filter bank(log)
	fbank0= GetFBANK_D_A()
	
	# load 16KHz sampling Mono Wave data and convert to mel-filter bank(log)
	# Specify some wav file
	fbankda=fbank0.get_fbank_d_a('trial_1.wav', fshow=True)
	
	# load mean and variance
	cmn0=Class_CMVN()
	
	# apply mean and variance normalization. Select either MVN or MAPCMN
	data0cmv=cmn0.MVN(fbankda)        # MVN   -cmnstatic
	
	#data0cvm=cmn0.MAPCMN(fbankda)    # MAPCMN  -cmnnoupdate
	#cmn0.reset_nowframenum()  # reset for next call	
	
	# dnn ready
	dnn0=Class_DNN()
	
	num_raw= data0cmv.shape[1]  # 120, number of one frame
	num_frames= data0cmv.shape[0] # number of input wave frames
	num_context = dnn0.num_input / num_raw # 11 : num_input = num_raw * num_context
	batch_size= num_frames - (num_context -1)  # 
	
	print (' num_raw', num_raw)
	print (' num_context', num_context)
	print (' num_frames', num_frames)
	
	# make dnn input data from  mel-filter bank(log) with mean and variance normalization
	d0=np.zeros( ( dnn0.num_input, batch_size) )
	din=data0cmv.reshape( data0cmv.size)  #  convert from 2D to 1D serial data

	nsp0=0   # set initial start frame position
	for i in range (batch_size):
 		d0[:,i]=din[ nsp0  :  nsp0  + dnn0.num_input ]
 		nsp0 += num_raw  # for next 
	
	# calculate dnn
	dnn_out1= dnn0(d0)
	print (' dnn out size', dnn_out1.shape)
	
# This file uses TAB.






