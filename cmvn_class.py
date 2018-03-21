#coding: utf-8

#######################################################
#Description:
# load julius dictation-kit-v4.4 Mean/Variance Normalization data (norm)
# and apply Mean/Variance Normalization
# warning: this is not complete compatible with julius.
# 
# This is based on C program in julius-4.4.2.zip of
#
# Copyright (c) 1991-2016 Kawahara Lab., Kyoto University
# Copyright (c) 1997-2000 Information-technology Promotion Agency, Japan
# Copyright (c) 2000-2005 Shikano Lab., Nara Institute of Science and Technology
# Copyright (c) 2005-2016 Julius project team, Nagoya Institute of Technology
#
# License: see LICENSE-Julius.txt
#
#Date 2018-03-19
#By Shun
#########################################################

import os
import numpy as np

# Check version
#  Python 2.7.12 on win32 (Windows version)
#  numpy (1.14.0)

#
# When run-xxx-dnncli.sh use, def ff() in dnnclient.py, received data, 1st frame maybe lost!?
# 


class Class_CMVN(object):
	def __init__(self, num_raw0=120, IN_DIR0='model/dnn', prior_filename0='norm'):
		self.num_raw=num_raw0
		self.mfcc_dim=num_raw0 / 3  # =40 as num_rawa0=120
		self.means, self.variances = self.get_norm( IN_DIR=IN_DIR0, prior_filename=prior_filename0)
		self.means0 = np.copy(self.means)
		self.means0[self.mfcc_dim:]=0.0  # only portion of mfcc_dim mean is available, others is zero
		
		self.nowframenum =0
		self.weight=100.0
		self.nowmfcc_sum = np.zeros(self.num_raw)


	#
	# run-xxx-dnncli.sh use  MAP-CMN_realtime. ex.in=file -paramtype FBANK_D_A_Z -cmnnoupdate
	#
	def MAPCMN_oneframe(self, mfcc_input):   # process one frame only
		
		self.nowframenum +=1
		
		# avoid overwrite original input
		mfcc=np.copy(mfcc_input) 
		
		self.nowmfcc_sum += mfcc
		
		x = (self.nowmfcc_sum + self.weight * self.means)/(self.nowframenum + self.weight)
		
		# mean normalization   (base MFCC only) 
		mfcc[0:self.mfcc_dim] -= x[0:self.mfcc_dim]
		
		# variance normalization (static)
		mfcc /= np.sqrt(self.variances)
		
		return mfcc
		
		
	def MAPCMN(self, fbank_d_a):   # process whole frames
		
		# get number of whole frames
		nframe=fbank_d_a.shape[0]
		
		# copy
		fbank_d_a_cmvn =np.copy( fbank_d_a ) 
		
		for l in range (nframe):
			self.nowframenum +=1
			self.nowmfcc_sum += fbank_d_a[l]
		
			x = (self.nowmfcc_sum + self.weight * self.means)/(self.nowframenum + self.weight)
		
			# mean normalization   (base MFCC only) 
			fbank_d_a_cmvn[l][0:self.mfcc_dim] -= x[0:self.mfcc_dim]
		
			# variance normalization (static)
			fbank_d_a_cmvn[l] /= np.sqrt(self.variances)
		
		#
		print ('nowframenum ', self.nowframenum)
		
		return fbank_d_a_cmvn


	def reset_nowframenum(self,):
		self.nowframenum=0
		self.nowmfcc_sum = np.zeros(self.num_raw)



	#
	# run-xxx-dnn.sh  use  MVN   ex.in=file -paramtype FBANK_D_A_Z  -cmnstatic
	#
	# Cepstrum Mean/Variance Normalization (buffered)
	#
	def MVN_oneframe(self,mfcc_input):  # process one frame only
		
		# avoid overwrite original input
		mfcc=np.copy(mfcc_input) 
		
		# mean normalization (base MFCC only)
		mfcc[0:self.mfcc_dim] -= self.means[0:self.mfcc_dim]
		
		# variance normalization (full MFCC)
		mfcc /= np.sqrt(self.variances)
		
		return mfcc

	def MVN(self,fbank_d_a):  # process whole frames
		
		# copy
		fbank_d_a_cmvn = np.copy(fbank_d_a)
		
		# mean normalization (base MFCC only)
		fbank_d_a_cmvn -= self.means0.reshape(1,self.means0.shape[0])
		
		# variance normalization (full MFCC)
		fbank_d_a_cmvn /= np.sqrt(self.variances).reshape(1,self.variances.shape[0])
		
		return fbank_d_a_cmvn




	def get_norm(self, IN_DIR='model/dnn', prior_filename='norm'):
		f=os.path.join(IN_DIR,prior_filename)
		assert os.path.exists(f), 'error, no file of ' + f
		print (' open of ', f)
		num_means=0
		num_variances =0
		for line in open(f):
			if len(line) == 0:
				continue
			elif line.find('<MEAN>') != -1:
				print ('found <MEAN>')
				word0, num0 = line[:-1].split(' ')
				num_means=int(num0)
				assert num_means == self.num_raw , 'error, num_means and num_raw is mismatch' 
				means= np.zeros(num_means)
			elif line.find('<VARIANCE>') != -1:
				print ('found <VARIANCE>')
				word0, num0 = line[:-1].split(' ')
				num_variances=int(num0)
				assert num_variances == self.num_raw , 'error, num_variances and num_raw is mismatch' 
				variances= np.zeros(num_variances)
			elif num_means != 0 and num_variances == 0:
				values0= line[:-1].split(' ')
				for i in range (num_means):
					means[i]=float(values0[i+1])
			elif num_means != 0 and num_variances != 0:
				values0= line[:-1].split(' ')
				for i in range (num_variances):
					variances[i]=float(values0[i+1])

		return means, variances




