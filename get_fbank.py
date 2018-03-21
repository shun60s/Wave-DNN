#coding: utf-8

#######################################################
#Description:
# This is a python implement to get FBANK_D_A from a wave file as input.
# warning: this is not complete compatible with julius.
#
# This is based on HTKFeat.py in PyHTK and C program in julius-4.4.2.zip.
#
# PyHTK: <https://github.com/danijel3/PyHTK>
# License: Apache License, Version 2.0 (see LICENSE-PyHTK)
#
#
# julius-4.4.2:
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


import numpy as np
import wave

# Check version
#  Python 2.7.12 on win32 (Windows version)
#  numpy 1.14.0 


class GetFBANK_D_A:
	def __init__(self,NFRAME=25, NSHIFT=10, sampling_rate=16000, num_banks=40,deltawindowlen=2):
		
		self.NFRAME=int(NFRAME * 0.001/ (1.0 /sampling_rate))   # xxx ms is What points ?  # 400 sr=16Khz  # 551 sr=22.05Khz
		self.NSHIFT=int(NSHIFT * 0.001/ (1.0 /sampling_rate))   # xxx ms is What points ? # 160 sr=16khz   # 220 sr=22.050khz
		self.fft_len = 2 ** int(np.asscalar((np.floor(np.log2(self.NFRAME)) + 1)))
		self.fft_data_len= self.fft_len / 2 + 1  # np.rfft output is N/2+1
		self.num_banks=num_banks
		self.sampling_rate=sampling_rate
		self.deltawindowlen=deltawindowlen
		print ('NFRAME ', self.NFRAME, ' NSHIFT ', self.NSHIFT, ' fft_len', self.fft_len)

		# Windows is Hamming
		self.window = np.hamming(self.NFRAME)
		# make filter bank weight
		self.melmat, self.loChan, self.loWt, self.klo, self.khi = self.cal_melmat()
		# pre-emphasis 
		self.preemph=0.97



	def cal_delta(self,db0):
	#
	#   db0[num_frame, num_vec]
	#
		B = 2.0 * (sum(np.arange(1, self.deltawindowlen + 1) ** 2))
		num_frame=db0.shape[0]
		num_vec=db0.shape[1]
		deltas = np.zeros( (num_frame, num_vec))
		for n in range(num_frame):
			delta1 = np.zeros(num_vec)
			for t in range(1, self.deltawindowlen + 1):
				tm = n - t
				tp = n + t
				if tm < 0:
					tm = 0
				if tp >= num_frame:
					tp = num_frame - 1

				delta1 += (t * (db0[tp] - db0[tm])) 

			#print (delta1)
			deltas[n,:]= delta1 / B
		
		return deltas


	def get_fbank_d_a(self,file_name,fshow=False ):
    	
		waveFile= wave.open( file_name, 'r')
		
		nchannles= waveFile.getnchannels()
		samplewidth = waveFile.getsampwidth()
		sampling_rate = waveFile.getframerate()
		nframes = waveFile.getnframes()
		
		assert sampling_rate == self.sampling_rate, ' sampling rate is miss match ! '
		
		
		if fshow :
			print("Channel num : ", nchannles)
			print("Sampling rate : ", sampling_rate)
			print("Frame num : ", nframes)
			print("Sample width : ", samplewidth)
		
		buf = waveFile.readframes(-1) # read all, or readframes( 1024)
		
		waveFile.close()
		
		if samplewidth == 2:
			data = np.frombuffer(buf, dtype='int16')
			fdata = data.astype(np.float32) / 32768.
		elif samplewidth == 4:
			data = np.frombuffer(buf, dtype='int32')
			fdata = data.astype(np.float32) / np.power(2.0, 31)
		
		# convert to 16bit integer scale
		fdata *= 32768.
		
		# convert to  MONO, if stereo input
		if nchannles == 2:
			#l_channel = fdata[::nchannles]
			#r_channel = fdata[1::nchannles]
			fdata= (fdata[::nchannles] + fdata[1::nchannles]) /2.0
		
		count= ((nframes - ( self.NFRAME - self.NSHIFT)) / self.NSHIFT)
		time_song = float(nframes) / sampling_rate
		time_unit = 1 / float(sampling_rate)
		
		if fshow :
			print("time song : ", time_song)
			print("time unit : ", time_unit)
			print("count : ", count)
		
		
		# initi spect 
		fbank = np.zeros([count,self.num_banks]) 
		pos = 0
		countr=0
		for fft_index in range(count):
			frame = fdata[pos:pos + self.NFRAME].copy()
			
			## pre-emphasis
			frame -= np.hstack((frame[0], frame[:-1])) * self.preemph
			
			windowed = self.window * frame
			fft_result = np.fft.rfft(windowed, n=self.fft_len)  # real input, output dimension N/2+1, zero padding
			fft_data = np.abs(fft_result) 
			fft_data2= np.dot(self.melmat, fft_data)
			fft_data2[ fft_data2 < 1.0] = 1.0  # as of julius if(temp < 1.0) temp = 1.0;
			fft_data2 = np.log(fft_data2) 
			
			fbank[countr] = fft_data2
			# index count up
			countr +=1
			# next
			pos += self.NSHIFT

		# get delta
		fbank_d= self.cal_delta(fbank)
		# get acceleration
		fbank_a= self.cal_delta(fbank_d)
		
		# con cat three data
		fbankda=np.concatenate( (fbank, fbank_d, fbank_a),axis=1)
		
		return fbankda 


	def cal_melmat(self,):
		nv2 = self.fft_len/2    #w->fb.fftN / 2; 512/2=256
		fres = self.sampling_rate / (self.fft_len * 700.0)  # freq_d700
		maxChan = self.num_banks + 1
		
		Mel = lambda k, freq_d700  : 1127.0 * (np.log(1.0 + (k - 1) * (freq_d700)))
		
		klo = 2
		khi = nv2
		mlo = 0
		mhi = Mel(nv2 + 1, fres)
		#print (' mlo, mhi ', mlo, mhi)
		
		cf=np.zeros( maxChan+1)   # size is numbank+2
		ms = mhi - mlo
		for chan in range(1, maxChan+1): 
			cf[chan] = (1.0 * chan / maxChan)*ms + mlo
		#print ('center of each channel',cf)
		
		loChan = np.zeros(nv2 + 1 +1)
		chan=1
		for k in range(1, nv2+1):
			if k < klo or k > khi:
				loChan[k] = -1
			else:
				melk = Mel(k, fres)
				while cf[chan] < melk and chan <= maxChan:
					chan+=1
				loChan[k] = chan - 1
		
		#print('loChan', loChan)
		
		loWt = np.zeros(nv2 + 1 +1)
		for k in range (1,nv2+1):
			chan = int(loChan[k])
			if k < klo or k > khi :
				loWt[k] = 0.0
			else:
				if chan > 0 :
					loWt[k] = (cf[chan + 1] - Mel(k, fres)) / (cf[chan + 1] - cf[chan])
				else:
					loWt[k] = (cf[1] - Mel(k, fres)) / (cf[1] - mlo)
		
		#print ('loWt', loWt)
		
		melmat=np.zeros((self.num_banks, self.fft_data_len))
		
		for k in range ( klo, khi+1):
			#A=spec[k-1] 
			bin = int(loChan[k])
			if bin > 0:
				melmat[bin-1][k-1] +=loWt[k]
			if bin < self.num_banks : 
				melmat[bin][k-1] += (1.0 - loWt[k])
		#return fbank[1:]
		
		return melmat, loChan, loWt, klo, khi
	
	
	


# this file use TAB
