import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import fftconvolve
import pyroomacoustics as pra
import random
import pickle

def clip(signal, high, low):
    '''
    Clip a signal from above at high and from below at low.
    '''
    s = signal.copy()

    s[np.where(s > high)] = high
    s[np.where(s < low)] = low

    return s

def normalize(signal, bits=None):
    '''
    normalize to be in a given range. 
    '''

    s = signal.copy()
    s /= np.abs(s).max()

    # if one wants to scale for bits allocated
    if bits is not None:
        s *= 2 ** (bits - 1) - 1
        s = clip(s, 2 ** (bits - 1) - 1, -2 ** (bits - 1))

    return s

# name of the rir
data = np.loadtxt("./read_data.txt", dtype=str, delimiter=",")
name = data[0, 0]
name_sub, _ = name.split('.')

# read one audio
fs, signal = wavfile.read(name)

temp = 0
room_settings = []
# set the number of room reverberations that you want to create
num_rooms = 1500

for i in range(num_rooms):
	print("Saved room reverberation: " + str(i))
	width = random.randint(3, 5)
	length = random.randint(4, 6)
	height = random.randint(2, 4)

	room_dim = [width, length, height]
	x_source = random.randint(0, width*10)/10.
	y_source = random.randint(0, length*10)/10.
	z_source = random.randint(0, height*10)/10.

	x_mic = random.randint(0, width*10)/10.
	y_mic = random.randint(0, length*10)/10.
	z_mic = random.randint(0, height*10)/10.

	source = [x_source, y_source, z_source]
	microphone = np.array([[x_mic], [y_mic], [z_mic]])

	room_setting = [width, length, height, x_source, y_source, z_source, x_mic, y_mic, z_mic]

	if room_setting not in room_settings:
		temp += 1			
			
		room_settings.append(room_setting)
		max_order =100

		# set max_order to a low value for a quick (but less accurate) RIR
		room = pra.ShoeBox(room_dim, fs=fs, max_order=max_order, absorption=0.2)

		# add source and set the signal to WAV file content
		room.add_source(source, signal=signal)

		# add two-microphone array
		room.add_microphone_array(pra.MicrophoneArray(microphone, room.fs))

		# compute image sources
		room.image_source_model(use_libroom=True)

		room.compute_rir()
		rir = room.rir[0][0]

		# save the room reverberations
		wavfile.write(name_sub +"_rir_" + str(temp) + '.wav', 16000, rir)

with open('room_setting.data', 'wb') as f: 
	pickle.dump(room_settings, f)
		

