import scipy.io.wavfile as wav
#import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft
from scipy.fftpack import ifft
from scipy import signal
import scipy
import librosa

def compute_PSD_matrix(audio, window_size):
    win = np.sqrt(8.0/3.) * librosa.core.stft(audio, center=False)
    z = abs(win / window_size)
    psd_max = np.max(z*z)
    psd = 10 * np.log10(z * z + 0.0000000000000000001)
    psd_max_ori = np.max(psd)
    PSD = 96 - np.max(psd) + psd
    
    #PSD = psd
    return PSD, psd_max_ori, psd_max   

def Bark(f):
    """returns the bark-scale value for input frequency f (in Hz)"""
    return 13*np.arctan(0.00076*f) + 3.5*np.arctan(pow(f/7500.0, 2))

def quiet(f):
     """returns threshold in quiet measured in SPL at frequency f (in Hz)"""
     thresh = 3.64*pow(f*0.001,-0.8) - 6.5*np.exp(-0.6*pow(0.001*f-3.3,2)) + 0.001*pow(0.001*f,4) - 12
     return thresh

# using two slopes as the spread function due to its conservative
def two_slops(bark_psd, delta_TM, bark_maskee):
    L_TM = []
    for tone_mask in range(bark_psd.shape[0]):
        bark_masker = bark_psd[tone_mask, 0]
        dz = bark_maskee - bark_masker
        zero_index = np.argmax(dz > 0)
        sf = np.zeros(len(dz))
        sf[:zero_index] = 27 * dz[:zero_index]
        sf[zero_index:] = (-27 + 0.37 * max(bark_psd[tone_mask, 1] - 40, 0)) * dz[zero_index:] 
        l_tm = bark_psd[tone_mask, 1] + delta_TM[tone_mask] + sf
        L_TM.append(l_tm)
    return L_TM
    
# using model 1 as the spread function 
def model1(bark_psd, delta_TM, bark_maskee):
    L_TM = []
    for tone_mask in range(bark_psd.shape[0]):
        bark_masker = bark_psd[tone_mask, 0]
        dz = bark_maskee - bark_masker
        indexes = [np.argmax(dz >= -3), np.argmax(dz > -1), np.argmax(dz > 0), np.argmax(dz > 1), np.argmax(dz >= 8)]
        if indexes[3] == 0:
            indexes[3] = len(bark_maskee)
            indexes[4] = len(bark_maskee)
        if indexes[4] == 0:
            indexes[4] = len(bark_maskee)
        sf = np.zeros(len(dz)) - np.inf
        if indexes[1] != indexes[0]:
            sf[indexes[0] : indexes[1]] = 17 * dz[indexes[0] : indexes[1]] - 0.4 * bark_psd[tone_mask, 1] + 11
        sf[indexes[1] : indexes[2]] = (0.4 * bark_psd[tone_mask, 1] + 6) * dz[indexes[1] : indexes[2]]  
        sf[indexes[2] : indexes[3]] = -17 * dz[indexes[2] : indexes[3]]  
        if indexes[3] != indexes[4]:
            sf[indexes[3] : indexes[4]] = -17 * dz[indexes[3] : indexes[4]] + 0.15 * bark_psd[tone_mask, 1] * (dz[indexes[3] : indexes[4]] - 1)
            
        l_tm = bark_psd[tone_mask, 1] + delta_TM[tone_mask] + sf
        L_TM.append(l_tm)
    return L_TM
      
def compute_th(PSD, barks, ATH, freqs):
    # Identification of tonal and nontonal maskers
    # masker is the index of the number of points
    
    length = len(PSD)
    masker_index = signal.argrelextrema(PSD, np.greater)[0]
    num_local_max = len(masker_index)
    #print("number of local maximum: {}".format(num_local_max))
    
    # delete the boundary of masker for smoothing
    if 0 in masker_index:
        masker_index = np.delete(0)
    if length - 1 in masker_index:
        masker_index = np.delete(length - 1)
    p_k = pow(10, PSD[masker_index]/10.)    
    p_k_prev = pow(10, PSD[masker_index - 1]/10.)
    p_k_post = pow(10, PSD[masker_index + 1]/10.)


    # treat all the maskers as tonal (conservative way)
    # smooth the PSD 
    P_TM = 10 * np.log10(p_k_prev + p_k + p_k_post)
    
    # bark_psd: the first column bark, the second column: P_TM, the third column: the index of points
    _BARK = 0
    _PSD = 1
    _INDEX = 2
    bark_psd = np.zeros([num_local_max, 3])
    bark_psd[:, _BARK] = barks[masker_index]
    bark_psd[:, _PSD] = P_TM
    bark_psd[:, _INDEX] = masker_index
    
    # delete the masker that are less than one half of critical band width from a neighbouring component
    for i in range(num_local_max):
        next = i + 1
        if next >= bark_psd.shape[0]:
            break
            
        while bark_psd[next, _BARK] - bark_psd[i, _BARK]  < 0.5:
            # masker must be higher than quiet threshold
            if quiet(freqs[int(bark_psd[i, _INDEX])]) > bark_psd[i, _PSD]:
                bark_psd = np.delete(bark_psd, (i), axis=0)
            if next == bark_psd.shape[0]:
                break
                
            if bark_psd[i, _PSD] < bark_psd[next, _PSD]:
                bark_psd = np.delete(bark_psd, (i), axis=0)
            else:
                bark_psd = np.delete(bark_psd, (next), axis=0)
            if next == bark_psd.shape[0]:
                break        
                   
    #print("the shape of bark_psd: {}".format(bark_psd.shape))
    
    # compute the individual masking thresholds
    # assume delta_TM for TMN is three twice of TMT 
    delta_TM = 1 * (-6.025  -0.275 * bark_psd[:, 0])
    L_TM = two_slops(bark_psd, delta_TM, barks) 
    #L_TM = model1(bark_psd, filter_mask, delta_TM, barks) 
    L_TM = np.array(L_TM)
    
    # compute the global masking threshold
    #L_g = 10 * np.log10(np.sum(pow(10, L_TM/10.), axis=0) + pow(10, ATH/10.) + 0.00000000000000000000001 )
    L_g = np.sum(pow(10, L_TM/10.), axis=0) + pow(10, ATH/10.) 
 
    return L_g

def generate_th(audio, fs, window_size=2048):
    PSD, psd_max_ori, psd_max= compute_PSD_matrix(audio , window_size)  
    freqs = librosa.core.fft_frequencies(fs, window_size)
    barks = Bark(freqs)

    # count the quiet threshold after it bark > 1
    ATH = np.zeros(len(barks)) - np.inf
    bark_ind = np.argmax(barks > 1)
    ATH[bark_ind:] = quiet(freqs[bark_ind:])

    L_g = []
    for i in range(PSD.shape[1]):
        L_g.append(compute_th(PSD[:,i], barks, ATH, freqs))
    L_g = np.array(L_g)
    return L_g, psd_max_ori, psd_max
