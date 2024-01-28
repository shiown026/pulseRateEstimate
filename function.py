#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 15:11:34 2024

@author: chiahunglee
"""

import glob

import numpy as np
import scipy as sp
import scipy.io
from scipy.signal import find_peaks

def LoadTroikaDataset():

    data_dir = "./datasets/troika/training_data"
    data_fls = sorted(glob.glob(data_dir + "/DATA_*.mat"))
    ref_fls = sorted(glob.glob(data_dir + "/REF_*.mat"))
    return data_fls, ref_fls

def LoadTroikaDataFile(data_fl):
    
    data = sp.io.loadmat(data_fl)['sig']
    return data[2:]


def AggregateErrorMetric(pr_errors, confidence_est):
    """
    Computes an aggregate error metric based on confidence estimates.

    Computes the MAE at 90% availability. 

    Args:
        pr_errors: a numpy array of errors between pulse rate estimates and corresponding 
            reference heart rates.
        confidence_est: a numpy array of confidence estimates for each pulse rate
            error.

    Returns:
        the MAE at 90% availability
    """
    # Higher confidence means a better estimate. The best 90% of the estimates
    #    are above the 10th percentile confidence.
    percentile90_confidence = np.percentile(confidence_est, 10)

    # Find the errors of the best pulse rate estimates
    best_estimates = pr_errors[confidence_est >= percentile90_confidence]

    # Return the mean absolute error
    return np.mean(np.abs(best_estimates))

def Evaluate():
    """
    Top-level function evaluation function.

    Runs the pulse rate algorithm on the Troika dataset and returns an aggregate error metric.

    Returns:
        Pulse rate error on the Troika dataset. See AggregateErrorMetric.
    """
    # Retrieve dataset files
    data_fls, ref_fls = LoadTroikaDataset()
    errs, confs = [], []
    for data_fl, ref_fl in zip(data_fls, ref_fls):
        # Run the pulse rate algorithm on each trial in the dataset
        errors, confidence = RunPulseRateAlgorithm(data_fl, ref_fl)
        errs.append(errors)
        confs.append(confidence)
        # Compute aggregate error metric
    errs = np.hstack(errs)
    confs = np.hstack(confs)
    return errs, confs, AggregateErrorMetric(errs, confs)

def freqTransform(sig, freqs, fft_len):
    
    # Take an FFT of the normalized signal
    norm_sig = (sig - np.mean(sig))/(max(sig)-min(sig))
    fft_sig = np.fft.rfft(norm_sig, fft_len)
    
    # Calculate magnitude
    mag_freq_sig = np.abs(fft_sig)
    
    return mag_freq_sig, fft_sig

def bandpassFilter(signal, fs):
    '''Bandpass filter the signal between 40 and 240 BPM'''
    
    # Convert to Hz
    lo, hi = 40/60, 240/60
    
    b, a = sp.signal.butter(3, (lo, hi), btype='bandpass', fs=fs)
    return sp.signal.filtfilt(b, a, signal)
    
def RunPulseRateAlgorithm(data_fl, ref_fl):

    
    # Load data using LoadTroikaDataFile
    ppg, accx, accy, accz = LoadTroikaDataFile(data_fl)
    ref = sp.io.loadmat(ref_fl)
    
    
    Fs=125
    
    winSize = 8*Fs # Ground truth BPM provided in 8 second windows
    winShift = 2*Fs # Successive ground truth windows overlap by 2 seconds
    
    preds = []
    errs = []
    confs = []
    
    
    # ANALYSE WINDOW
    
    offset = 0
    
    for eval_window_idx in range(len(ref['BPM0'])):
        window_start = offset
        window_end = winSize+offset
        offset += winShift
        
        
        ppg_window = ppg[window_start:window_end]
        accx_window = accx[window_start:window_end]
        accy_window = accy[window_start:window_end]
        accz_window = accz[window_start:window_end]
        
        ppg_bandpass = bandpassFilter(ppg_window, fs=Fs)
        accx_bandpass = bandpassFilter(accx_window, fs=Fs)
        accy_bandpass = bandpassFilter(accy_window, fs=Fs)
        accz_bandpass = bandpassFilter(accz_window, fs=Fs)
        
        # Aggregate accelerometer data into single signal
        acc_mag_unfiltered = np.sqrt(accx_bandpass**2+accy_bandpass**2+accz_bandpass**2)
        acc_mag = bandpassFilter(acc_mag_unfiltered, fs=Fs)
        
        eaks = find_peaks(ppg_bandpass, height = 10, distance=35)[0]
        
        # Use FFT length larger than the input signal size for higher spectral resolution.
        fft_len=len(ppg_bandpass)*4
        
        # Create an array of frequency bins
        freqs = np.fft.rfftfreq(fft_len, 1 / Fs) # bins of width 0.12207031
        
        mag_freq_ppg, fft_ppg = freqTransform(ppg_bandpass, freqs, fft_len)
        mag_freq_acc, fft_acc = freqTransform(acc_mag, freqs, fft_len)
        
        peaks_ppg = find_peaks(mag_freq_ppg, height=30, distance=1)[0]
        peaks_acc = find_peaks(mag_freq_acc, height=30, distance=1)[0]
        
        # Sort peaks in order of peak magnitude
        sorted_freq_peaks_ppg = sorted(peaks_ppg, key=lambda i:mag_freq_ppg[i], reverse=True)
        sorted_freq_peaks_acc = sorted(peaks_acc, key=lambda i:mag_freq_acc[i], reverse=True)
        
        # Use the frequency peak with the highest magnitude, unless the peak is also present in the accelerometer peaks.
        use_peak = sorted_freq_peaks_ppg[0]
        
        for i in range(len(sorted_freq_peaks_ppg)):
            # Check nearest two peaks also
            cond1 = sorted_freq_peaks_ppg[i] in sorted_freq_peaks_acc
            cond2 = sorted_freq_peaks_ppg[i]-1 in sorted_freq_peaks_acc
            cond3 = sorted_freq_peaks_ppg[i]+1 in sorted_freq_peaks_acc
            if cond1 or cond2 or cond3:
                continue
            else:
                use_peak = sorted_freq_peaks_ppg[i]
                break
            
        chosen_freq = freqs[use_peak]
        pred = chosen_freq * 60 # pridict Heart rate in window
        conf = CalcConfidence(chosen_freq, freqs, fft_ppg)
        preds.append(pred)
        confs.append(conf)
        err = pred - ref['BPM0'][eval_window_idx][0]
        errs.append(err)
        
    errors, confidence = np.array(errs), np.array(confs)
    
    return errors, confidence



def CalcConfidence(chosen_freq, freqs, fft_ppg):
     '''
     Calculates a confidence value for a given frequency by computing
     the ratio of energy concentrated near that frequency compared to the full signal.
     '''
     win = (40/60.0)
     win_freqs = (freqs >= chosen_freq - win) & (freqs <= chosen_freq + win)
     abs_fft_ppg = np.abs(fft_ppg)
     
     # Sum frequency spectrum near pulse rate estimate and divide by sum of entire spectrum
     conf_val = np.sum(abs_fft_ppg[win_freqs])/np.sum(abs_fft_ppg)
     
     return conf_val

