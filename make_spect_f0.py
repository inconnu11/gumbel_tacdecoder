import os
import sys
import pickle
import numpy as np
import soundfile as sf
from scipy import signal
from librosa.filters import mel
from numpy.random import RandomState
from pysptk import sptk
from utils import butter_highpass
from utils import speaker_normalization
from utils import pySTFT

    

mel_basis = mel(16000, 1024, fmin=90, fmax=7600, n_mels=80).T
min_level = np.exp(-100 / 20 * np.log(10))
b, a = butter_highpass(30, 16000, order=5)

if not os.path.exists('assets/data_train/train_chosen.pkl'):
    spk2gen = pickle.load(open('assets/data_train/spk2gen.pkl', "rb"))
    spkgencollectman = {}
    mancounter=0
    spkgencollectwoman = {}
    womancounter=0
    for key,value in spk2gen.items():
        if key =='p315' or key =='p376':
            continue# not found?
        # if mancounter==10 and womancounter ==10:
        #     break
        if value=='M':
            #if mancounter<10:
            spkgencollectman[key] = value
            mancounter=mancounter+1
        else:
            #if womancounter<10:
            spkgencollectwoman[key] = value
            womancounter=womancounter+1
        
        
    spkchosen = spkgencollectwoman.copy()#62 F 45 M
    spkchosen.update(spkgencollectman)

    with open('assets/data_train/train_chosen.pkl', 'wb') as handle:
        pickle.dump(spkchosen, handle) 

spk2gen = pickle.load(open('assets/data_train/train_chosen.pkl',"rb"))
# Modify as needed
# rootDir = 'assets/wavs'
rootDir = '/datapool/home/zxt20/MYDATA/VCTK/VCTK_106_trim'
targetDir_f0 = 'assets/raptf0'
targetDir = 'assets/spmel'

   
dirName, subdirList, _ = next(os.walk(rootDir))
print('Found directory: %s' % dirName)

for subdir in sorted(subdirList):
    
   
    if subdir not in spk2gen.keys():
        continue
    print(subdir)
   
    if not os.path.exists(os.path.join(targetDir, subdir)):
        os.makedirs(os.path.join(targetDir, subdir))
    if not os.path.exists(os.path.join(targetDir_f0, subdir)):
        os.makedirs(os.path.join(targetDir_f0, subdir))    
    _,_, fileList = next(os.walk(os.path.join(dirName,subdir)))
    
    if spk2gen[subdir] == 'M': #是指low high 的基频吗
        lo, hi = 50, 250
    elif spk2gen[subdir] == 'F':
        lo, hi = 100, 600
    else:
        raise ValueError
        
    prng = RandomState(int(subdir[1:])) 
    for fileName in sorted(fileList):
        # read audio file
        x, fs = sf.read(os.path.join(dirName,subdir,fileName))
        assert fs == 16000
        if x.shape[0] % 256 == 0:
            x = np.concatenate((x, np.array([1e-06])), axis=0)
        y = signal.filtfilt(b, a, x)
        wav = y * 0.96 + (prng.rand(y.shape[0])-0.5)*1e-06
        
        # compute spectrogram
        D = pySTFT(wav).T
        D_mel = np.dot(D, mel_basis)
        D_db = 20 * np.log10(np.maximum(min_level, D_mel)) - 16
        S = (D_db + 100) / 100        
        
        # extract f0
        f0_rapt = sptk.rapt(wav.astype(np.float32)*32768, fs, 256, min=lo, max=hi, otype=2)
        index_nonzero = (f0_rapt != -1e10)
        mean_f0, std_f0 = np.mean(f0_rapt[index_nonzero]), np.std(f0_rapt[index_nonzero])
        f0_norm = speaker_normalization(f0_rapt, index_nonzero, mean_f0, std_f0)
        
        assert len(S) == len(f0_rapt)
            
        np.save(os.path.join(targetDir, subdir, fileName[:-4]),
                S.astype(np.float32), allow_pickle=False)    
        np.save(os.path.join(targetDir_f0, subdir, fileName[:-4]),
                f0_norm.astype(np.float32), allow_pickle=False)
