import copy
import torch
import numpy as np
from scipy import signal
from librosa.filters import mel
from scipy.signal import get_window



def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs     # 奈奎斯特频率
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a
    
    
    
def pySTFT(x, fft_length=1024, hop_length=256):
    
    x = np.pad(x, int(fft_length//2), mode='reflect')
    
    noverlap = fft_length - hop_length
    shape = x.shape[:-1]+((x.shape[-1]-noverlap)//hop_length, fft_length)
    strides = x.strides[:-1]+(hop_length*x.strides[-1], x.strides[-1])
    result = np.lib.stride_tricks.as_strided(x, shape=shape,
                                             strides=strides)
    
    fft_window = get_window('hann', fft_length, fftbins=True)
    result = np.fft.rfft(fft_window * result, n=fft_length).T
    
    return np.abs(result)    



def speaker_normalization(f0, index_nonzero, mean_f0, std_f0):
    # f0 is logf0
    f0 = f0.astype(float).copy()
    #index_nonzero = f0 != 0
    f0[index_nonzero] = (f0[index_nonzero] - mean_f0) / std_f0 / 4.0
    f0[index_nonzero] = np.clip(f0[index_nonzero], -1, 1)
    f0[index_nonzero] = (f0[index_nonzero] + 1) / 2.0
    return f0


def quantize_f0_numpy(x, num_bins=256):
    # validation 的时候用_numpy
    # x is logf0
    assert x.ndim==1
    # shape 的数字有几个，ndim 就等于几
    x = x.astype(float).copy()
    uv = (x<=0)
    x[uv] = 0.0
    assert (x >= 0).all() and (x <= 1).all()
    x = np.round(x * (num_bins-1))
    x = x + 1
    x[uv] = 0.0
    enc = np.zeros((len(x), num_bins+1), dtype=np.float32)
    #  对每一个log F0进行量化，量化成一个256的向量，除此之外，enc会再加一维 vuv
    enc[np.arange(len(x)), x.astype(np.int32)] = 1.0
    # 257维的f0+vuv编码成one hot向量
    return enc, x.astype(np.int64)


def quantize_f0_torch(x, num_bins=256):
    # train的时候用_torch
    # x is logf0
    # [batch_size, seq_len, dim=1]
    B = x.size(0)
    x = x.view(-1).clone()
    # 展开成1维tensor
    # view 相当于 reshape
    uv = (x<=0)
    x[uv] = 0
    assert (x >= 0).all() and (x <= 1).all()
    # 条件为true时正常运行，否则触发异常
    # [0,1]之间
    x = torch.round(x * (num_bins-1))
    # [0,255]
    x = x + 1
    # 对每一个元素 +1？
    # x [1,256]之间
    x[uv] = 0
    enc = torch.zeros((x.size(0), num_bins+1), device=x.device)
    enc[torch.arange(x.size(0)), x.long()] = 1
    return enc.view(B, -1, num_bins+1), x.view(B, -1).long()



def get_mask_from_lengths(lengths, max_len):
    ids = torch.arange(0, max_len, device=lengths.device)
    mask = (ids >= lengths.unsqueeze(1)).bool()
    return mask
    
    

def pad_seq_to_2(x, len_out=128):
    len_pad = (len_out - x.shape[1])
    assert len_pad >= 0
    return np.pad(x, ((0,0),(0,len_pad),(0,0)), 'constant'), len_pad    