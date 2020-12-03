import os
import torch
import matplotlib.pyplot as plt
import pickle
import numpy as np
from hparams import hparams
from utils import pad_seq_to_2
from utils import quantize_f0_numpy
from model import Parrot as Generator
import argparse
# from model import Generator_6 as F0_Converter
device = 'cuda:0'

'''config'''
def str2bool(v):
    return v.lower() in ('true')
parser = argparse.ArgumentParser()

# Training configuration.
parser.add_argument('--num_iters', type=int, default=1000000, help='number of total iterations')
# 100w step
parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for Adam optimizer')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
parser.add_argument('--ortho_lr', type=float, default=0.0001, help='learning rate for OrthoDisen')
# parser.add_argument('--beta1_ortho', type=float, default=0.9, help='beta1 for Adam_ortho optimizer')
# parser.add_argument('--beta2_ortho', type=float, default=0.999, help='beta2 for Adam_ortho optimizer')
parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')

# Miscellaneous.
parser.add_argument('--use_tensorboard', type=str2bool, default=True)
parser.add_argument('--device_id', type=int, default=0)

# Directories.
parser.add_argument('--log_dir', type=str, default='run/logs')
parser.add_argument('--model_save_dir', type=str, default='run/models')
# model 不止保存 G 的参数还要保存 Ortho 模块的参数
parser.add_argument('--sample_dir', type=str, default='run/samples')

# Step size.
parser.add_argument('--log_step', type=int, default=10)
parser.add_argument('--sample_step', type=int, default=1000)
parser.add_argument('--model_save_step', type=int, default=1000)

config = parser.parse_args()

'''config'''

G = Generator(hparams, config).eval().to(device)
#g_checkpoint = torch.load('pre-trained-model/660000-G.ckpt', map_location=lambda storage, loc: storage)
#g_checkpoint = torch.load('/home/jie-wang19/Speechsplitexp/speech_split_baseline/Base_origin/pre-trained-model/660000-G.ckpt', map_location=lambda storage, loc: storage)
# g_checkpoint = torch.load('/home/jie-wang19/SpflowFindBest/Base_a_5/run_base/models/108000-G.ckpt', map_location=lambda storage, loc: storage)
g_checkpoint = torch.load('./run_l1/models/139000.ckpt', map_location=lambda storage, loc: storage)
G.load_state_dict(g_checkpoint['model'])


#metadata = pickle.load(open('assets/demo.pkl', "rb"))
# metadata = pickle.load(open('/home/jie-wang19/Speechsplitexp/speech_split_baseline/Base_origin/pre-trained-model/demo.pkl', "rb"))
# metadata = pickle.load(open('/home/jie-wang19/Speechsplitexp/speech_split_baseline/Base_a_15/assets/spmel/train.pkl', "rb"))
# metadata = pickle.load(open('/home/jie-wang19/Speechsplitexp/speech_split_baseline/Base_a_15/assets/spmel/train.pkl', "rb"))
metadata = pickle.load(open('./assets/spmel/test.pkl',"rb"))
sbmt_i = metadata[-1]# -2 p364_308 -1 p364_309 #236_456 104 #231_456 082
# P225
# 取出 speaker i 的信息，一共有3维：
# 0 ：speaker名字
# 1 ：speaker的onehot
# 2 ：speaker的mel、F0、长度、uid

emb_org = torch.from_numpy(sbmt_i[1])[np.newaxis,:].to(device)
#emb_org = torch.from_numpy(np.pad(sbmt_i[1],(0,62),'constant',constant_values=(0))[np.newaxis, :]).to(device)
# i spk的one-hot的编码表示
path = './'
# x_org = np.load(os.path.join(path, 'spmel/p225/p225_001.npy'))
# f0_org = np.load(os.path.join(path, 'raptf0/p225/p225_001.npy'))
# len_org = x_org.shape[0]
# uid_org = sbmt_i[2][0]
x_org, f0_org, len_org, uid_org = sbmt_i[2]
# i speaker 的utterance等信息（选了某一句？）
uttr_org_pad, len_org_pad = pad_seq_to_2(x_org[np.newaxis, :, :], 400)
uttr_org_pad = torch.from_numpy(uttr_org_pad).to(device)
f0_org_pad = np.pad(f0_org, (0, 400 - len_org), 'constant', constant_values=(0, 0))
f0_org_quantized = quantize_f0_numpy(f0_org_pad)[0]
f0_org_onehot = f0_org_quantized[np.newaxis, :, :]
f0_org_onehot = torch.from_numpy(f0_org_onehot).to(device)
uttr_f0_org = torch.cat((uttr_org_pad, f0_org_onehot), dim=-1)

sbmt_j = metadata[3] # 3 p225_335 #11 225_351 #0 225_331
# P231
emb_trg = torch.from_numpy(sbmt_j[1])[np.newaxis,:].to(device)
#emb_trg = torch.from_numpy(np.pad(sbmt_j[1],(0,62),'constant',constant_values=(0))[np.newaxis, :]).to(device)

# path = '/home/jie-wang19/Speechsplitexp/speech_split_baseline/Base_origin/assets'
# x_trg = np.load(os.path.join(path, 'spmel/p226/p226_001.npy'))
# f0_trg = np.load(os.path.join(path, 'raptf0/p226/p226_001.npy'))
# len_trg = x_trg.shape[0]
# uid_trg = sbmt_j[2][0]

x_trg, f0_trg, len_trg, uid_trg = sbmt_j[2]
uttr_trg_pad, len_trg_pad = pad_seq_to_2(x_trg[np.newaxis, :, :], 400)
uttr_trg_pad = torch.from_numpy(uttr_trg_pad).to(device)
f0_trg_pad = np.pad(f0_trg, (0, 400 - len_trg), 'constant', constant_values=(0, 0))
f0_trg_quantized = quantize_f0_numpy(f0_trg_pad)[0]
f0_trg_onehot = f0_trg_quantized[np.newaxis, :, :]
f0_trg_onehot = torch.from_numpy(f0_trg_onehot).to(device)

# with torch.no_grad():
#     f0_pred = P(uttr_org_pad, f0_trg_onehot)[0]
#     # 经过P（F0 converter），就是encoder？
#     f0_pred_quantized = f0_pred.argmax(dim=-1).squeeze(0)
#     f0_con_onehot = torch.zeros((1, 400, 257), device=device)
#     f0_con_onehot[0, torch.arange(400), f0_pred_quantized] = 1
# uttr_f0_trg = torch.cat((uttr_org_pad, f0_con_onehot), dim=-1)
# f0_trg = org + f0_onehot


uttr_f0_trg_NOP = torch.cat((uttr_org_pad, f0_trg_onehot), dim=-1)


# conditions = ['F']
conditions = ['R', 'F', 'U', 'RF', 'RU', 'FU', 'RFU']
# R：rhythm
# F：pitch
# U：timbre
# spect_vc = []
spect_vc_NOP= []
with torch.no_grad():
    for condition in conditions:
        if condition == 'R':
            # x_identic_val = G(uttr_f0_org, uttr_trg_pad, emb_org)
            # x_identic_val_zero =G(uttr_f0_org, torch.zeros_like(uttr_trg_pad), emb_org) 
            x_NOP, _, _, _, _ = G(uttr_f0_org, uttr_trg_pad, emb_org)
            # 经过生成器就是decoder？
        if condition == 'F':
          #  x_identic_val = G(uttr_f0_trg, uttr_org_pad, emb_org)
            x_NOP, _, _, _, _ = G(uttr_f0_trg_NOP, uttr_org_pad, emb_org )
        if condition == 'U':
        #     x_identic_val = G(uttr_f0_org, uttr_org_pad, emb_trg)
            x_NOP, _, _, _, _ = G(uttr_f0_org, uttr_org_pad, emb_trg )
        if condition == 'RF':
        #     x_identic_val = G(uttr_f0_trg, uttr_trg_pad, emb_org)
            x_NOP, _, _, _, _ = G(uttr_f0_trg_NOP, uttr_trg_pad, emb_org )
        if condition == 'RU':
        #     x_identic_val = G(uttr_f0_org, uttr_trg_pad, emb_trg)
            x_NOP, _, _, _, _ = G(uttr_f0_org, uttr_trg_pad, emb_trg )
        if condition == 'FU':
        #     x_identic_val = G(uttr_f0_trg, uttr_org_pad, emb_trg)
            x_NOP, _, _, _, _ = G(uttr_f0_trg_NOP, uttr_org_pad, emb_trg )
        if condition == 'RFU':
        #     x_identic_val = G(uttr_f0_trg, uttr_trg_pad, emb_trg)
            x_NOP, _, _, _, _ = G(uttr_f0_trg_NOP, uttr_trg_pad, emb_trg )

        # # 这里的语句意思是不管转不转换R，都需要截取？
        # # 那可以保证截取到的是语义完整的一句话？
        if 'R' in condition:
            # uttr_trg = x_identic_val[0, :len_trg, :].cpu().numpy()
            uttr_trg_NOP = x_NOP[0, :len_trg, :].cpu().numpy()
        else:
            #uttr_trg = x_identic_val[0, :len_org, :].cpu().numpy()
            uttr_trg_NOP = x_NOP[0, :len_org, :].cpu().numpy()

        # fig, (ax1) = plt.subplots(1, 1, sharex=True)
        # min_value = np.min(x_identic_val_zero[0].cpu().numpy().T)
        # max_value = np.max(x_identic_val_zero[0].cpu().numpy().T)
        # im1 = ax1.imshow(x_identic_val_zero.cpu().numpy().T, aspect='auto', vmin=min_value, vmax=max_value)
        # plt.savefig('/home/jie-wang19/P-net_test/{source}_{target}_{uid_org}_{condition}_lost.png'.format(source=sbmt_i[0],target=sbmt_j[0],uid_org=uid_org,condition=condition), dpi=150)
        # plt.close(fig)

        #spect_vc.append(('{}_{}_{}_{}_{}'.format(sbmt_i[0], sbmt_j[0], uid_org, uid_trg, condition), uttr_trg))
        spect_vc_NOP.append(('{}_{}_{}_{}_{}_NOP'.format(sbmt_i[0], sbmt_j[0], uid_org, uid_trg, condition), uttr_trg_NOP))
    # %%

# spectrogram to waveform
import torch
import librosa
import pickle
import os
from synthesis import build_model
from synthesis import wavegen
# import audio
# if not os.path.exists('results'):
#     os.makedirs('results')
# if not os.path.exists('results_p'):
#     os.makedirs('results_p')
if not os.path.exists('results_nop_1_rhym'):
    os.makedirs('results_nop_1_rhym')
model = build_model().to(device)
# checkpoint = torch.load("/home/jie-wang19/Speechsplitexp/speech_split_baseline/Base_origin/pre-trained-model/checkpoint_step001000000_ema.pth")
checkpoint = torch.load("/datapool/home/zxt20/JieWang2020ICASSP/SpeechFlow-master_ordin/pre-trained-model/wave_netcheckpoint_step001000000_ema.pth")
# 预训练好的wavenet vocoder
model.load_state_dict(checkpoint["state_dict"])
# i = 0
# for spect in spect_vc:
#     # i += 1
#     name = spect[0]
#     c = spect[1]
#     #waveform = audio.inv_mel_spectrogram(c.T, hparams)
#     print(name)
#     waveform = wavegen(model, c=c)
#     librosa.output.write_wav('results_p/' + name + '.wav', waveform, sr=16000)

for sp in spect_vc_NOP:
    nn = sp[0]
    c = sp[1]
    print(nn)
    waveform_NOP = wavegen(model, c=c)
    librosa.output.write_wav('results_nop_1_rhym/' + nn + '.wav', waveform_NOP, sr=16000)
# %%
