import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.autograd.Function as Function
import numpy as np
from hparams import hparams
import math
from math import ceil 
from utils import get_mask_from_lengths
from torch.autograd import Variable

class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


    
class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


class Encoder_t(nn.Module):
    """Rhythm Encoder
    """
    def __init__(self, hparams):
        super().__init__()

        self.dim_neck_2 = hparams.dim_neck_2
        # neck_2 = 1
        self.freq_2 = hparams.freq_2
        # freq_2 = 8
        self.dim_freq = hparams.dim_freq
        # dim_freq = 80
        self.dim_enc_2 = hparams.dim_enc_2
        # enc_2 = 128
        self.dim_emb = hparams.dim_spk_emb
        self.chs_grp = hparams.chs_grp
        # xx group
        
        convolutions = []
        for i in range(1):
            # 5*1的CNN，followed GN
            conv_layer = nn.Sequential(
                ConvNorm(self.dim_freq if i==0 else self.dim_enc_2,
                         self.dim_enc_2,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='relu'),
                nn.GroupNorm(self.dim_enc_2//self.chs_grp, self.dim_enc_2))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)
        
        self.lstm = nn.LSTM(self.dim_enc_2, self.dim_neck_2, 1, batch_first=True, bidirectional=True)
        # input_size = 128, hidden_size = 1, num_layers = 1
        

    def forward(self, x, mask):
                
        for conv in self.convolutions:
            x = F.relu(conv(x))
        x = x.transpose(1, 2)
        
        self.lstm.flatten_parameters()
        # 为了提高内存的利用率和效率，调用flatten_parameters让parameter的数据存放成contiguous chunk(连续的块)。
        # 类似调用tensor.contiguous

        outputs, _ = self.lstm(x)
        if mask is not None:
            outputs = outputs * mask
        out_forward = outputs[:, :, :self.dim_neck_2]
        out_backward = outputs[:, :, self.dim_neck_2:]
            
        codes = torch.cat((out_forward[:,self.freq_2-1::self.freq_2,:], out_backward[:,::self.freq_2,:]), dim=-1)
        # 降采样, downsample factor = 8

        return codes        
    
    
    
class Encoder_6(nn.Module):
    """F0 encoder
    """
    def __init__(self, hparams):
        super().__init__()

        self.dim_neck_3 = hparams.dim_neck_3
        self.freq_3 = hparams.freq_3
        self.dim_f0 = hparams.dim_f0
        self.dim_enc_3 = hparams.dim_enc_3
        self.dim_emb = hparams.dim_spk_emb
        self.chs_grp = hparams.chs_grp
        self.register_buffer('len_org', torch.tensor(hparams.max_len_pad))
        
        convolutions = []
        for i in range(3):
            conv_layer = nn.Sequential(
                ConvNorm(self.dim_f0 if i==0 else self.dim_enc_3,
                         self.dim_enc_3,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='relu'),
                nn.GroupNorm(self.dim_enc_3//self.chs_grp, self.dim_enc_3))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)
        
        self.lstm = nn.LSTM(self.dim_enc_3, self.dim_neck_3, 1, batch_first=True, bidirectional=True)
        
        self.interp = InterpLnr(hparams)

    def forward(self, x):
                
        for conv in self.convolutions:
            x = F.relu(conv(x))
            x = x.transpose(1, 2)
            x = self.interp(x, self.len_org.expand(x.size(0)))
            x = x.transpose(1, 2)
        x = x.transpose(1, 2)    
        
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)
        out_forward = outputs[:, :, :self.dim_neck_3]
        out_backward = outputs[:, :, self.dim_neck_3:]
        
        codes = torch.cat((out_forward[:,self.freq_3-1::self.freq_3,:],
                           out_backward[:,::self.freq_3,:]), dim=-1)    

        return codes 
    
    
    
class Encoder_7(nn.Module):
    """Sync Encoder module
    """
    def __init__(self, hparams):
        super().__init__()

        self.dim_neck = hparams.dim_neck
        self.freq = hparams.freq
        self.freq_3 = hparams.freq_3
        self.dim_enc = hparams.dim_enc
        self.dim_enc_3 = hparams.dim_enc_3
        self.dim_freq = hparams.dim_freq
        self.chs_grp = hparams.chs_grp
        self.register_buffer('len_org', torch.tensor(hparams.max_len_pad))
        self.dim_neck_3 = hparams.dim_neck_3
        self.dim_f0 = hparams.dim_f0
        # f0 : 257维
        
        # convolutions for code 1
        convolutions = []
        for i in range(3):
            # 5*1的CNN，followed GN
            conv_layer = nn.Sequential(
                ConvNorm(self.dim_freq if i==0 else self.dim_enc,
                         self.dim_enc,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='relu'),
                nn.GroupNorm(self.dim_enc//self.chs_grp, self.dim_enc))
            convolutions.append(conv_layer)
        self.convolutions_1 = nn.ModuleList(convolutions)
        
        self.lstm_1 = nn.LSTM(self.dim_enc, self.dim_neck, 2, batch_first=True, bidirectional=True)
        # input_size = embedding_size = dim_enc
        # hidden_size = dim_neck
        # num_layers = 2
        
        # convolutions for f0
        convolutions = []
        for i in range(3):
            # 5*1的CNN，followed GN
            conv_layer = nn.Sequential(
                ConvNorm(self.dim_f0 if i==0 else self.dim_enc_3,
                         self.dim_enc_3,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='relu'),
                nn.GroupNorm(self.dim_enc_3//self.chs_grp, self.dim_enc_3))
            convolutions.append(conv_layer)
        self.convolutions_2 = nn.ModuleList(convolutions)
        
        self.lstm_2 = nn.LSTM(self.dim_enc_3, self.dim_neck_3, 1, batch_first=True, bidirectional=True)
        # hidden size : 32
        
        self.interp = InterpLnr(hparams)

        
    def forward(self, x_f0):
        # random resampling
        
        x = x_f0[:, :self.dim_freq, :]
        # dim_freq = 80
        f0 = x_f0[:, self.dim_freq:, :]
        
        for conv_1, conv_2 in zip(self.convolutions_1, self.convolutions_2):
            x = F.relu(conv_1(x))
            f0 = F.relu(conv_2(f0))
            x_f0 = torch.cat((x, f0), dim=1).transpose(1, 2)
            x_f0 = self.interp(x_f0, self.len_org.expand(x.size(0)))
            # random resampling 同时对content、pitch中的rhythm信息破坏
            x_f0 = x_f0.transpose(1, 2)
            # x_f0 :[batch_size, feature_hidden, seq_len]
            x = x_f0[:, :self.dim_enc, :]
            # dim_enc : 512
            f0 = x_f0[:, self.dim_enc:, :]
            
        x_f0 = x_f0.transpose(1, 2)
        # x_f0 : [batch_size, seq_len, feature_hidden]
        x = x_f0[:, :, :self.dim_enc]
        f0 = x_f0[:, :, self.dim_enc:]
        
        # code 1
        x = self.lstm_1(x)[0]
        f0 = self.lstm_2(f0)[0]
        
        x_forward = x[:, :, :self.dim_neck]
        x_backward = x[:, :, self.dim_neck:]
        # dim_neck(blstm_dim) : 8
        
        f0_forward = f0[:, :, :self.dim_neck_3]
        f0_backward = f0[:, :, self.dim_neck_3:]
        # dim_neck_3(blstm_dim) : 32

        # down sampling
        codes_x = torch.cat((x_forward[:,self.freq-1::self.freq,:], 
                             x_backward[:,::self.freq,:]), dim=-1)
        
        codes_f0 = torch.cat((f0_forward[:,self.freq_3-1::self.freq_3,:], 
                              f0_backward[:,::self.freq_3,:]), dim=-1)
        
        return codes_x, codes_f0

    
class Decoder_3(nn.Module):
    """Decoder module
    """
    def __init__(self, hparams):
        super().__init__()
        self.dim_neck = hparams.dim_neck
        #self.dim_neck_2 = hparams.dim_neck_2
        self.dim_neck_2 = hparams.enc_mbv_size
        self.dim_emb = hparams.dim_spk_emb
        # spk embedding：82
        self.dim_freq = hparams.dim_freq
        self.dim_neck_3 = hparams.dim_neck_3
        
        # self.lstm = nn.LSTM(self.dim_neck*2+self.dim_neck_2*2+self.dim_neck_3*2+self.dim_emb, 
        #                     512, 3, batch_first=True, bidirectional=True)
        self.lstm = nn.LSTM(self.dim_neck*2+self.dim_neck_2+self.dim_neck_3*2+self.dim_emb, 
                            512, 3, batch_first=True, bidirectional=True)
        # 3层的bi-lstm
        
        self.linear_projection = LinearNorm(1024, self.dim_freq)
        # dim_freq : 输出80维的mel

    def forward(self, x):
        
        outputs, _ = self.lstm(x)
        
        decoder_output = self.linear_projection(outputs)

        return decoder_output          
    

class Decoder_4(nn.Module):
    """For F0 converter
    """
    def __init__(self, hparams):
        super().__init__()
        self.dim_neck_2 = hparams.dim_neck_2
        self.dim_f0 = hparams.dim_f0
        self.dim_neck_3 = hparams.dim_neck_3
        
        self.lstm = nn.LSTM(self.dim_neck_2*2+self.dim_neck_3*2, 
                            256, 2, batch_first=True, bidirectional=True)
        
        self.linear_projection = LinearNorm(512, self.dim_f0)

    def forward(self, x):
        
        outputs, _ = self.lstm(x)
        
        decoder_output = self.linear_projection(outputs)

        return decoder_output         
    
    

class Generator_3(nn.Module):
    """SpeechSplit model"""
    def __init__(self, hparams):
        super().__init__()
        
        self.encoder_1 = Encoder_7(hparams)
        self.encoder_2 = Encoder_t(hparams)
        self.decoder = Decoder_3(hparams)
    
        self.freq = hparams.freq
        self.freq_2 = hparams.freq_2
        # freq_2 = 8
        self.freq_3 = hparams.freq_3
        # freq_3 = 8


    def forward(self, x_f0, x_org, c_trg):
        
        x_1 = x_f0.transpose(2,1)
        codes_x, codes_f0 = self.encoder_1(x_1)
        code_exp_1 = codes_x.repeat_interleave(self.freq, dim=1)
        code_exp_3 = codes_f0.repeat_interleave(self.freq_3, dim=1)
        
        x_2 = x_org.transpose(2,1)
        codes_2 = self.encoder_2(x_2, None)
        code_exp_2 = codes_2.repeat_interleave(self.freq_2, dim=1)
        
        encoder_outputs = torch.cat((code_exp_1, code_exp_2, code_exp_3, 
                                     c_trg.unsqueeze(1).expand(-1,x_1.size(-1),-1)), dim=-1)
        
        mel_outputs = self.decoder(encoder_outputs)
        
        return mel_outputs
    
    
    def rhythm(self, x_org):
        x_2 = x_org.transpose(2,1)
        codes_2 = self.encoder_2(x_2, None)
        
        return codes_2

    
class Generator_6(nn.Module):
    """F0 converter
    """
    def __init__(self, hparams):
        super().__init__()
        
        self.encoder_2 = Encoder_t(hparams)
        self.encoder_3 = Encoder_6(hparams)
        self.decoder = Decoder_4(hparams)
        self.freq_2 = hparams.freq_2
        self.freq_3 = hparams.freq_3


    def forward(self, x_org, f0_trg):
        
        x_2 = x_org.transpose(2,1)
        codes_2 = self.encoder_2(x_2, None)
        code_exp_2 = codes_2.repeat_interleave(self.freq_2, dim=1)
        
        x_3 = f0_trg.transpose(2,1)
        codes_3 = self.encoder_3(x_3)
        code_exp_3 = codes_3.repeat_interleave(self.freq_3, dim=1)
        
        encoder_outputs = torch.cat((code_exp_2, code_exp_3), dim=-1)
        
        mel_outputs = self.decoder(encoder_outputs)
        
        return mel_outputs


class InterpLnr(nn.Module):
    # 实现 random resampling
    
    def __init__(self, hparams):
        super().__init__()
        self.max_len_seq = hparams.max_len_seq
        self.max_len_pad = hparams.max_len_pad
        # 192
        
        self.min_len_seg = hparams.min_len_seg
        # 19 frames
        self.max_len_seg = hparams.max_len_seg
        # 32 frames
        
        self.max_num_seg = self.max_len_seq // self.min_len_seg + 1

    def pad_sequences(self, sequences):
        channel_dim = sequences[0].size()[-1]
        out_dims = (len(sequences), self.max_len_pad, channel_dim)
        out_tensor = sequences[0].data.new(*out_dims).fill_(0)
        
        for i, tensor in enumerate(sequences):
            length = tensor.size(0)
            out_tensor[i, :length, :] = tensor[:self.max_len_pad]
            
        return out_tensor 

    def forward(self, x, len_seq):  
        
        if not self.training:
            return x
        
        device = x.device
        batch_size = x.size(0)
        
        # indices of each sub segment
        indices = torch.arange(self.max_len_seg*2, device=device).unsqueeze(0).expand(batch_size*self.max_num_seg, -1)
        # scales of each sub segment
        scales = torch.rand(batch_size*self.max_num_seg, 
                            device=device) + 0.5
        
        idx_scaled = indices / scales.unsqueeze(-1)
        idx_scaled_fl = torch.floor(idx_scaled)
        lambda_ = idx_scaled - idx_scaled_fl
        
        len_seg = torch.randint(low=self.min_len_seg, 
                                high=self.max_len_seg, 
                                size=(batch_size*self.max_num_seg,1),
                                device=device)
        
        # end point of each segment
        idx_mask = idx_scaled_fl < (len_seg - 1)
       
        offset = len_seg.view(batch_size, -1).cumsum(dim=-1)
        # cumsum 沿着某个轴的累加求和
        # offset starts from the 2nd segment
        offset = F.pad(offset[:, :-1], (1,0), value=0).view(-1, 1)
        # F.pad 是内置的矩阵填充方法：四维上下左右，六维上下左右前后
        
        idx_scaled_org = idx_scaled_fl + offset
        
        len_seq_rp = torch.repeat_interleave(len_seq, self.max_num_seg)
        idx_mask_org = idx_scaled_org < (len_seq_rp - 1).unsqueeze(-1)
        
        idx_mask_final = idx_mask & idx_mask_org
        
        counts = idx_mask_final.sum(dim=-1).view(batch_size, -1).sum(dim=-1)
        
        index_1 = torch.repeat_interleave(torch.arange(batch_size, device=device), counts)
        
        index_2_fl = idx_scaled_org[idx_mask_final].long()
        index_2_cl = index_2_fl + 1
        
        y_fl = x[index_1, index_2_fl, :]
        y_cl = x[index_1, index_2_cl, :]
        lambda_f = lambda_[idx_mask_final].unsqueeze(-1)
        
        y = (1-lambda_f)*y_fl + lambda_f*y_cl
        
        sequences = torch.split(y, counts.tolist(), dim=0)
       
        seq_padded = self.pad_sequences(sequences)
        
        return seq_padded


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class TransformerLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root)."""
        super(TransformerLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class TransformerSpecPredictionHead(nn.Module):
    def __init__(self, hparams, output_dim, device):
        super(TransformerSpecPredictionHead, self).__init__()
        self.output_dim = output_dim
        self.shapes=[2048,1024,512]
        input_shape = self.output_dim
        self.transform_act_fn = gelu
        self.dense_layers = []
        self.TransformerNorms = []
        self.device = device
        

        for output in self.shapes:
            dense_layer = nn.Linear(input_shape,output)
            LayerNorm = TransformerLayerNorm(output,eps=hparams.layer_norm_eps)
            self.dense_layers.append(dense_layer)
            self.TransformerNorms.append(LayerNorm)
            input_shape = output

        self.dense_final = nn.Linear(self.shapes[-1], hparams.dim_ortho)
        self.LayerNorm_final = TransformerLayerNorm(hparams.dim_ortho, eps=hparams.layer_norm_eps)
        self.output_final = nn.Linear(hparams.dim_ortho, self.output_dim)

        self.dense_layers = [self.dense_layer.to(self.device) for self.dense_layer in self.dense_layers]
        self.TransformerNorms = [self.TransformerNorm.to(self.device) for self.TransformerNorm in self.TransformerNorms]
        

    def forward(self, hidden_states):
        inputs = hidden_states.to(self.device)
        for dense_layer,LayerNorm in zip(self.dense_layers,self.TransformerNorms):
            hi = dense_layer(inputs)
            hid = self.transform_act_fn(hi)
            hidd = LayerNorm(hid)
            inputs=hidd
        hi = self.dense_final(inputs)
        hid = self.transform_act_fn(hi)
        hidd = self.LayerNorm_final(hid)
        linear_output = self.output_final(hidd)
        return linear_output


from torch.autograd import Function


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


class GRL(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def forward(self, input, alpha):
        return ReverseLayerF.apply(input, alpha)


# def gradient_rever(x, alpha):
#     return ReverseLayerF()(x, alpha)

# class GradReverse(Function):
#     def forward(self, x):
#         return x.view_as(x)
#
#     def backward(self, grad_output):
#         return (grad_output * -lambd)
class Mask_Softmax(nn.Module):
    def __init__(self, plus=1.0):
        super(Mask_Softmax, self).__init__()
        self.plus = plus
    def forward(self, logits):
        logits_exp = logits.exp()
        partition = logits_exp.sum(dim=-1, keepdim=True) + self.plus
        return logits_exp / partition
class Gumbel_Softmax(nn.Module):
    
    def __init__(self, temperature=1):
        super(Gumbel_Softmax, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        # initial temperature for gumbel softmax (default: 1)
        self.temperature = temperature
        self.mask_softmax = Mask_Softmax()

    def forward(self, logits, hard=True):
        y = self._gumbel_softmax_sample(logits, hard)
        return y

    def _sample_gumbel(self, shape, eps=1e-20):
        U = torch.rand(shape)
        return -torch.log(-torch.log(U + eps) + eps)

    def _gumbel_softmax_sample(self, logits, hard=False):#logit的输入范围是否后续有变化 是不是sample的交集过大
        # logits = self.softmax(logits)
        # logits_log = torch.log(logits)
        #sample = Variable(self._sample_gumbel(logits.size()[-1]), requires_grad=True) #长河
        sample = Variable(self._sample_gumbel(logits.size()), requires_grad=True)
        if logits.is_cuda:
            sample = sample.cuda()
        y = logits + sample
        #y = logits_log + sample
        y_soft =  self.softmax(y / self.temperature)
        #y_soft = self.mask_softmax(y / self.temperature)
        a = y_soft.data[0]

        if hard:
            # Straight through.
            index = y_soft.max(-1, keepdim=True)[1]
            y_hard = torch.zeros_like(logits).scatter_(-1, index, 1.0)
            ret = y_hard - y_soft.detach() + y_soft
        else:
            # Reparametrization trick.
            ret = y_soft
        return ret

class Parrot(nn.Module):
    """SpeechSplit model"""
    # 直接把 generator_3(原G) copy过来 + self.ortho
    # 原 generator_3 不用了，直接用encoder_1,encoder_2, decoder来取 原G 的parameters
    def __init__(self, hparams, config):
        super().__init__()

        self.embedding = nn.Embedding(hparams.dim_spk_emb, hparams.embedding_spk)
        self.gumbel_softmax = Gumbel_Softmax()
        self.MBVlinear = nn.Linear(hparams.dim_neck_2*2, hparams.enc_mbv_size*2)
        # 20,16
        std = math.sqrt(2.0 / (hparams.dim_spk_emb + hparams.embedding_spk))
        val = math.sqrt(3.0 ) * std
        self.embedding.weight.data.uniform_(-val, val)
        # print(self.embedding)
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:{}'.format(config.device_id) if self.use_cuda else 'cpu')

        # 设置各个的bottleneck feature 的 dim
        self.dim_neck = hparams.dim_neck
        self.dim_neck_2 = hparams.dim_neck_2
        self.dim_neck_3 = hparams.dim_neck_3
        self.dim_neck_4 = hparams.dim_spk_emb
        self.mask_content = torch.ones(4, self.dim_neck * 2)
        self.mask_content[0, :] = 0
        # self.mask_content[4, :] = 0
        #self.mask_rhythm = torch.ones(4, self.dim_neck_2 * 2)
        self.mask_rhythm = torch.ones(4, hparams.enc_mbv_size)
        self.mask_rhythm[1, :] = 0
        self.mask_pitch = torch.ones(4, self.dim_neck_3 * 2)
        self.mask_pitch[2, :] = 0
        # self.mask_pitch[4, :] = 0
        # self.mask_one_hot = torch.ones(5, self.dim_neck_4)
        self.mask_one_hot = torch.ones(4, hparams.embedding_spk)
        self.mask_one_hot[3, :] = 0
        # self.mask_one_hot[4, :] = 0
        # 得到一组[3, dim_content + dim_rhythm + dim_pitch]的矩阵
        self.mask_p = torch.cat((self.mask_content, self.mask_rhythm, self.mask_pitch,self.mask_one_hot), dim=1)
        
        
        self.mask_part = self.mask_p.repeat(hparams.max_len_pad//4 + 1, 1)[:hparams.max_len_pad,:]#padding to (408,c+r+f+e)

        #MBV拆开计算loss
        self.mask_C_P_U = torch.cat((self.mask_content, self.mask_pitch,self.mask_one_hot), dim=1)
        self.mask_C_P_U = self.mask_C_P_U.repeat(hparams.max_len_pad//4 + 1, 1)[:hparams.max_len_pad,:]
        self.invert_mask_C_P_U = (self.mask_C_P_U == 0).long()
        self.mask_R = self.mask_rhythm
        self.mask_R = self.mask_R.repeat(hparams.max_len_pad//4 + 1, 1)[:hparams.max_len_pad,:]
        self.invert_mask_R = (self.mask_R== 0).long()



        
        #self.mask_par = self.mask_pa.repeat(hparams.batch_size // 3, 1)
       
        # 1：predicted
        self.invert_mask = (self.mask_part == 0).long()

        self.batch_size = hparams.batch_size
        self.encoder_1 = Encoder_7(hparams)
        self.encoder_2 = Encoder_t(hparams)
        # encoder_2 : rhythm encoder
        self.decoder = Decoder_3(hparams)
        self.GRL = GRL(alpha=1)

        #self.ortho = TransformerSpecPredictionHead(hparams, self.dim_neck*2+self.dim_neck_2*2+self.dim_neck_3*2+hparams.embedding_spk, self.device)
        self.ortho = TransformerSpecPredictionHead(hparams, self.dim_neck*2+hparams.enc_mbv_size+self.dim_neck_3*2+hparams.embedding_spk, self.device)
        # 要设定一下输入的feature维度，在这里就是3种特征的维度加起来
        # self.ortho = TransformerSpecPredictionHead(hparams, self.dim_neck*2+self.dim_neck_2*2+self.dim_neck_3*2+self.dim_neck_4, self.device)

        self.freq = hparams.freq
        self.freq_2 = hparams.freq_2
        # freq_2 = 8
        self.freq_3 = hparams.freq_3
        # freq_3 = 8
        self.loss = nn.L1Loss()


    def grouped_parameters(self,):
        # 返回模型中所有的参数
        params_G = [p for p in self.encoder_1.parameters()]
        params_G.extend([p for p in self.encoder_2.parameters()])
        params_G.extend([p for p in self.decoder.parameters()])
        # parameters_main
        params_ortho = [p for p in self.ortho.parameters()]
        # parameters_orthogonal
        return params_G, params_ortho

    def forward(self, x_f0, x_org, c_trg):
        # print(c_trg.shape)

        # [16,20]16是bacth size
        # x_f0 : pitch contour P
        # x_org : speech S
        # c_trg : target speaker
        x_f0 = x_f0.to(self.device)
        x_org = x_org.to(self.device)
        c_trg = c_trg.to(self.device)
        x_1 = x_f0.transpose(2, 1)
        # [batch_size, dim, seq_len]
        codes_x, codes_f0 = self.encoder_1(x_1)
        code_exp_1 = codes_x.repeat_interleave(self.freq, dim=1)
        # content 上采样
        # [batch_size, seq_len * 8, dim]
        code_exp_3 = codes_f0.repeat_interleave(self.freq_3, dim=1)
        # pitch 上采样
        # code_1 : content 16=8*2(双向lstm)
        # code_3 : pitch 64=32*2(双向lstm)

        x_2 = x_org.transpose(2, 1)
        codes_2 = self.encoder_2(x_2, None)
        # code_2 : rhythm 2=1*2(双向lstm)
        code_exp_2 = codes_2.repeat_interleave(self.freq_2, dim=1)
        #MBV
        code_exp_2 = self.MBVlinear(code_exp_2)
        code_exp_2 = code_exp_2.view(code_exp_2.size(0), code_exp_2.size(1), hparams.enc_mbv_size, 2)#batch,t_step,7,2
        # rhythm 上采样
        code_exp_2 = self.gumbel_softmax(code_exp_2)[:,:,:,0]#取第一维
        #[16, 20]
        # aa = c_trg.unsqueeze(-1)
        bb = c_trg.cpu().numpy().tolist()
        # print(bb)
        # print(bb.size)
        # sp_query = torch.LongTensor()
        # sp_query = [ b.index(1) for b in bb]
        sp_query = []
        # for b in bb:
        #     try :
        #         sp_query.append(b.index(1))
        #     except ValueError as e:
        #         sp_query.append(0)
        for b in bb:
            if 1.0 in b:
                sp_query.append(b.index(1))
            else:
                sp_query.append(0)
        # print(sp_query)
        # print(c_trg.shape)
        # # c_trg = [1,20]
        # # sp_query = (c_trg==1).long()
        # # sp_query = np.where(c_trg==1)
        # aa = c_trg.cpu().numpy().tolist()
        # # print(aa)
        # sp_query = aa.index(1.0)
        # print(sp_query.shape)
        
        # sp_query.to(torch.Tensor)


        # sp_query = torch.Tensor(sp_query).unsqueeze(1).unsqueeze(1)
        # print(sp_query.shape)

        # speaker_embedding = [self.embedding(input=a) for a in sp_query]
        #[16, 16]
        # speaker_embedding = [self.embedding(input=sp) for sp in sp_query]
        speaker_embedding = self.embedding(torch.LongTensor(sp_query).to(self.device))
        #[16, 16]
        # print(speaker_embedding.shape)
        # speaker_embedding = [self.embedding(sp) for sp in sp_query]

        # speaker_embedding = [self.embedding(sp) for sp in sp_query]
        # # speaker_embedding [1, 16]
        # tt = speaker_embedding.unsqueeze(1).expand(-1, x_1.size(-1), -1)
        # print(tt.shape)


        encoder_outputs = torch.cat((code_exp_1, code_exp_2, code_exp_3,
                                     c_trg.unsqueeze(1).expand(-1, x_1.size(-1), -1)), dim=-1)
        
        # print(encoder_outputs.shape)
        # =================================================================================== #
        #                              mask and predict                                       #
        # =================================================================================== #
        # ortho 模块只需要返回mask的loss即可

        # ortho_inputs_integral = torch.cat((code_exp_1, code_exp_2, code_exp_3, speaker_embedding.unsqueeze(1).expand(-1, x_1.size(-1), -1)), dim=-1)#16,408,c+r+f+e 184
        ortho_inputs_integral = torch.cat((code_exp_1, code_exp_2, code_exp_3, speaker_embedding.unsqueeze(1).expand(-1, x_1.size(-1), -1)), dim=-1)#16,408,c+r+f+e 184
        # mask_part = self.mask_part.unsqueeze(1).expand(-1, x_1.size(-1), -1)
        # invert_mask = self.invert_mask.unsqueeze(1).expand(-1, x_1.size(-1), -1)
        #print(self.mask_part[:5,:])
        mask_part = self.mask_part #408,184
        
        invert_mask = self.invert_mask
        
        ortho_inp = ortho_inputs_integral.to(self.device) * mask_part.to(self.device) #(408,c+r+f+e)
        # 通过调用函数而不是类来完成这个操作
        # 通过梯度反转层gradient reversal layer
        ortho_inputs = self.GRL(ortho_inp, 1)
        self.ortho = self.ortho.to(self.device)
        ortho_inputs = ortho_inputs.to(self.device)
        feature_predict = self.ortho(ortho_inputs)
        
        # 模型会预测所有feature，但是我们希望只有mask掉的特征的loss对梯度进行更新，所有对于没有mask的将ground truth copy过去，这个结果就是第1项
        # loss_ortho = self.loss((feature_predict.to(self.device) * self.invert_mask.to(self.device)).clone() + ortho_inputs_integral.to(self.device) * self.mask_part.to(self.device), ortho_inputs_integral.to(self.device))
        # c_trg : timbre
        mel_outputs = self.decoder(encoder_outputs)

        
        #将Rhythm分开计算BCE loss。

        feature_predict_C_P_U = torch.cat((feature_predict[:,:,:self.dim_neck*2],feature_predict[:,:,self.dim_neck*2+hparams.enc_mbv_size:]),dim=-1)
        mask_part_C_P_U = self.mask_C_P_U
        invert_mask_C_P_U = self.invert_mask_C_P_U
        ortho_inputs_integral_C_P_U = torch.cat((ortho_inputs_integral[:,:,:self.dim_neck*2],ortho_inputs_integral[:,:,self.dim_neck*2+hparams.enc_mbv_size:]),dim=-1)


        feature_predict_R = feature_predict[:,:,self.dim_neck*2:self.dim_neck*2+hparams.enc_mbv_size]
        mask_part_R = self.mask_R
        invert_mask_R = self.invert_mask_R
        ortho_inputs_integral_R = ortho_inputs_integral[:,:,self.dim_neck*2:self.dim_neck*2+hparams.enc_mbv_size]


        # outputs = [mel_outputs, feature_predict, ortho_inputs_integral, mask_part, invert_mask]
        return mel_outputs, [feature_predict_C_P_U, feature_predict_R], [ortho_inputs_integral_C_P_U, ortho_inputs_integral_R],[mask_part_C_P_U,mask_part_R],[invert_mask_C_P_U,invert_mask_R]
        # return mel_outputs, loss_ortho

    def rhythm(self, x_org):
        x_2 = x_org.transpose(2, 1)
        codes_2 = self.encoder_2(x_2, None)

        return codes_2


# if __name__=='__main__':
#     T = TransformerSpecPredictionHead(hparams,58)
#     c = torch.ones((16,3,58))
#     d = T(c)
#     print(d)