# from model import Generator_3 as Generator
from model_frame import InterpLnr
# from model import OrthoDisen
from model_frame import Parrot
# from loss import ParrotLoss
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime
import pickle
import math

from taco2.logger import Tacotron2Logger
from utils import pad_seq_to_2, quantize_f0_torch, quantize_f0_numpy

# use demo data for simplicity
# make your own validation set as needed
validation_pt = pickle.load(open('./assets/spmel/validate.pkl', "rb"))


# 自己写validation set上的


class Solver(object):
    """Solver for training"""

    def __init__(self, vcc_loader, config, hparams):
        """Initialize configurations."""

        # Data loader.
        self.vcc_loader = vcc_loader
        self.hparams = hparams

        # Training configurations.
        self.num_iters = config.num_iters
        self.lr_main = config.g_lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters

        # weight
        self.w_main = hparams.loss_reconstruction_w + 1
        self.w_ortho = hparams.loss_disentanglement_w
        self.loss_o = torch.nn.L1Loss()
        self.loss_BCE = torch.nn.BCEWithLogitsLoss()
        # self.threshold = hparams.threshold

        # self.beta1_ortho = config.beta1_ortho
        # self.beta2_ortho = config.beta2_ortho
        # self.lr_ortho = config.ortho_lr

        # Miscellaneous.
        self.use_tensorboard = config.use_tensorboard
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:{}'.format(config.device_id) if self.use_cuda else 'cpu')

        # Directories.
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir
        self.output_dir = '/workspace/cpfs-data/gumbel_tacdecoder/taco2_output'

        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step

        # Build the model and tensorboard.
        self.build_model(config)
        if self.use_tensorboard:
            self.build_tensorboard()

        # torch.autograd.set_detect_anomaly(True)

    def prepare_directories_and_logger(self, output_directory, log_directory, rank):
        if rank == 0:
            if not os.path.isdir(output_directory):
                os.makedirs(output_directory)
                os.chmod(output_directory, 0o775)
            logger = Tacotron2Logger(os.path.join(output_directory, log_directory))
        else:
            logger = None
        return logger

    def build_model(self, config):
        # 定义模型
        self.model = Parrot(self.hparams, config)
        # self.G = Generator(self.hparams)
        # self.ortho = OrthoDisen(self.hparams)
        self.Interp = InterpLnr(self.hparams)
        #self.pre_r_encoder_path = '/datapool/home/zxt20/JieWang2021ICME/Baselines/pretrained/660000-G.ckpt'
        self.pre_r_encoder_path = '/workspace/cpfs-data/pretrained/660000-G.ckpt'
        self.logger = self.prepare_directories_and_logger(self.output_dir, self.log_dir, 0)
        # self.parameters_main = self.model.grouped_parameters()
        # 定义2个优化器optimizer
        # self.optimizer_ortho = torch.optim.Adam(self.parameters_orth, self.lr_ortho, [self.beta1_ortho, self.beta2_ortho])
        self.model ,self.frozen_state_dict= get_Rhythm_Encoder_parameters(self.pre_r_encoder_path,self.model)
        
        # self.parameters_main = self.model.grouped_parameters()
        # 定义2个优化器optimizer
        # self.optimizer_ortho = torch.optim.Adam(self.parameters_orth, self.lr_ortho, [self.beta1_ortho, self.beta2_ortho])
        self.optimizer_main = torch.optim.Adam(self.model.parameters(), self.lr_main, [self.beta1, self.beta2])
        # self.optimizer_ortho = torch.optim.Adam(self.parameters_orth, self.lr_ortho, [self.beta1, self.beta2])
        # betas:Tuple[float, float] 用于计算梯度以及梯度平方的运行平均值的系数
        # beta1： 一阶钜估计的指数衰减率
        # beta2：二阶矩估计的指数衰减
        # model.todevice
        self.print_network(self.model, 'model_overall')
        self.model.to(self.device)
        # self.G.to(self.device)
        # self.ortho.to(self.device)
        self.Interp.to(self.device)

        # 正交解耦模块的训练参数设置

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            # 统计模型的参数量
            num_params += p.numel()
            # p.numel 返回元素的数量
        print(model)
        # 把encoder、decoder以及各自的结构输出
        print(name)
        # 输出网络名称 G
        print("The number of parameters: {}".format(num_params))

    def print_optimizer(self, opt, name):
        print(opt)
        print(name)

    def restore_model(self, resume_iters):
        print('Loading the trained models from step {}...'.format(resume_iters))
        # G_path = os.path.join(self.model_save_dir, '{}.ckpt'.format(resume_iters))
        ckpt_path = os.path.join(self.model_save_dir, '{}.ckpt'.format(resume_iters))
        checkpoint_dict = torch.load(ckpt_path, map_location=lambda storage, loc: storage)

        self.model.load_state_dict(checkpoint_dict['model'])
        self.optimizer_main.load_state_dict(checkpoint_dict['optimizer_main'])
        self.lr_main = self.optimizer_main.param_groups[0]['lr']
        # self.optimizer_ortho.load_state_dict(checkpoint_dict['optimizer_ortho'])
        # self.lr_ortho = self.optimizer_ortho.param_groups[0]['lr']

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(self.log_dir)

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.optimizer_main.zero_grad()
        # self.optimizer_ortho.zero_grad()

    # =====================================================================================================================

    def train(self):
        # Set data loader.
        data_loader = self.vcc_loader

        # Fetch fixed inputs for debugging.
        data_iter = iter(data_loader)

        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            print('Resuming ...')
            start_iters = self.resume_iters
            self.num_iters += self.resume_iters
            self.restore_model(self.resume_iters)
            self.print_optimizer(self.optimizer_main, 'G_optimizer')
            # self.print_optimizer(self.optimizer_ortho, 'OrthoDisen_optimizer')

        # criterion = ParrotLoss(self.hparams).cuda()

        # Learning rate cache for decaying.
        lr_main = self.lr_main
        print('Current learning rates, lr_g: {}.'.format(lr_main))
        # lr_ortho = self.lr_ortho
        # print('Current learning rates, lr_ortho: {}.'.format(lr_ortho))

        # Print logs in specified order
        keys = ['overall/loss_id', 'main/loss_id', 'ortho/loss_id']

        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(start_iters, self.num_iters):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch real images and labels.
            try:
                x_real_org, emb_org, f0_org, len_org = next(data_iter)
            except:
                data_iter = iter(data_loader)
                x_real_org, emb_org, f0_org, len_org = next(data_iter)

            x_real_org = x_real_org.to(self.device)
            # x_real : melsp ？
            emb_org = emb_org.to(self.device)
            len_org = len_org.to(self.device)
            f0_org = f0_org.to(self.device)

            # =================================================================================== #
            #                               2. Train the generator                                #
            # =================================================================================== #

            self.model.train()
            b = list(self.model.named_parameters())
            # Identity mapping loss
            x_f0 = torch.cat((x_real_org, f0_org), dim=-1)
            # x_f0 : [batch_size, seq_len, dim_feature]
            x_f0_intrp = self.Interp(x_f0, len_org)
            # x_f0_intrp : [batch_size, seq_len, dim_feature]
            f0_org_intrp = quantize_f0_torch(x_f0_intrp[:, :, -1])[0]
            x_f0_intrp_org = torch.cat((x_f0_intrp[:, :, :-1], f0_org_intrp), dim=-1)
            # x_f0_intrp_org : [batch_size, seq_len, dim_feature]
            self.model = self.model.to(self.device)

            # x_identic = self.G(x_f0_intrp_org, x_real_org, emb_org)
            mel_outputs, feature_predicts, ortho_inputs_integrals, mask_parts, invert_masks = self.model(x_f0_intrp_org, x_real_org, emb_org, len_org, len_org, len_org)
            loss_main_id = F.mse_loss(x_real_org, mel_outputs, reduction='mean')

            loss_ortho_id_L1 = self.loss_o(ortho_inputs_integrals[0].cuda(),
                                        feature_predicts[0].cuda() * invert_masks[0].cuda() + ortho_inputs_integrals[0].cuda() * mask_parts[0].cuda())

            temp = feature_predicts[1].cuda() * invert_masks[1].cuda() + ortho_inputs_integrals[1].cuda() * mask_parts[1].cuda()
            loss_ortho_id_BCE = self.loss_BCE(feature_predicts[1].cuda() * invert_masks[1].cuda() + ortho_inputs_integrals[1].cuda() * mask_parts[1].cuda(),
                                        ortho_inputs_integrals[1].cuda())

            loss_main = loss_main_id
            loss_ortho_id = loss_ortho_id_L1+loss_ortho_id_BCE

            loss_ortho = loss_ortho_id

            ''''''
            w_ini = 100
            decay_rate = 0.999
            decay_steps = 12500
            ''''''
            # w_decay = w_ini * decay_rate ^ (i / decay_steps)
            w_decay = w_ini * math.pow(decay_rate , (i+1) / decay_steps)

            loss_overall_id = self.w_main * w_decay * loss_main + self.w_ortho * loss_ortho
            loss_overall = loss_overall_id / (self.w_main * w_ini)
            # identity:一致性，所以输入的都是original spk的
            # loss_main_id = F.mse_loss(x_real_org, x_identic, reduction='mean')
            # mse loss
            # Backward and optimize.
            # loss_main = loss_main_id
            # loss_ortho = loss_ortho_id
            self.reset_grad()

            """ loss_main 训练 """
            # for p in self.parameters_orth:
            #     p.requires_grad_(requires_grad=False)
            # zero grad : Sets gradients of all model parameters to zero.
            # 因为gradient是累积的accumulate，所以要让gradient朝着minimum的方向去走
            loss_overall.backward()
            self.optimizer_main.step()

            # for p in self.parameters_orth:
            #     p.requires_grad_(requires_grad=True)
            # for p in self.parameters_main:
            #     p.requires_grad_(requires_grad=False)
            #
            # loss_ortho.backward()
            # self.optimizer_ortho.step()

            # loss.backward()获得所有parameter的gradient
            # optimizer存了这些parameter的指针
            # step()根据这些parameter的gradient对parameter的值进行更新
            # https://www.zhihu.com/question/266160054

            # Logging.
            loss = {}
            loss['overall/loss_id'] = loss_overall_id.item()
            loss['main/loss_id'] = loss_main_id.item()
            loss['ortho/loss_id'] = loss_ortho_id.item()

            # =================================================================================== #
            #                                 4. Miscellaneous
            #                                 # 杂项
            # =================================================================================== #

            # Print out training information.
            if (i + 1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i + 1, self.num_iters)
                for tag in keys:
                    log += ", {}: {:.8f}".format(tag, loss[tag])
                print(log)

                if self.use_tensorboard:
                    for tag, value in loss.items():
                        self.writer.add_scalar(tag, value, i + 1)

            # Save model checkpoints.
            if (i + 1) % self.model_save_step == 0:
                # model_save_step : 1000
                model_path = os.path.join(self.model_save_dir, '{}.ckpt'.format(i + 1))
                torch.save({'model': self.model.state_dict(),
                            'optimizer_main': self.optimizer_main.state_dict()}, model_path)
                # 保存了 model + optimizer
                print('Saved model checkpoints at iteration {} into {}...'.format(i, self.model_save_dir))

                # Validation.
            if (i + 1) % self.sample_step == 0:
                self.model = self.model.eval()
                with torch.no_grad():
                    loss_overall_val_list = []
                    loss_main_val_list = []
                    loss_frame_main_val_list = []
                    loss_ortho_val_list = []
                    for val_sub in validation_pt:
                        # validation_pt: load 进 demo.pkl
                        emb_org_val = torch.from_numpy(val_sub[1][np.newaxis,:]).to(self.device)
                        # spk的one hot embedding
                        for k in range(2, 3):
                            x_real_pad, _ = pad_seq_to_2(val_sub[k][0][np.newaxis, :, :], 408)
                            len_org = torch.tensor([val_sub[k][2]]).to(self.device)
                            f0_org = np.pad(val_sub[k][1], (0, 408 - val_sub[k][2]), 'constant', constant_values=(0, 0))
                            f0_quantized = quantize_f0_numpy(f0_org)[0]
                            f0_onehot = f0_quantized[np.newaxis, :, :]
                            f0_org_val = torch.from_numpy(f0_onehot).to(self.device)
                            x_real_pad = torch.from_numpy(x_real_pad).to(self.device)
                            x_f0 = torch.cat((x_real_pad, f0_org_val), dim=-1)

                            mel_outputs, feature_predicts, ortho_inputs_integrals, mask_parts, invert_masks = self.model(x_f0, x_real_pad, emb_org_val, len_org, len_org, len_org)
                            loss_main_val = F.mse_loss(x_real_pad, mel_outputs, reduction='sum')
                            loss_frame_main_val = F.mse_loss(x_real_pad, mel_outputs, reduction='mean')
                            loss_ortho_id_L1_val = self.loss_o(ortho_inputs_integrals[0].cuda(),
                                                        feature_predicts[0].cuda() * invert_masks[0].cuda() + ortho_inputs_integrals[0].cuda() * mask_parts[0].cuda())

                            loss_ortho_id_BCE_val = self.loss_BCE(feature_predicts[1].cuda() * invert_masks[1].cuda() + ortho_inputs_integrals[1].cuda() * mask_parts[1].cuda()
                                                    ,ortho_inputs_integrals[1].cuda())
                                                    
                            loss_ortho_val = loss_ortho_id_L1_val + loss_ortho_id_BCE_val
                            ''''''
                            # w_ini = 0.5
                            # decay_rate = 0.1
                            # decay_steps = 1000
                            # # w_decay = w_ini * decay_rate ^ (i / decay_steps)
                            w_ini = 100
                            decay_rate = 0.999
                            decay_steps = 12500
                            w_decay = w_ini * math.pow(decay_rate, (i+1) / decay_steps)
                            
                            ''''''
                            loss_overall_id = self.w_main * w_decay * loss_main_val + self.w_ortho * loss_ortho_val
                            loss_overall_id = loss_overall_id / (w_ini*self.w_main)
                            # loss_overall_id = self.w_main * loss_main_val + self.w_ortho * loss_ortho_val
                            # 分别的 loss list
                            loss_overall_val_list.append(loss_overall_id.item())
                            loss_main_val_list.append(loss_main_val.item())
                            loss_ortho_val_list.append(loss_ortho_val.item())
                            loss_frame_main_val_list.append(loss_frame_main_val.item())
                val_overall_loss = np.mean(loss_overall_val_list)
                val_main_loss = np.mean(loss_main_val_list)
                val_ortho_loss = np.mean(loss_ortho_val_list)
                val_frame_main_loss = np.mean(loss_frame_main_val_list)
                print('Validation overall loss : {}, main loss: {}, ortho loss: {}, frame_main_loss:{}'.format(val_overall_loss, val_main_loss, val_ortho_loss,val_frame_main_loss))

                self.logger.log_validation(self.model, x_real_pad, mel_outputs, i + 1)
                if self.use_tensorboard:
                    self.writer.add_scalar('Validation_overall_loss', val_overall_loss, i + 1)
                    self.writer.add_scalar('Validation_main_loss', val_main_loss, i + 1)
                    self.writer.add_scalar('Validation_frame_main_loss', val_frame_main_loss, i + 1)
                    self.writer.add_scalar('Validation_ortho_loss', val_ortho_loss, i + 1)


            # plot test samples
            if (i + 1) % self.sample_step == 0:
                self.model = self.model.eval()
                with torch.no_grad():
                    for val_sub in validation_pt[:3]:
                        emb_org_val = torch.from_numpy(val_sub[1][np.newaxis,:]).to(self.device)
                        for k in range(2, 3):
                            x_real_pad, _ = pad_seq_to_2(val_sub[k][0][np.newaxis, :, :], 408)
                            len_org = torch.tensor([val_sub[k][2]]).to(self.device)
                            f0_org = np.pad(val_sub[k][1], (0, 408 - val_sub[k][2]), 'constant', constant_values=(0, 0))
                            f0_quantized = quantize_f0_numpy(f0_org)[0]
                            f0_onehot = f0_quantized[np.newaxis, :, :]
                            f0_org_val = torch.from_numpy(f0_onehot).to(self.device)
                            x_real_pad = torch.from_numpy(x_real_pad).to(self.device)
                            # 以下三行：把其中的某一个特征置0
                            x_f0 = torch.cat((x_real_pad, f0_org_val), dim=-1)
                            x_f0_F = torch.cat((x_real_pad, torch.zeros_like(f0_org_val)), dim=-1)
                            x_f0_C = torch.cat((torch.zeros_like(x_real_pad), f0_org_val), dim=-1)

                            x_identic_val,_ , _, _, _ = self.model(x_f0, x_real_pad, emb_org_val, len_org, len_org, len_org)
                            x_identic_woF,_ , _, _, _  = self.model(x_f0_F, x_real_pad, emb_org_val, len_org, len_org, len_org)
                            x_identic_woR,_ , _, _, _  = self.model(x_f0, torch.zeros_like(x_real_pad), emb_org_val, len_org, len_org, len_org)
                            x_identic_woC,_ , _, _, _  = self.model(x_f0_C, x_real_pad, emb_org_val, len_org, len_org, len_org)
                            x_identic_woU,_ , _, _, _  = self.model(x_f0, x_real_pad, torch.zeros_like(emb_org_val), len_org, len_org, len_org)

                            melsp_gd_pad = x_real_pad[0].cpu().numpy().T
                            # ground truth
                            melsp_out = x_identic_val[0].cpu().numpy().T
                            # 4部分完整
                            melsp_woF = x_identic_woF[0].cpu().numpy().T
                            # 没有 pitch
                            melsp_woR = x_identic_woR[0].cpu().numpy().T
                            # 没有 rhythm
                            melsp_woC = x_identic_woC[0].cpu().numpy().T
                            # 没有content
                            melsp_woU = x_identic_woU[0].cpu().numpy().T
                            # 没有U

                            min_value = np.min(np.hstack([melsp_gd_pad, melsp_out, melsp_woF, melsp_woR, melsp_woC,melsp_woU]))
                            max_value = np.max(np.hstack([melsp_gd_pad, melsp_out, melsp_woF, melsp_woR, melsp_woC,melsp_woU]))

                            fig, (ax1, ax2, ax3, ax4, ax5,ax6) = plt.subplots(6, 1, sharex=True)
                            im1 = ax1.imshow(melsp_gd_pad, aspect='auto', vmin=min_value, vmax=max_value)
                            im2 = ax2.imshow(melsp_out, aspect='auto', vmin=min_value, vmax=max_value)
                            im3 = ax3.imshow(melsp_woC, aspect='auto', vmin=min_value, vmax=max_value)
                            im4 = ax4.imshow(melsp_woR, aspect='auto', vmin=min_value, vmax=max_value)
                            im5 = ax5.imshow(melsp_woF, aspect='auto', vmin=min_value, vmax=max_value)
                            im6 = ax6.imshow(melsp_woU, aspect='auto', vmin=min_value, vmax=max_value)
                            plt.savefig(f'{self.sample_dir}/{i + 1}_{val_sub[0]}_{k}_{val_sub[2][3]}.png', dpi=150)
                            plt.close(fig)


def get_Rhythm_Encoder_parameters(pretrained_model_path,Mymodel):
    pretrained_model = torch.load(pretrained_model_path)
    Layers = pretrained_model['model'].items()
    My_Layers_dict = Mymodel.state_dict()
    state_dict = {k:v for k,v in Layers if k.split('_')[1][0]=='2'}# encoder_2. is rhythm encoder
    #get_rhythm_encoder
    My_Layers_dict.update(state_dict)
    Mymodel.load_state_dict(My_Layers_dict)

    #frozen rhythm

    # for param in Mymodel.named_parameters():
    #     if param[0] in state_dict: #frozen rhythm encoder
    #         param[1].requires_grad = False

    return Mymodel,state_dict
