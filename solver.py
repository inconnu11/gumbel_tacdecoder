from model import Generator_3 as Generator
from model import InterpLnr
from model import OrthoDisen
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime
import pickle

from utils import pad_seq_to_2, quantize_f0_torch, quantize_f0_numpy

# use demo data for simplicity
# make your own validation set as needed
validation_pt = pickle.load(open('assets/demo.pkl', "rb"))
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
        self.g_lr = config.g_lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters

        self.beta1_ortho = config.beta1_ortho
        self.beta2_ortho = config.beta2_ortho
        self.ortho_lr = config.ortho_lr

        # Miscellaneous.
        self.use_tensorboard = config.use_tensorboard
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:{}'.format(config.device_id) if self.use_cuda else 'cpu')

        # Directories.
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir

        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step

        # Build the model and tensorboard.
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

    def grouped_parameters(self):
        # 返回模型中所有的参数
        params_all = [p for p in self.G.parameters()]
        params_all.extend([p for p in self.ortho.parameters()])
        return params_all
            
    def build_model(self):        
        self.G = Generator(self.hparams)
        
        self.Interp = InterpLnr(self.hparams)
            
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.print_network(self.G, 'G')
        
        self.G.to(self.device)
        self.Interp.to(self.device)

        # 正交解耦模块的训练参数设置
        self.ortho = OrthoDisen(self.hparams)
        params_all = self.grouped_parameters()
        self.ortho_optimizer = torch.optim.Adam(params_all, self.ortho_lr, [self.beta1_ortho, self.beta2_ortho])
        # betas:Tuple[float, float] 用于计算梯度以及梯度平方的运行平均值的系数
        # beta1： 一阶钜估计的指数衰减率
        # beta2：二阶矩估计的指数衰减率
        self.print_network(self.ortho, 'OrthoDisen')
        self.ortho.to(self.device)


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

        self.G.load_state_dict(checkpoint_dict['model'])
        self.g_optimizer.load_state_dict(checkpoint_dict['optimizer_G'])
        self.g_lr = self.g_optimizer.param_groups[0]['lr']

        self.ortho.load_state_dict(checkpoint_dict['model'])
        self.ortho_optimizer.load_state_dict(checkpoint_dict['optimizer_D'])
        self.g_lr = self.ortho_optimizer.param_groups[0]['lr']

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(self.log_dir)

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.ortho_optimizer.zero_grad()

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
            self.print_optimizer(self.g_optimizer, 'G_optimizer')
            self.print_optimizer(self.ortho_optimizer, 'OrthoDisen_optimizer')
                        
        # Learning rate cache for decaying.
        g_lr = self.g_lr
        print('Current learning rates, g_lr: {}.'.format(g_lr))
        ortho_lr = self.ortho_lr
        print('Current learning rates, ortho_lr: {}.'.format(ortho_lr))
        
        # Print logs in specified order
        keys = ['G/loss_id']
        keys_ortho = ['ortho/loss_dis']
            
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
            
            self.model = self.model.train()
                        
            # Identity mapping loss
            x_f0 = torch.cat((x_real_org, f0_org), dim=-1)
            x_f0_intrp = self.Interp(x_f0, len_org) 
            f0_org_intrp = quantize_f0_torch(x_f0_intrp[:,:,-1])[0]
            x_f0_intrp_org = torch.cat((x_f0_intrp[:,:,:-1], f0_org_intrp), dim=-1)
            
            # x_identic = self.G(x_f0_intrp_org, x_real_org, emb_org)
            x_identic = self.model(x_f0_intrp_org, x_real_org, emb_org)
            # identity:一致性，所以输入的都是original spk的
            g_loss_id = F.mse_loss(x_real_org, x_identic, reduction='mean')
            # x_real_org：ground truth
            # x_identic：预测的predicted
            # mse loss
           
            # Backward and optimize.
            g_loss = g_loss_id
            self.reset_grad()
            # zero grad : Sets gradients of all model parameters to zero.
            # 因为gradient是累积的accumulate，所以要让gradient朝着minimum的方向去走
            g_loss.backward()
            # loss.backward()获得所有parameter的gradient
            self.g_optimizer.step()
            # optimizer存了这些parameter的指针
            # step()根据这些parameter的gradient对parameter的值进行更新
            # https://www.zhihu.com/question/266160054

            # =================================================================================== #
            #                               3. Train the orthogonal                                #
            # =================================================================================== #

            self.ortho = self.ortho.train()
            # TODO : 根据张景轩的 + bert loss 改 3. train the orthogonal

            # Identity mapping loss
            x_f0 = torch.cat((x_real_org, f0_org), dim=-1)
            x_f0_intrp = self.Interp(x_f0, len_org)
            f0_org_intrp = quantize_f0_torch(x_f0_intrp[:, :, -1])[0]
            x_f0_intrp_org = torch.cat((x_f0_intrp[:, :, :-1], f0_org_intrp), dim=-1)

            x_identic = self.G(x_f0_intrp_org, x_real_org, emb_org)
            # identity:一致性，所以输入的都是original spk的
            ortho_loss_id = F.mse_loss(x_real_org, x_identic, reduction='mean')
            # mse loss

            # Backward and optimize.
            ortho_loss = ortho_loss_id
            self.reset_grad()
            # zero grad : Sets gradients of all model parameters to zero.
            # 因为gradient是累积的accumulate，所以要让gradient朝着minimum的方向去走
            g_loss.backward()
            # loss.backward()获得所有parameter的gradient
            self.ortho_optimizer.step()
            # optimizer存了这些parameter的指针
            # step()根据这些parameter的gradient对parameter的值进行更新
            # https://www.zhihu.com/question/266160054





            # Logging.
            loss = {}
            loss['G/loss_id'] = g_loss_id.item()

            # =================================================================================== #
            #                                 4. Miscellaneous
            #                                 # 杂项
            # =================================================================================== #

            # Print out training information.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                for tag in keys:
                    log += ", {}: {:.8f}".format(tag, loss[tag])
                print(log)

                if self.use_tensorboard:
                    for tag, value in loss.items():
                        self.writer.add_scalar(tag, value, i+1)

            # Save model checkpoints.
            if (i+1) % self.model_save_step == 0:
                # model_save_step : 1000
                G_path = os.path.join(self.model_save_dir, '{}.ckpt'.format(i+1))
                torch.save({'model': self.G.state_dict(),
                            'optimizer_G': self.g_optimizer.state_dict(),
                            'optimizer_Ortho': self.ortho_optimizer.state_dict()}, G_path)
                # 保存了 model + optimizer
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))            

            # Validation.
            if (i+1) % self.sample_step == 0:
                self.G = self.G.eval()
                with torch.no_grad():
                    loss_val = []
                    for val_sub in validation_pt:
                        # validation_pt: load 进 demo.pkl
                        emb_org_val = torch.from_numpy(val_sub[1]).to(self.device)
                        # spk的one hot embedding
                        for k in range(2, 3):
                            x_real_pad, _ = pad_seq_to_2(val_sub[k][0][np.newaxis,:,:], 192)
                            len_org = torch.tensor([val_sub[k][2]]).to(self.device) 
                            f0_org = np.pad(val_sub[k][1], (0, 408-val_sub[k][2]), 'constant', constant_values=(0, 0))
                            f0_quantized = quantize_f0_numpy(f0_org)[0]
                            f0_onehot = f0_quantized[np.newaxis, :, :]
                            f0_org_val = torch.from_numpy(f0_onehot).to(self.device) 
                            x_real_pad = torch.from_numpy(x_real_pad).to(self.device) 
                            x_f0 = torch.cat((x_real_pad, f0_org_val), dim=-1)
                            x_identic_val = self.G(x_f0, x_real_pad, emb_org_val)
                            g_loss_val = F.mse_loss(x_real_pad, x_identic_val, reduction='sum')
                            loss_val.append(g_loss_val.item())
                val_loss = np.mean(loss_val) 
                print('Validation loss: {}'.format(val_loss))
                if self.use_tensorboard:
                    self.writer.add_scalar('Validation_loss', val_loss, i+1)

            # plot test samples
            if (i+1) % self.sample_step == 0:
                self.G = self.G.eval()
                with torch.no_grad():
                    for val_sub in validation_pt:
                        emb_org_val = torch.from_numpy(val_sub[1]).to(self.device)         
                        for k in range(2, 3):
                            x_real_pad, _ = pad_seq_to_2(val_sub[k][0][np.newaxis,:,:], 408)
                            len_org = torch.tensor([val_sub[k][2]]).to(self.device) 
                            f0_org = np.pad(val_sub[k][1], (0, 408-val_sub[k][2]), 'constant', constant_values=(0, 0))
                            f0_quantized = quantize_f0_numpy(f0_org)[0]
                            f0_onehot = f0_quantized[np.newaxis, :, :]
                            f0_org_val = torch.from_numpy(f0_onehot).to(self.device) 
                            x_real_pad = torch.from_numpy(x_real_pad).to(self.device)
                            # 以下三行：把其中的某一个特征置0
                            x_f0 = torch.cat((x_real_pad, f0_org_val), dim=-1)
                            x_f0_F = torch.cat((x_real_pad, torch.zeros_like(f0_org_val)), dim=-1)
                            x_f0_C = torch.cat((torch.zeros_like(x_real_pad), f0_org_val), dim=-1)
                            
                            x_identic_val = self.G(x_f0, x_real_pad, emb_org_val)
                            x_identic_woF = self.G(x_f0_F, x_real_pad, emb_org_val)
                            x_identic_woR = self.G(x_f0, torch.zeros_like(x_real_pad), emb_org_val)
                            # woR：without rhythm
                            x_identic_woC = self.G(x_f0_C, x_real_pad, emb_org_val)
                            
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
                            
                            min_value = np.min(np.hstack([melsp_gd_pad, melsp_out, melsp_woF, melsp_woR, melsp_woC]))
                            max_value = np.max(np.hstack([melsp_gd_pad, melsp_out, melsp_woF, melsp_woR, melsp_woC]))
                            
                            fig, (ax1,ax2,ax3,ax4,ax5) = plt.subplots(5, 1, sharex=True)
                            im1 = ax1.imshow(melsp_gd_pad, aspect='auto', vmin=min_value, vmax=max_value)
                            im2 = ax2.imshow(melsp_out, aspect='auto', vmin=min_value, vmax=max_value)
                            im3 = ax3.imshow(melsp_woC, aspect='auto', vmin=min_value, vmax=max_value)
                            im4 = ax4.imshow(melsp_woR, aspect='auto', vmin=min_value, vmax=max_value)
                            im5 = ax5.imshow(melsp_woF, aspect='auto', vmin=min_value, vmax=max_value)
                            plt.savefig(f'{self.sample_dir}/{i+1}_{val_sub[0]}_{k}.png', dpi=150)
                            plt.close(fig)
