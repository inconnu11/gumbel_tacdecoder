from tfcompat.hparam import HParams

# NOTE: If you want full control for model architecture. please take a look
# at the code and change whatever you want. Some hyper parameters are hardcoded.
# Note: This is 1.5* dim_necks
# Default hyperparameters:
hparams = HParams(
    loss_reconstruction_w=9,
    loss_disentanglement_w=1,
    # threshold=4e-4,
    # model   
    freq = 8,          # content codes降采样
    dim_neck = 8,      # (blstm dim)content_encoder
    freq_2 = 8,        # rhythm codes降采样 rhythm_encoder(encoder_2,encoder_t)
    dim_neck_2 = 1,    # (blstm dim)rhythm_encoder(encoder_2,encoder_t)
    freq_3 = 8,        # pitch codes降采样
    dim_neck_3 = 32,   # (blstm dim)pitch_encoder
    
    dim_enc = 512,     # (conv dim) content
    dim_enc_2 = 128,   # (conv dim) rhythm (encoder_2,encoder_t)
    dim_enc_3 = 256,   # (conv dim) pitch

    # x(mel) : 80维
    dim_freq = 80,
    dim_spk_emb = 107, # 82 at first
    embedding_spk = 64,
    # f0 ：257维
    dim_f0 = 257,
    dim_dec = 512,
    dim_ortho=1024,
    layer_norm_eps=1e-12,
    len_raw = 128,
    chs_grp = 16,
    
    # interp
    # 为了random resampling，先分段segment
    # 每个段segment长度：19帧～32帧
    min_len_seg = 19,
    max_len_seg = 32,
    # min_len_seq = 64,
    min_len_seq = 32,
    # max_len_seq = 128,
    max_len_seq = 48,
    max_len_pad = 408,#192 at first,
    
    # data loader
    root_dir = 'assets/spmel',
    feat_dir = 'assets/raptf0',
    batch_size = 16,
    mode = 'train',
    shuffle = True,
    num_workers = 0,
    samplier = 8,
    #MBV
    enc_mbv_size = 7,
    
)


def hparams_debug_string():
    values = hparams.values()
    hp = ['  %s: %s' % (name, values[name]) for name in values]
    return 'Hyperparameters:\n' + '\n'.join(hp)
