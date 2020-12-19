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

    # Decoder parameters
    n_frames_per_step=1,  # currently only 1 is supported
    decoder_rnn_dim=1024,
    prenet_dim=256,
    max_decoder_steps=1000,
    gate_threshold=0.5,
    p_attention_dropout=0.1,
    p_decoder_dropout=0.1,

    # Attention parameters
    attention_rnn_dim=1024,
    attention_dim=128,

    # Location Layer parameters
    attention_location_n_filters=32,
    attention_location_kernel_size=31,

    # Mel-post processing network parameters
    postnet_embedding_dim=512,
    postnet_kernel_size=5,
    postnet_n_convolutions=5,

    n_mel_channels=80,

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
