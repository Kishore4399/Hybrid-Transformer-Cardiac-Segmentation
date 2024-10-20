import ml_collections

def get_Hybrid3DTransformer_BTS_config():
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (1, 1, 1)})
    config.patches.grid = (1, 1, 1)
    config.hidden_size = 512
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 4096
    config.transformer.num_heads = 8
    config.transformer.num_layers = 4
    config.transformer.attention_dropout_rate = 0.1
    config.transformer.dropout_rate = 0.1
    config.patch_size = 1

    config.conv_first_channel = 128
    config.encoder_channels = (16, 32, 64, 128)
    config.down_factor = 2
    config.down_num = 3
    config.decoder_channels = (64, 32, 16)
    config.skip_channels = (64, 32, 16)
    config.n_dims = 3
    config.n_skip = 3
    return config

def get_HFTrans2_16_config():
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (2, 2, 2)})
    config.patches.grid = (2, 2, 2)
    config.hidden_size = 512
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 4096
    config.transformer.num_heads = 8
    config.transformer.num_layers = 4
    config.transformer.attention_dropout_rate = 0.1
    config.transformer.dropout_rate = 0.1
    config.patch_size = 1

    config.conv_first_channel = 128
    config.encoder_channels = (16, 32, 64, 128)
    config.down_factor = 2
    config.down_num = 3
    config.decoder_channels = (64, 32, 16)
    config.skip_channels = (192, 96, 48)
    config.n_dims = 3
    config.n_skip = 3
    return config
