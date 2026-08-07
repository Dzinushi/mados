# model settings
model = dict(
        type='EncoderDecoder',
        backbone=dict(
                type="MSCAN",
                in_chans=11,
                embed_dims=[32, 64, 160, 256],
                mlp_ratios=[8, 8, 4, 4],
                drop_rate=0.0,
                drop_path_rate=0.1,
                depths=[3, 3, 5, 2],
                norm_cfg=dict(type="SyncBN", requires_grad=True),
                layer_norm=dict(type="AdaptiveLayerNorm", l1=1.0, l2=1.0),
                act_layer=dict(type="TReLU", r1=1.0, r2=0.01)),
                # act_layer=dict(type="GELU")), # original GELU without params
        decode_head=dict(
                type='LightHamHead',
                in_channels=[32, 64, 160, 256],
                in_index=[0, 1, 2, 3],
                channels=256,
                ham_channels=256,
                ham_kwargs=dict(MD_R=16),
                dropout_ratio=0.1,
                num_classes=15,
                norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                align_corners=False,
                act_cfg=dict(type="TReLU", r1=1.0, r2=0.01)) # remove this param for original
)
