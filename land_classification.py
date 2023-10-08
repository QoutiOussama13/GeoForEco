import os
#from mmcv.fileio.file_client import PetrelBackend
#dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
cudnn_benchmark = True
custom_imports = dict(imports=['geospatial_fm'])
num_frames = 3
img_size = 224
num_workers = 2

# model
# TO BE DEFINED BY USER: model path
pretrained_weights_path = 'Prithvi_100M.pt'
num_layers = 6
patch_size = 16
embed_dim = 768
num_heads = 8
tubelet_size = 1
max_epochs = 10
eval_epoch_interval = 5

loss_weights_multi = [
    0.386375, 0.661126, 0.548184, 0.640482, 0.876862, 0.925186, 3.249462
]
loss_func = dict(
    type='CrossEntropyLoss',
    use_sigmoid=False,
    class_weight=loss_weights_multi,
    avg_non_ignore=True)
output_embed_dim = embed_dim*num_frames


# TO BE DEFINED BY USER: Save directory
experiment = 'experiment 101'
project_dir = './Model/'
work_dir = os.path.join(project_dir, experiment)
save_path = work_dir


gpu_ids = range(0, 1)
dataset_type = 'GeospatialDataset'

# TO BE DEFINED BY USER: data directory


data_root = 'C:/Users/pc/Desktop/hls-foundation-os/DataTIFExtensio/'

splits = dict(
    train = 'train_chip_names.txt',
    val= 'validation_chip_names.txt',
    test= 'test_chip_names.txt',
)



img_norm_cfg = dict(
    means=[
        494.905781, 815.239594, 924.335066, 2968.881459 
    ],
    stds=[
        284.925432, 357.84876, 575.566823, 896.601013
    ])

bands = [0, 1, 2,4]

tile_size = 224
orig_nsize = 512
crop_size = (4, 212)

train_pipeline = [
    dict(type='LoadGeospatialImageFromFile', to_float32=True, channels_last=True), 
    dict(type='LoadGeospatialAnnotations', reduce_zero_label=True), 
    dict(type='RandomFlip', prob=0.5),
    dict(type='ToTensor', keys=['img', 'gt_semantic_seg']),
    # to channels first
    dict(type="TorchPermute", keys=["img"], order=(2, 0, 1)),
    dict(type='TorchNormalize', **img_norm_cfg),
    dict(type='TorchResize', new_size=crop_size, keys=['img']),
    dict(type='Reshape', keys=['img'], new_shape=(4, 1, 212, 512)),  # Update number of bands and frames
    dict(type='Reshape', keys=['gt_semantic_seg'], new_shape=(1, tile_size, tile_size)),
    dict(type='CastTensor', keys=['gt_semantic_seg'], new_type="torch.LongTensor"),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

test_pipeline = [
    dict(type='LoadGeospatialImageFromFile', to_float32=True, channels_last=True),
    dict(type='ToTensor', keys=['img']),
     # to channels first
    dict(type="TorchPermute", keys=["img"], order=(2, 0, 1)),
    dict(type='TorchNormalize', **img_norm_cfg),
    dict(type='Reshape', keys=['img'], new_shape=(len(bands), num_frames, -1, -1), look_up = {'2': 1, '3': 2}),
    dict(type='CastTensor', keys=['img'], new_type="torch.FloatTensor"),
    dict(type='CollectTestList', keys=['img'],
         meta_keys=['img_info', 'seg_fields', 'img_prefix', 'seg_prefix', 'filename', 'ori_filename', 'img',
                    'img_shape', 'ori_shape', 'pad_shape', 'scale_factor', 'img_norm_cfg']),
]

CLASSES = ('Snow', 
           'Thin cirrus', 
           'Vegetation', 
           'Non-vegetation', 
           'Water', 
           'Unclassified', 
           'Cloud shadows',)


dataset = 'GeospatialDataset'

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type=dataset,
        CLASSES=CLASSES,
        reduce_zero_label=True,
        data_root=data_root,
        img_dir='C:/Users/pc/Desktop/hls-foundation-os/DataTIFExtensio/train_file',
        ann_dir='C:/Users/pc/Desktop/hls-foundation-os/DataTIFExtensio/train_file',
        pipeline=train_pipeline,
        img_suffix='_merged.tif',
        seg_map_suffix='.mask.tif',
        split=splits['train']),
    val=dict(
        type=dataset,
        CLASSES=CLASSES,
        reduce_zero_label=True,
        data_root=data_root,
        img_dir='C:/Users/pc/Desktop/hls-foundation-os/DataTIFExtensio/validation_file',
        ann_dir='C:/Users/pc/Desktop/hls-foundation-os/DataTIFExtensio/validation_file',
        pipeline=test_pipeline,
        img_suffix='_merged.tif',
        seg_map_suffix='.mask.tif',
        split=splits['val']
    ),
    test=dict(
        type=dataset,
        CLASSES=CLASSES,
        reduce_zero_label=True,
        data_root=data_root,
        img_dir='C:/Users/pc/Desktop/hls-foundation-os/DataTIFExtensio/test_file',
        ann_dir='C:/Users/pc/Desktop/hls-foundation-os/DataTIFExtensio/test_file',
        pipeline=test_pipeline,
        img_suffix='_merged.tif',
        seg_map_suffix='.mask.tif',
        split=splits['val']
    ))

optimizer = dict(
    type='Adam', lr=1.5e-05, betas=(0.9, 0.999), weight_decay=0.05)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-06,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)
log_config = dict(
    interval=10,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])

checkpoint_config = dict(
    by_epoch=True,
    interval=100,
    out_dir=save_path)

evaluation = dict(interval=eval_epoch_interval, metric='mIoU', pre_eval=True, save_best='mIoU', by_epoch=True)
reduce_train_set = dict(reduce_train_set=False)
reduce_factor = dict(reduce_factor=1)
runner = dict(type='EpochBasedRunner', max_epochs=max_epochs)
workflow = [('train', 1)]
norm_cfg = dict(type='BN', requires_grad=True)

model = dict(
    type='TemporalEncoderDecoder',
    frozen_backbone=False,
    backbone=dict(
        type='TemporalViTEncoder',
        pretrained=pretrained_weights_path,
        img_size=img_size,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=1,
        in_chans=len(bands),
        embed_dim=embed_dim,
        depth=6,
        num_heads=num_heads,
        mlp_ratio=4.0,
        norm_pix_loss=False),
    neck=dict(
        type='ConvTransformerTokensToEmbeddingNeck',
        embed_dim=embed_dim*num_frames,
        output_embed_dim=output_embed_dim,
        drop_cls_token=True,
        Hp=14,
        Wp=14),
    decode_head=dict(
        num_classes=len(CLASSES),
        in_channels=output_embed_dim,
        type='FCNHead',
        in_index=-1,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=loss_func),
    auxiliary_head=dict(
        num_classes=len(CLASSES),
        in_channels=output_embed_dim,
        type='FCNHead',
        in_index=-1,
        channels=256,
        num_convs=2,
        concat_input=False,
        dropout_ratio=0.1,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=loss_func),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', stride=(int(tile_size/2), int(tile_size/2)), crop_size=(tile_size, tile_size)))
auto_resume = False
