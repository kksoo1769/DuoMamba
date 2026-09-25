# --------------------------------------------------------
# DuoMamba — ACCV 2026
# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import ast
import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = ['']

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Batch size for a single GPU, could be overwritten by command line argument
_C.DATA.BATCH_SIZE = 128
# Path to dataset, could be overwritten by command line argument
_C.DATA.DATA_PATH = ''
# Dataset name
_C.DATA.DATASET = 'imagenet'
# Input image size
_C.DATA.IMG_SIZE = 224
# Interpolation to resize image (random, bilinear, bicubic)
_C.DATA.INTERPOLATION = 'bicubic'
# Use zipped dataset instead of folder dataset
# could be overwritten by command line argument
_C.DATA.ZIP_MODE = False
# Cache Data in Memory, could be overwritten by command line argument
_C.DATA.CACHE_MODE = 'part'
# Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
_C.DATA.PIN_MEMORY = True
# Number of data loading threads
_C.DATA.NUM_WORKERS = 10

#persistent workers
_C.DATA.PERSISTENT_WORKERS = True
# Subset ratio for training data (1.0 = full dataset, 0.1 = 10% of dataset)
_C.DATA.SUBSET_RATIO = 1.0
# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model type
_C.MODEL.TYPE = 'duomamba'
# Model name
_C.MODEL.NAME = 'duomamba_tiny'
# Pretrained weight from checkpoint, could be imagenet22k pretrained weight
# could be overwritten by command line argument
_C.MODEL.PRETRAINED = ''
# Checkpoint to resume, could be overwritten by command line argument
_C.MODEL.RESUME = ''
# Number of classes, overwritten in data preparation
_C.MODEL.NUM_CLASSES = 1000
# Dropout rate
_C.MODEL.DROP_RATE = 0.0
# Drop path rate
_C.MODEL.DROP_PATH_RATE = 0.1
# Label Smoothing
_C.MODEL.LABEL_SMOOTHING = 0.1
#for ddp platform
_C.MODEL.DDP = 'torch'

# DuoMamba parameters
_C.MODEL.DUOMAMBA = CN()
_C.MODEL.DUOMAMBA.PATCH_SIZE = 4
_C.MODEL.DUOMAMBA.IN_CHANS = 3
_C.MODEL.DUOMAMBA.NUM_HEADS = [2, 4, 8, 16]
_C.MODEL.DUOMAMBA.DEPTHS = [2, 4, 8, 4]
_C.MODEL.DUOMAMBA.EMBED_DIM = 64
_C.MODEL.DUOMAMBA.MLP_RATIO = 4.0
_C.MODEL.DUOMAMBA.DROP_RATE = 0.0
_C.MODEL.DUOMAMBA.DROP_PATH_RATE = 0.2
_C.MODEL.DUOMAMBA.EXPAND = 2
_C.MODEL.DUOMAMBA.NGROUPS = 1
_C.MODEL.DUOMAMBA.CHUNK_SIZE = 256
_C.MODEL.DUOMAMBA.MIXER_TYPES = ['DuoScanMixer', 'DuoScanMixer', 'DuoScanMixer', 'MHSA']
_C.MODEL.DUOMAMBA.D_STATE = 16
_C.MODEL.DUOMAMBA.GATING = True
_C.MODEL.DUOMAMBA.LPU = False
_C.MODEL.DUOMAMBA.STEM_VER = 2
_C.MODEL.DUOMAMBA.DOWNSAMPLER_VER = 2
_C.MODEL.DUOMAMBA.NORM_TYPE = 'ln'
_C.MODEL.DUOMAMBA.NORM_BEFORE_GATE = True
# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 300
_C.TRAIN.WARMUP_EPOCHS = 20
_C.TRAIN.WEIGHT_DECAY = 0.05
_C.TRAIN.BASE_LR = 5e-4
_C.TRAIN.WARMUP_LR = 5e-7
_C.TRAIN.MIN_LR = 5e-6
# Clip gradient norm
_C.TRAIN.CLIP_GRAD = 5.0
# Auto resume from latest checkpoint
_C.TRAIN.AUTO_RESUME = True
# Gradient accumulation steps
# could be overwritten by command line argument
_C.TRAIN.ACCUMULATION_STEPS = 1
# Whether to use gradient checkpointing to save memory
# could be overwritten by command line argument
_C.TRAIN.USE_CHECKPOINT = False
# Mesa distillation/trick settings
_C.TRAIN.MESA = -1.0

# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'cosine'
# Epoch interval to decay LR, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30
# LR decay rate, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1
# warmup_prefix used in CosineLRScheduler
_C.TRAIN.LR_SCHEDULER.WARMUP_PREFIX = True
# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adamw'
# Optimizer Epsilon
_C.TRAIN.OPTIMIZER.EPS = 1e-8
# Optimizer Betas
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
# SGD momentum
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9

# -----------------------------------------------------------------------------
# Augmentation settings
# -----------------------------------------------------------------------------
_C.AUG = CN()
# Color jitter factor
_C.AUG.COLOR_JITTER = 0.4
# Use AutoAugment policy. "v0" or "original"
_C.AUG.AUTO_AUGMENT = 'rand-m9-mstd0.5-inc1'
# Random erase prob
_C.AUG.REPROB = 0.25
# Random erase mode
_C.AUG.REMODE = 'pixel'
# Random erase count
_C.AUG.RECOUNT = 1
# Mixup alpha, mixup enabled if > 0
_C.AUG.MIXUP = 0.8
# Cutmix alpha, cutmix enabled if > 0
_C.AUG.CUTMIX = 1.0
# Cutmix min/max ratio, overrides alpha and enables cutmix if set
_C.AUG.CUTMIX_MINMAX = None
# Probability of performing mixup or cutmix when either/both is enabled
_C.AUG.MIXUP_PROB = 1.0
# Probability of switching to cutmix when both mixup and cutmix enabled
_C.AUG.MIXUP_SWITCH_PROB = 0.5
# How to apply mixup/cutmix params. Per "batch", "pair", or "elem"
_C.AUG.MIXUP_MODE = 'batch'

# -----------------------------------------------------------------------------
# Testing settings
# -----------------------------------------------------------------------------
_C.TEST = CN()
# Whether to use center crop when testing
_C.TEST.CROP = True
# Whether to use SequentialSampler as validation sampler
_C.TEST.SEQUENTIAL = False
_C.TEST.SHUFFLE = False

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
# Enable Pytorch automatic mixed precision (amp).
_C.AMP_ENABLE = True
# AMP dtype: 'auto' (use bf16 if supported, else fp16), 'bf16', or 'fp16'
_C.AMP_DTYPE = 'auto'
# Path to output folder, overwritten by command line argument
_C.OUTPUT = ''
# Tag of experiment, overwritten by command line argument
_C.TAG = 'default'
# Frequency to save checkpoint
_C.SAVE_FREQ = 1
# Frequency to logging info
_C.PRINT_FREQ = 10
# Fixed random seed
_C.SEED = 0
# Perform evaluation only, overwritten by command line argument
_C.EVAL_MODE = False
# Test throughput only, overwritten by command line argument
_C.THROUGHPUT_MODE = False
# Test traincost only, overwritten by command line argument
_C.TRAINCOST_MODE = False


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader) or {}

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    # A parent file freezes the shared node; the child may change chunk-size type.
    config.defrost()
    _maybe_adjust_chunk_size_type(config, yaml_cfg)
    config.merge_from_other_cfg(CN(yaml_cfg))
    config.freeze()


def _maybe_adjust_chunk_size_type(config, yaml_cfg):
    chunk_size = (
        yaml_cfg.get('MODEL', {})
        .get('DUOMAMBA', {})
        .get('CHUNK_SIZE', None)
    )
    if chunk_size is None:
        return

    if isinstance(chunk_size, (list, tuple)):
        if not isinstance(config.MODEL.DUOMAMBA.CHUNK_SIZE, (list, tuple)):
            config.MODEL.DUOMAMBA.CHUNK_SIZE = []
        return

    if isinstance(config.MODEL.DUOMAMBA.CHUNK_SIZE, (list, tuple)):
        current = config.MODEL.DUOMAMBA.CHUNK_SIZE
        if current and isinstance(current[0], int):
            config.MODEL.DUOMAMBA.CHUNK_SIZE = int(current[0])
        else:
            config.MODEL.DUOMAMBA.CHUNK_SIZE = 256


def _maybe_adjust_chunk_size_type_from_opts(config, opts):
    if not opts:
        return
    if len(opts) % 2 != 0:
        return

    for key, value in zip(opts[0::2], opts[1::2]):
        if key != 'MODEL.DUOMAMBA.CHUNK_SIZE':
            continue

        try:
            parsed_value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            parsed_value = value

        if isinstance(parsed_value, (list, tuple)):
            if not isinstance(config.MODEL.DUOMAMBA.CHUNK_SIZE, (list, tuple)):
                config.MODEL.DUOMAMBA.CHUNK_SIZE = []
        else:
            if isinstance(config.MODEL.DUOMAMBA.CHUNK_SIZE, (list, tuple)):
                current = config.MODEL.DUOMAMBA.CHUNK_SIZE
                if current and isinstance(current[0], int):
                    config.MODEL.DUOMAMBA.CHUNK_SIZE = int(current[0])
                else:
                    config.MODEL.DUOMAMBA.CHUNK_SIZE = 256


def update_config(config, args):
    _update_config_from_file(config, args.cfg)

    config.defrost()
    if args.opts:
        if len(args.opts) % 2:
            raise ValueError('Configuration overrides must be KEY VALUE pairs.')
        for index in range(0, len(args.opts), 2):
            override = args.opts[index:index + 2]
            _maybe_adjust_chunk_size_type_from_opts(config, override)
            config.merge_from_list(override)

    def _check_args(name):
        if hasattr(args, name) and eval(f'args.{name}'):
            return True
        return False

    # merge from specific arguments
    if _check_args('batch_size'):
        config.DATA.BATCH_SIZE = args.batch_size
    if _check_args('data_path'):
        config.DATA.DATA_PATH = args.data_path
    if _check_args('zip'):
        config.DATA.ZIP_MODE = True
    if _check_args('cache_mode'):
        config.DATA.CACHE_MODE = args.cache_mode
    if _check_args('pretrained'):
        config.MODEL.PRETRAINED = args.pretrained
    if _check_args('resume'):
        config.MODEL.RESUME = args.resume
    if _check_args('accumulation_steps'):
        config.TRAIN.ACCUMULATION_STEPS = args.accumulation_steps
    if _check_args('use_checkpoint'):
        config.TRAIN.USE_CHECKPOINT = True
    if _check_args('disable_amp'):
        config.AMP_ENABLE = False
    if _check_args('output'):
        config.OUTPUT = args.output
    if _check_args('tag'):
        config.TAG = args.tag
    if _check_args('eval'):
        config.EVAL_MODE = True
    if _check_args('throughput'):
        config.THROUGHPUT_MODE = True
    if _check_args('traincost'):
        config.TRAINCOST_MODE = True
    if _check_args('enable_persistance'):
        config.DATA.PERSISTENT_WORKERS = True
    # MESA: allow YAML to set the weight; CLI override only if explicitly provided
    if getattr(args, 'mesa', None) is not None:
        config.TRAIN.MESA = args.mesa

    ## Overwrite optimizer if not None, currently we use it for [fused_adam, fused_lamb]
    if _check_args('optim'):
        config.TRAIN.OPTIMIZER.NAME = args.optim

    if _check_args('ddp'):
        config.MODEL.DDP = args.ddp
    # output folder
    config.OUTPUT = os.path.join(config.OUTPUT, config.MODEL.NAME, config.TAG)

    config.freeze()


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)

    return config
