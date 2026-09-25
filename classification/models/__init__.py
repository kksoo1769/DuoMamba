from .duomamba import BackboneDuoMamba, DuoMamba, DuoMambaBlock, DuoScanMixer


def build_duomamba_model(config, is_pretrain=False):
    model_type = config.MODEL.TYPE
    if model_type == "duomamba":
        model_cfg = config.MODEL.DUOMAMBA
        model = DuoMamba(
            num_classes=config.MODEL.NUM_CLASSES,
            patch_size=model_cfg.PATCH_SIZE,
            in_chans=model_cfg.IN_CHANS,
            embed_dim=model_cfg.EMBED_DIM,
            depths=model_cfg.DEPTHS,
            num_heads=model_cfg.NUM_HEADS,
            mlp_ratio=model_cfg.MLP_RATIO,
            drop_rate=model_cfg.DROP_RATE,
            drop_path_rate=model_cfg.DROP_PATH_RATE,
            expand=model_cfg.EXPAND,
            ngroups=model_cfg.NGROUPS,
            chunk_size=model_cfg.CHUNK_SIZE,
            mixer_types=model_cfg.MIXER_TYPES,
            d_state=model_cfg.D_STATE,
            gating=model_cfg.GATING,
            lpu=model_cfg.LPU,
            stem_ver=model_cfg.STEM_VER,
            downsampler_ver=model_cfg.DOWNSAMPLER_VER,
            norm_type=model_cfg.NORM_TYPE,
            norm_before_gate=model_cfg.NORM_BEFORE_GATE,
        )
        return model
    raise ValueError(f"Unsupported model type: {model_type}")


def build_model(config, is_pretrain=False):
    return build_duomamba_model(config, is_pretrain)
