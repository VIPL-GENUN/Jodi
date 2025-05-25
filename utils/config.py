import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Union


@dataclass
class BaseConfig:
    def get(self, attribute_name, default=None):
        return getattr(self, attribute_name, default)

    def pop(self, attribute_name, default=None):
        if hasattr(self, attribute_name):
            value = getattr(self, attribute_name)
            delattr(self, attribute_name)
            return value
        else:
            return default

    def __str__(self):
        return json.dumps(asdict(self), indent=4)


@dataclass
class DataConfig(BaseConfig):
    datasets: List[Any] = field(default_factory=list)


@dataclass
class ModelConfig(BaseConfig):
    model: str = "Jodi_600M_P1_D28"
    image_size: int = 1024
    mixed_precision: str = "bf16"  # ['fp16', 'fp32', 'bf16']
    fp32_attention: bool = True
    load_from: Optional[str] = None
    resume_from: Optional[Dict[str, Any]] = field(
        default_factory=lambda: {
            "checkpoint": None,
            "load_ema": False,
            "resume_lr_scheduler": True,
            "resume_optimizer": True,
        }
    )
    pe_interpolation: float = 1.0
    attn_type: str = "linear"
    autocast_linear_attn: bool = False
    mlp_acts: List[Optional[str]] = field(default_factory=lambda: ["silu", "silu", None])
    mlp_ratio: float = 2.5
    use_pe: bool = False
    qk_norm: bool = False
    class_dropout_prob: float = 0.1
    linear_head_dim: int = 32
    cross_norm: bool = False
    cfg_scale: int = 4
    guidance_type: str = "classifier-free"
    extra: Any = None


@dataclass
class AEConfig(BaseConfig):
    vae_type: str = "dc-ae"
    vae_pretrained: str = "mit-han-lab/dc-ae-f32c32-sana-1.0"
    weight_dtype: str = "bfloat16"
    scale_factor: float = 0.41407
    vae_latent_dim: int = 32
    vae_downsample_rate: int = 32
    sample_posterior: bool = True
    extra: Any = None


@dataclass
class TextEncoderConfig(BaseConfig):
    text_encoder_name: str = "gemma-2-2b-it"
    caption_channels: int = 2304
    y_norm: bool = True
    y_norm_scale_factor: float = 1.0
    model_max_length: int = 300
    chi_prompt: List[Optional[str]] = field(default_factory=lambda: [])
    extra: Any = None


@dataclass
class SchedulerConfig(BaseConfig):
    train_sampling_steps: int = 1000
    predict_v: bool = True
    noise_schedule: str = "linear_flow"
    flow_shift: float = 1.0
    # logit-normal timestep
    weighting_scheme: Optional[str] = "logit_normal"
    logit_mean: float = 0.0
    logit_std: float = 1.0
    extra: Any = None


@dataclass
class TrainingConfig(BaseConfig):
    num_workers: int = 4
    seed: int = 43
    train_batch_size: int = 32
    num_epochs: int = 100
    gradient_accumulation_steps: int = 1
    grad_checkpointing: bool = False
    gradient_clip: float = 1.0
    gc_step: int = 1
    optimizer: Dict[str, Any] = field(
        default_factory=lambda: {"eps": 1.0e-10, "lr": 0.0001, "type": "AdamW", "weight_decay": 0.03}
    )
    lr_schedule: str = "constant"
    lr_schedule_args: Dict[str, int] = field(default_factory=lambda: {"num_warmup_steps": 500})
    auto_lr: Dict[str, str] = field(default_factory=lambda: {"rule": "sqrt"})
    ema_rate: float = 0.9999
    eval_batch_size: int = 16
    use_flash_attn: bool = False
    eval_sampling_steps: int = 250
    lora_rank: int = 4
    log_interval: int = 50
    mask_type: str = "null"
    mask_loss_coef: float = 0.0
    load_mask_index: bool = False
    snr_loss: bool = False
    real_prompt_ratio: float = 1.0
    training_hours: float = 10000.0
    save_image_epochs: int = 1
    save_model_epochs: int = 1
    save_model_steps: int = 1000000
    visualize: bool = False
    null_embed_root: str = "output/pretrained_models/"
    valid_prompt_embed_root: str = "output/tmp_embed/"
    validation_prompts: List[str] = field(
        default_factory=lambda: [
            "cat",
            "child africa, double exposure",
            "colorful psychedelic mushrooms in a forest at night",
            "A hot air balloon in the shape of a heart. Grand Canyon",
            "a blue Porsche 356 parked in front of a yellow brick wall.",
            "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
            "an old rusted robot wearing pants and a jacket riding skis in a supermarket.",
            "A portrait of a human growing colorful flowers from her hair. Hyperrealistic oil painting. Intricate details.",
        ]
    )
    work_dir: str = "/cache/exps/"
    skip_step: int = 0
    loss_type: str = "huber"
    huber_c: float = 0.001
    num_ddim_timesteps: int = 50
    w_max: float = 15.0
    w_min: float = 3.0
    ema_decay: float = 0.95
    debug_nan: bool = False
    extra: Any = None


@dataclass
class JodiConfig(BaseConfig):
    data: DataConfig
    model: ModelConfig
    vae: AEConfig
    text_encoder: TextEncoderConfig
    scheduler: SchedulerConfig
    train: TrainingConfig
    work_dir: str = "output/"
    resume_from: Optional[str] = None
    resume_from_sana: bool = False
    load_from: Optional[str] = None
    debug: bool = False
    caching: bool = False
    report_to: str = "tensorboard"
    tracker_project_name: str = "t2i-evit-baseline"
    name: str = "baseline"
    loss_report_name: str = "loss"
    extend_to_new_domain: bool = False
    extend_to_new_domain_copy_id: Union[int, List] = 0


def model_init_config(config: JodiConfig, latent_size: int = 32):

    return {
        "input_size": latent_size,
        "pe_interpolation": config.model.pe_interpolation,
        "config": config,
        "model_max_length": config.text_encoder.model_max_length,
        "qk_norm": config.model.qk_norm,
        "caption_channels": config.text_encoder.caption_channels,
        "y_norm": config.text_encoder.y_norm,
        "attn_type": config.model.attn_type,
        "mlp_ratio": config.model.mlp_ratio,
        "mlp_acts": list(config.model.mlp_acts),
        "in_channels": config.vae.vae_latent_dim,
        "y_norm_scale_factor": config.text_encoder.y_norm_scale_factor,
        "use_pe": config.model.use_pe,
        "linear_head_dim": config.model.linear_head_dim,
        "cross_norm": config.model.cross_norm,
    }
