# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

from typing import List

from .datatypes import (
    CheckpointQuantizationFormat,
    HardwareRequirements,
    ModelId,
    ModelSKU,
    SamplingParams,
    SamplingStrategy,
)


CONTEXT_LENGTH = 131072
VOCAB_SIZE = 128256


def all_model_skus() -> List[ModelSKU]:
    return base_models() + instruct_models()


def recommended_sampling_params() -> SamplingParams:
    return SamplingParams(
        strategy=SamplingStrategy.top_p,
        temperature=1.0,
        top_p=0.9,
    )


def base_models() -> List[ModelSKU]:
    return [
        ModelSKU(
            model_id=ModelId.llama3_1_8b,
            description_markdown="Llama 3.1 8b model",
            max_seq_length=CONTEXT_LENGTH,
            huggingface_id="meta-llama/Meta-Llama-3.1-8B",
            hardware_requirements=HardwareRequirements(
                gpu_count=1,
                memory_gb_per_gpu=20,
            ),
            recommended_sampling_params=recommended_sampling_params(),
            model_args={
                "dim": 4096,
                "n_layers": 32,
                "n_heads": 32,
                "n_kv_heads": 8,
                "vocab_size": VOCAB_SIZE,
                "ffn_dim_multiplier": 1.3,
                "multiple_of": 1024,
                "norm_eps": 1e-05,
                "rope_theta": 500000.0,
                "use_scaled_rope": True,
            },
        ),
        ModelSKU(
            model_id=ModelId.llama3_1_70b,
            description_markdown="Llama 3.1 70b model",
            huggingface_id="meta-llama/Meta-Llama-3.1-70B",
            max_seq_length=CONTEXT_LENGTH,
            hardware_requirements=HardwareRequirements(
                gpu_count=8,
                memory_gb_per_gpu=20,
            ),
            recommended_sampling_params=recommended_sampling_params(),
            model_args={
                "dim": 8192,
                "n_layers": 80,
                "n_heads": 64,
                "n_kv_heads": 8,
                "vocab_size": VOCAB_SIZE,
                "ffn_dim_multiplier": 1.3,
                "multiple_of": 4096,
                "norm_eps": 1e-05,
                "rope_theta": 500000.0,
                "use_scaled_rope": True,
            },
        ),
        ModelSKU(
            model_id=ModelId.llama3_1_405b,
            description_markdown="Llama 3.1 405b model (BF16 weights)",
            huggingface_id=None,
            max_seq_length=CONTEXT_LENGTH,
            hardware_requirements=HardwareRequirements(
                gpu_count=8,
                memory_gb_per_gpu=120,
            ),
            recommended_sampling_params=recommended_sampling_params(),
            model_args={
                "dim": 16384,
                "n_layers": 126,
                "n_heads": 128,
                "n_kv_heads": 8,
                "vocab_size": VOCAB_SIZE,
                "ffn_dim_multiplier": 1.2,
                "multiple_of": 4096,
                "norm_eps": 1e-05,
                "rope_theta": 500000.0,
                "use_scaled_rope": True,
            },
        ),
        ModelSKU(
            model_id=ModelId.llama3_1_405b,
            description_markdown="Llama 3.1 405b model (FP8 quantized)",
            max_seq_length=CONTEXT_LENGTH,
            huggingface_id="meta-llama/Meta-Llama-3.1-405B-FP8",
            hardware_requirements=HardwareRequirements(
                gpu_count=8,
                memory_gb_per_gpu=70,
            ),
            quantization_format=CheckpointQuantizationFormat.fp8_mixed,
            recommended_sampling_params=recommended_sampling_params(),
            model_args={
                "dim": 16384,
                "n_layers": 126,
                "n_heads": 128,
                "n_kv_heads": 8,
                "vocab_size": VOCAB_SIZE,
                "ffn_dim_multiplier": 1.2,
                "multiple_of": 4096,
                "norm_eps": 1e-05,
                "rope_theta": 500000.0,
                "use_scaled_rope": True,
            },
        ),
        ModelSKU(
            model_id=ModelId.llama3_1_405b,
            description_markdown="Llama 3.1 405b model (BF16 weights)",
            huggingface_id="meta-llama/Meta-Llama-3.1-405B",
            max_seq_length=CONTEXT_LENGTH,
            hardware_requirements=HardwareRequirements(
                gpu_count=16,
                memory_gb_per_gpu=70,
            ),
            recommended_sampling_params=recommended_sampling_params(),
            model_args={
                "dim": 16384,
                "n_layers": 126,
                "n_heads": 128,
                "n_kv_heads": 8,
                "vocab_size": VOCAB_SIZE,
                "ffn_dim_multiplier": 1.2,
                "multiple_of": 4096,
                "norm_eps": 1e-05,
                "rope_theta": 500000.0,
                "use_scaled_rope": True,
            },
        ),
    ]


def instruct_models() -> List[ModelSKU]:
    return [
        ModelSKU(
            model_id=ModelId.llama3_1_8b_instruct,
            description_markdown="Llama 3.1 8b instruct model",
            max_seq_length=CONTEXT_LENGTH,
            huggingface_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
            hardware_requirements=HardwareRequirements(
                gpu_count=1,
                memory_gb_per_gpu=20,
            ),
            recommended_sampling_params=recommended_sampling_params(),
            model_args={
                "dim": 4096,
                "n_layers": 32,
                "n_heads": 32,
                "n_kv_heads": 8,
                "vocab_size": VOCAB_SIZE,
                "ffn_dim_multiplier": 1.3,
                "multiple_of": 1024,
                "norm_eps": 1e-05,
                "rope_theta": 500000.0,
                "use_scaled_rope": True,
            },
        ),
        ModelSKU(
            model_id=ModelId.llama3_1_70b_instruct,
            description_markdown="Llama 3.1 70b instruct model",
            huggingface_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
            max_seq_length=CONTEXT_LENGTH,
            hardware_requirements=HardwareRequirements(
                gpu_count=8,
                memory_gb_per_gpu=20,
            ),
            recommended_sampling_params=recommended_sampling_params(),
            model_args={
                "dim": 8192,
                "n_layers": 80,
                "n_heads": 64,
                "n_kv_heads": 8,
                "vocab_size": VOCAB_SIZE,
                "ffn_dim_multiplier": 1.3,
                "multiple_of": 4096,
                "norm_eps": 1e-05,
                "rope_theta": 500000.0,
                "use_scaled_rope": True,
            },
        ),
        ModelSKU(
            model_id=ModelId.llama3_1_405b_instruct,
            description_markdown="Llama 3.1 405b instruct model (BF16 weights)",
            huggingface_id=None,
            max_seq_length=CONTEXT_LENGTH,
            hardware_requirements=HardwareRequirements(
                gpu_count=8,
                memory_gb_per_gpu=120,
            ),
            recommended_sampling_params=recommended_sampling_params(),
            model_args={
                "dim": 16384,
                "n_layers": 126,
                "n_heads": 128,
                "n_kv_heads": 8,
                "vocab_size": VOCAB_SIZE,
                "ffn_dim_multiplier": 1.2,
                "multiple_of": 4096,
                "norm_eps": 1e-05,
                "rope_theta": 500000.0,
                "use_scaled_rope": True,
            },
        ),
        ModelSKU(
            model_id=ModelId.llama3_1_405b_instruct,
            description_markdown="Llama 3.1 405b instruct model (FP8 quantized)",
            huggingface_id="meta-llama/Meta-Llama-3.1-405B-Instruct-FP8",
            max_seq_length=CONTEXT_LENGTH,
            hardware_requirements=HardwareRequirements(
                gpu_count=8,
                memory_gb_per_gpu=70,
            ),
            quantization_format=CheckpointQuantizationFormat.fp8_mixed,
            recommended_sampling_params=recommended_sampling_params(),
            model_args={
                "dim": 16384,
                "n_layers": 126,
                "n_heads": 128,
                "n_kv_heads": 8,
                "vocab_size": VOCAB_SIZE,
                "ffn_dim_multiplier": 1.2,
                "multiple_of": 4096,
                "norm_eps": 1e-05,
                "rope_theta": 500000.0,
                "use_scaled_rope": True,
            },
        ),
        ModelSKU(
            model_id=ModelId.llama3_1_405b_instruct,
            description_markdown="Llama 3.1 405b instruct model (BF16 weights)",
            huggingface_id="meta-llama/Meta-Llama-3.1-405B-Instruct",
            max_seq_length=CONTEXT_LENGTH,
            hardware_requirements=HardwareRequirements(
                gpu_count=16,
                memory_gb_per_gpu=70,
            ),
            recommended_sampling_params=recommended_sampling_params(),
            model_args={
                "dim": 16384,
                "n_layers": 126,
                "n_heads": 128,
                "n_kv_heads": 8,
                "vocab_size": VOCAB_SIZE,
                "ffn_dim_multiplier": 1.2,
                "multiple_of": 4096,
                "norm_eps": 1e-05,
                "rope_theta": 500000.0,
                "use_scaled_rope": True,
            },
        ),
    ]


def llama_meta_folder_path(model: ModelSKU) -> str:
    gpu = model.hardware_requirements.gpu_count
    is_fp8 = model.quantization_format == CheckpointQuantizationFormat.fp8_mixed

    if model.model_id == ModelId.llama3_1_405b:
        if gpu == 16:
            return "Meta-Llama-3.1-405B-MP16"
        elif is_fp8:
            return "Meta-Llama-3.1-405B"
        else:
            return "Meta-Llama-3.1-405B-MP8"
    elif model.model_id == ModelId.llama3_1_405b_instruct:
        if gpu == 16:
            return "Meta-Llama-3.1-405B-Instruct-MP16"
        elif is_fp8:
            return "Meta-Llama-3.1-405B-Instruct"
        else:
            return "Meta-Llama-3.1-405B-Instruct-MP8"

    path = model.huggingface_id.split("/")[-1]
    return path


# Sadness because Cloudfront rejects our HEAD requests to find Content-Length
def llama_meta_pth_size(model: ModelSKU) -> int:
    if model.model_id not in (ModelId.llama3_1_405b, ModelId.llama3_1_405b_instruct):
        return 0

    gpu = model.hardware_requirements.gpu_count
    is_fp8 = model.quantization_format == CheckpointQuantizationFormat.fp8_mixed
    if gpu == 16:
        return 51268302389
    elif is_fp8:
        return 60903742309
    else:
        return 101470976045
