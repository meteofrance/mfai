

from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from mfai.pytorch.models.llms.gpt2 import GPT2, GPT2Settings
from mfai.tensorflow import download_and_load_gpt2


Gpt2SizesType = Literal["124M", "355M", "774M", "1558M"]
GPT2_SIZES: tuple[Gpt2SizesType] = ("124M", "355M", "774M", "1558M")


def load_gpt2_from_dict(gpt2: GPT2, params: dict):
    """
    Loads weights into self using a dict
    likely coming from a tensorflow or other framework
    training. Use this to finetune from the official weights.
    """

    # we allow context length longer than official implementation
    # extra parameters are just normally initialised and not loaded
    # from supplied weights

    if gpt2.pos_emb.weight.shape[0] > len(params["wpe"]):
        gpt2.pos_emb.weight = torch.nn.Parameter(
            gpt2.pos_emb.weight.index_put(
                (torch.LongTensor(range(len(params["wpe"]))),),
                torch.tensor(params["wpe"]),
            )
        )
    else:
        gpt2.pos_emb.weight = assign(gpt2.pos_emb.weight, params["wpe"])

    # we allow for adding special tokens
    if gpt2.tok_emb.weight.shape[0] > len(params["wte"]):
        gpt2.tok_emb.weight = torch.nn.Parameter(
            gpt2.tok_emb.weight.index_put(
                (torch.LongTensor(range(len(params["wte"]))),),
                torch.tensor(params["wte"]),
            )
        )
    else:
        gpt2.tok_emb.weight = assign(gpt2.tok_emb.weight, params["wte"])

    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1
        )
        gpt2.trf_blocks[b].att.W_query.weight = assign(
            gpt2.trf_blocks[b].att.W_query.weight, q_w.T
        )
        gpt2.trf_blocks[b].att.W_key.weight = assign(
            gpt2.trf_blocks[b].att.W_key.weight, k_w.T
        )
        gpt2.trf_blocks[b].att.W_value.weight = assign(
            gpt2.trf_blocks[b].att.W_value.weight, v_w.T
        )

        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1
        )
        gpt2.trf_blocks[b].att.W_query.bias = assign(
            gpt2.trf_blocks[b].att.W_query.bias, q_b
        )
        gpt2.trf_blocks[b].att.W_key.bias = assign(
            gpt2.trf_blocks[b].att.W_key.bias, k_b
        )
        gpt2.trf_blocks[b].att.W_value.bias = assign(
            gpt2.trf_blocks[b].att.W_value.bias, v_b
        )

        gpt2.trf_blocks[b].att.out_proj.weight = assign(
            gpt2.trf_blocks[b].att.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T,
        )
        gpt2.trf_blocks[b].att.out_proj.bias = assign(
            gpt2.trf_blocks[b].att.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"],
        )

        gpt2.trf_blocks[b].ff.layers[0].weight = assign(
            gpt2.trf_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T,
        )
        gpt2.trf_blocks[b].ff.layers[0].bias = assign(
            gpt2.trf_blocks[b].ff.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"],
        )
        gpt2.trf_blocks[b].ff.layers[2].weight = assign(
            gpt2.trf_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T,
        )
        gpt2.trf_blocks[b].ff.layers[2].bias = assign(
            gpt2.trf_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"],
        )

        gpt2.trf_blocks[b].norm1.scale = assign(
            gpt2.trf_blocks[b].norm1.scale, params["blocks"][b]["ln_1"]["g"]
        )
        gpt2.trf_blocks[b].norm1.shift = assign(
            gpt2.trf_blocks[b].norm1.shift, params["blocks"][b]["ln_1"]["b"]
        )
        gpt2.trf_blocks[b].norm2.scale = assign(
            gpt2.trf_blocks[b].norm2.scale, params["blocks"][b]["ln_2"]["g"]
        )
        gpt2.trf_blocks[b].norm2.shift = assign(
            gpt2.trf_blocks[b].norm2.shift, params["blocks"][b]["ln_2"]["b"]
        )

    gpt2.final_norm.scale = assign(gpt2.final_norm.scale, params["g"])
    gpt2.final_norm.shift = assign(gpt2.final_norm.shift, params["b"])

    # same here we allow for extra tokens
    if gpt2.out_head.weight.shape[0] > len(params["wte"]):
        gpt2.out_head.weight = torch.nn.Parameter(
            gpt2.out_head.weight.index_put(
                (torch.LongTensor(range(len(params["wte"]))),),
                torch.tensor(params["wte"]),
            )
        )
    else:
        gpt2.out_head.weight = assign(gpt2.out_head.weight, params["wte"])

    return gpt2


def download_gpt2(
    model_size: str,
    models_dir: str | Path,
) -> tuple[dict, dict]:
    """
    Downloads GPT2 official weights from openai with a fallback
    to the LLMs-from-scratch repository.
    """
    import tensorflow as tf

    # Validate model size
    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"Model size not in {allowed_sizes}")

    # Define paths
    model_dir = os.path.join(models_dir, model_size)
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
    backup_base_url = "https://f001.backblazeb2.com/file/LLMs-from-scratch/gpt2"
    filenames = [
        "checkpoint",
        "encoder.json",
        "hparams.json",
        "model.ckpt.data-00000-of-00001",
        "model.ckpt.index",
        "model.ckpt.meta",
        "vocab.bpe",
    ]

    # Download files
    os.makedirs(model_dir, exist_ok=True)
    for filename in filenames:
        file_url = os.path.join(base_url, model_size, filename)
        backup_url = os.path.join(backup_base_url, model_size, filename)
        file_path = os.path.join(model_dir, filename)
        download_file(file_url, file_path, backup_url)

    # Load settings and params
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    settings = json.load(
        open(os.path.join(model_dir, "hparams.json"), "r", encoding="utf-8")
    )
    params = load_weights_from_tf_checkpoint(tf_ckpt_path, settings)

    # Instantiate a gpt2 class and populate it from the downloaded params
    gpt2_settings = GPT2Settings(
        model_size=size,
        attn_tf_compat=True
    )
    gpt2 = GPT2(settings)
    gpt2 = load_gpt2_from_dict(gpt2, params)
    torch.save(gpt2.state_dict(), path / f"gpt2_{size}.pkl")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Downloads gpt2 model weights as pytorch checkpoints.",
    )
    parser.add_argument("-o", "--output-dir", type=Path, required=True)
    parser.add_argument("-s", "--sizes", type=str, required=False)
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    sizes: list[str] = args.sizes.split(",")
    if not all(sizes in GPT2_SIZES):
        raise ValueError(
            f"Argument -s --sizes is expected to be in {GPT2_SIZES}.\n\t"
            "Multiple values can be given, separated with commas.\n\t"
            "Like so: --size 124M,355M,774M,1558M"
        )
    
    for size in sizes:
        download_gpt2(size, output_dir)
