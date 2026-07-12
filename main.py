# from IPython.display import clear_output, display

#!   pip install git+https://github.com/ndif-team/nnsight.git@dev
#!   pip install --upgrade diffusers transformers accelerate
#!   wget https://raw.githubusercontent.com/JadenFiotto-Kaufman/CVPR2026-HOW/master/colab/util.py

# clear_output()

import util

# ---------------------------------------------------------
import torch
import matplotlib.pyplot as plt
from nnsight import DiffusionModel

sd = DiffusionModel(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=torch.float16,
    safety_checker=None,
    dispatch=True,
    device_map="cuda",
)
# clear_output()

PROMPT = "Starry Night"
SEED = 43
NUM_INFERENCE_STEPS = 50

# ---------------------------------------------------------
import os
import torch
from nnsight import CONFIG, DiffusionModel

# Point at the workshop's NDIF host. No API key needed for the CVPR
# deployment; the cell uses the env var if set, otherwise falls back to
# a local cloudflare tunnel (replace with the URL from the talk).
CONFIG.API.HOST = os.environ.get("NDIF_HOST", "https://labs-greater-adjust-below.trycloudflare.com")
print(f"using NDIF at {CONFIG.API.HOST}")

# `dispatch=False` -> weights stay on meta locally; the full model lives
# on NDIF. The module tree + tokenizer still load eagerly so we can
# tokenize and write interventions against the right paths.
flux = DiffusionModel("black-forest-labs/FLUX.2-klein-4B", dispatch=False)
# clear_output()

PROMPT_2 = "A cat in a park on the grass by a tree"
CONCEPTS = ["cat", "grass", "sky", "tree"]
NUM_INFERENCE_STEPS_2 = 4
SEED_2 = 0

# ---------------------------------------------------------
with flux.session(remote=True):
    # Encode the prompt — return value is (prompt_embeds, text_ids).
    prompt_embeds = flux.pipeline.encode_prompt(
        prompt=PROMPT_2,
        device="cuda",
        num_images_per_prompt=1,
    )[0].save()

    # Encode each concept and keep just the position-3 (concept word) row.
    concept_rows = list().save()
    for c in CONCEPTS:
        emb = flux.pipeline.encode_prompt(
            prompt=c,
            device="cuda",
            num_images_per_prompt=1,
        )[0]
        concept_rows.append(emb[:, 3:4, :])

print("prompt_embeds shape: ", tuple(prompt_embeds.shape))
print(
    "concept rows: ", len(concept_rows), "× shape", tuple(concept_rows[0].shape)
)

concept_embeds = torch.cat(concept_rows, dim=1)  # [1, L_c, D]

# Splice: prompt first, concepts at the tail of the encoder sequence.
prompt_embeds_full = torch.cat(
    [prompt_embeds, concept_embeds.to(prompt_embeds.dtype)],
    dim=1,
)
L_txt = prompt_embeds.shape[1]
L_c = concept_embeds.shape[1]
L_img = (1024 // 16) ** 2  # 64 × 64 = 4096

# ---------------------------------------------------------
allow = torch.ones(L_txt + L_c + L_img, L_txt + L_c + L_img, dtype=torch.bool)
c_start, c_end = L_txt, L_txt + L_c
allow[:c_start, c_start:c_end] = False  # prompt → concept blocked
allow[c_end:, c_start:c_end] = False  # image → concept blocked
allow[c_start:c_end, :c_start] = False  # concept → prompt blocked

text_ids = torch.zeros(1, L_txt + L_c, 4, dtype=torch.long)
text_ids[:, :L_txt, 3] = torch.arange(L_txt)

# ---------------------------------------------------------
with flux.generate(
    prompt_embeds=prompt_embeds_full,
    attention_kwargs={"attention_mask": allow},
    width=1024,
    height=1024,
    num_inference_steps=NUM_INFERENCE_STEPS_2,
    seed=SEED_2,
    remote=True,
) as tracer:
    # CPU accumulator — works whether captured tensors come back from
    # NDIF on cuda:0 (remote GPU) or end up local; we just `.cpu()` each
    # score before adding. Pay one device-transfer per (step × layer),
    # not per residual.
    score_acc = torch.zeros(1, L_c, L_img, dtype=torch.float32).save()

    for _step in tracer.iter[:]:
        # Override the pipeline-derived txt_ids with our zero-position version.
        new_kwargs = dict(flux.transformer.inputs[1])
        new_kwargs["txt_ids"] = text_ids
        flux.transformer.inputs = (flux.transformer.inputs[0], new_kwargs)

        # Capture per-block PRE-projection attention outputs in forward order:
        # `to_add_out` (encoder) is called BEFORE `to_out[0]` (image) in
        # Flux2AttnProcessor — access in that order for nnsight's one-shot hooks.
        for blk in flux.transformer.transformer_blocks:
            enc_attn_pre = blk.attn.to_add_out.inputs[0][0]  # [1, L_txt+L_c, D]
            img_attn_pre = blk.attn.to_out[0].inputs[0][0]  # [1, L_img, D]
            concept_pre = enc_attn_pre[:, L_txt:]  # [1, L_c, D]
            scores = torch.einsum(
                "bpd,bcd->bcp",
                img_attn_pre.float(),
                concept_pre.float(),
            ).softmax(
                dim=-2
            )  # [1, L_c, L_img]
            score_acc.add_(scores.cpu())

    result = tracer.result.save()

n_blocks = len(flux.transformer.transformer_blocks)
n_accumulated = NUM_INFERENCE_STEPS_2 * n_blocks
result_image = result.images[0]

util.show_concept_heatmaps(
    image=result_image,
    score_acc=score_acc,
    n_accumulated=n_accumulated,
    concepts=CONCEPTS,
)

# ---------------------------------------------------------
import requests
import io
import PIL.Image
from nnsight import VisionLanguageModel

# Same NDIF host as Section 2 — set in the env or rewritten here.
CONFIG.API.HOST = os.environ.get("NDIF_HOST", "https://labs-greater-adjust-below.trycloudflare.com")
print(f"using NDIF at {CONFIG.API.HOST}")

llava = VisionLanguageModel("llava-hf/llava-1.5-7b-hf", dispatch=False)
# clear_output()