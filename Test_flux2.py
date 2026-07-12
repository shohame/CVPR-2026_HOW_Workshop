import torch
from nnsight import DiffusionModel

from pathlib import Path

LOCAL_MODEL_PATH = "flux2-klein-4b"

flux = DiffusionModel(
    LOCAL_MODEL_PATH,
    dispatch=True,
    torch_dtype=torch.bfloat16,  # drop/adjust if VRAM constrained
    device_map="cuda",           # or "cuda", "cpu", etc.
)
print ('Done loading models...')

PROMPT_2 = "A cat in a park on the grass by a tree"
CONCEPTS = ["cat", "grass", "sky", "tree"]
NUM_INFERENCE_STEPS_2 = 4
SEED_2 = 0

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
    "concept rows:        ", len(concept_rows), "× shape", tuple(concept_rows[0].shape)
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
