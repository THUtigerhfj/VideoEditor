---
language:
- en
license: other
license_name: flux-1-dev-non-commercial-license
license_link: LICENSE.md
extra_gated_prompt: By clicking "Agree", you agree to the [FluxDev Non-Commercial License Agreement](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev/blob/main/LICENSE.md)
  and acknowledge the [Acceptable Use Policy](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev/blob/main/POLICY.md).
tags:
- image-generation
- flux
- diffusion-single-file
---


![image/jpeg](https://cdn-uploads.huggingface.co/production/uploads/61fc209cef99814f1705e934/XWyTYtmdWKRPc6AAppa4f.jpeg)

`FLUX.1 Fill [dev]` is a 12 billion parameter rectified flow transformer capable of filling areas in existing images based on a text description.
For more information, please read our [blog post](https://blackforestlabs.ai/flux-1-tools/).

# Key Features
1. Cutting-edge output quality, second only to our state-of-the-art model `FLUX.1 Fill [pro]`.
2. Blends impressive prompt following with completing the structure of your source image.
3. Trained using guidance distillation, making `FLUX.1 Fill [dev]` more efficient.
4. Open weights to drive new scientific research, and empower artists to develop innovative workflows.
5. Generated outputs can be used for personal, scientific, and commercial purposes as described in the [`FLUX.1 [dev]` Non-Commercial License](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md).

# Usage
We provide a reference implementation of `FLUX.1 Fill [dev]`, as well as sampling code, in a dedicated [github repository](https://github.com/black-forest-labs/flux).
Developers and creatives looking to build on top of `FLUX.1 Fill [dev]` are encouraged to use this as a starting point.

## API Endpoints
The FLUX.1 models are also available in our API [bfl.ml](https://docs.bfl.ml/)


![image/png](https://cdn-uploads.huggingface.co/production/uploads/64510d6304397681bcf9725b/Z1gyNmGAfGigQtLdUCPXw.png)

## Diffusers

To use `FLUX.1 Fill [dev]` with the 🧨 diffusers python library, first install or upgrade diffusers

```shell
pip install -U diffusers
```

Then you can use `FluxFillPipeline` to run the model

```python
import torch
from diffusers import FluxFillPipeline
from diffusers.utils import load_image

image = load_image("https://huggingface.co/datasets/diffusers/diffusers-images-docs/resolve/main/cup.png")
mask = load_image("https://huggingface.co/datasets/diffusers/diffusers-images-docs/resolve/main/cup_mask.png")

pipe = FluxFillPipeline.from_pretrained("black-forest-labs/FLUX.1-Fill-dev", torch_dtype=torch.bfloat16).to("cuda")
image = pipe(
    prompt="a white paper cup",
    image=image,
    mask_image=mask,
    height=1632,
    width=1232,
    guidance_scale=30,
    num_inference_steps=50,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(0)
).images[0]
image.save(f"flux-fill-dev.png")
```

To learn more check out the [diffusers](https://huggingface.co/docs/diffusers/main/en/api/pipelines/flux) documentation

---

# Limitations
- This model is not intended or able to provide factual information.
- As a statistical model this checkpoint might amplify existing societal biases.
- The model may fail to generate output that matches the prompts.
- Prompt following is heavily influenced by the prompting-style.
- There may be slight-color shifts in areas that are not filled in
- Filling in complex textures may produce lines at the edges of the filled-area.

# Out-of-Scope Use
The model and its derivatives may not be used

- In any way that violates any applicable national, federal, state, local or international law or regulation.
- For the purpose of exploiting, harming or attempting to exploit or harm minors in any way; including but not limited to the solicitation, creation, acquisition, or dissemination of child exploitative content.
- To generate or disseminate verifiably false information and/or content with the purpose of harming others.
- To generate or disseminate personal identifiable information that can be used to harm an individual.
- To harass, abuse, threat