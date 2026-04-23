# Stable Diffusion from Scratch

A complete implementation of Stable Diffusion v1.5 built from scratch in PyTorch — no Diffusers library, no high-level abstractions. Every component is implemented manually.

## What's Implemented

| Component | File | Description |
|---|---|---|
| VAE Encoder | `encoder.py` | Encodes images into 4-channel latent space |
| VAE Decoder | `decoder.py` | Decodes latents back to pixel space |
| CLIP Text Encoder | `clip.py` | Encodes text prompts into conditioning vectors |
| U-Net Diffusion Model | `diffusion.py` | Predicts noise at each denoising step |
| DDPM Sampler | `ddpm.py` | Implements the reverse diffusion process |
| Self-Attention | `attention.py` | Multi-head self and cross-attention modules |
| Inference Pipeline | `pipeline.py` | End-to-end generation with CFG support |

## Features

- **Text-to-image** generation with classifier-free guidance (CFG)
- **Image-to-image** generation with configurable noise strength
- **CPU and GPU** support (CUDA and Apple MPS)
- Configurable guidance scale, inference steps, and random seed

## Setup

### 1. Install dependencies

```bash
pip install torch torchvision transformers pillow tqdm
```

### 2. Download model weights

Download the following files from [Hugging Face](https://huggingface.co/runwayml/stable-diffusion-v1-5):

- `v1-5-pruned-emaonly.ckpt` → place in `data/`
- `vocab.json` → place in `data/`
- `merges.txt` → place in `data/`

```
stable-diffusion-from-scratch/
├── data/
│   ├── v1-5-pruned-emaonly.ckpt
│   ├── vocab.json
│   └── merges.txt
├── attention.py
├── clip.py
├── decoder.py
├── encoder.py
├── diffusion.py
├── ddpm.py
├── pipeline.py
├── model_loader.py
├── model_converter.py
└── demo.py
```

### 3. Run inference

```bash
python demo.py
```

The output image will be saved as `output.png`.

## Configuration

Edit `demo.py` to change:

```python
ALLOW_CUDA = True       # Enable GPU (NVIDIA)
ALLOW_MPS  = True       # Enable GPU (Apple Silicon)

prompt = "your prompt here"
cfg_scale = 8           # Guidance scale (1-14)
num_inference_steps = 50  # Steps (more = better quality, slower)
seed = 42               # Random seed for reproducibility
```

## CPU vs GPU

| Device | Steps | Estimated Time |
|---|---|---|
| CPU | 20 | ~15-20 minutes |
| CPU | 50 | ~40-60 minutes |
| GPU (RTX 3080) | 50 | ~20 seconds |
| Apple M2 (MPS) | 50 | ~3-5 minutes |

For CPU usage, the script automatically reduces steps to 20.

## Architecture Notes

The implementation follows the original [Stable Diffusion](https://github.com/CompVis/stable-diffusion) architecture:

- The **VAE** compresses 512×512 RGB images into 64×64×4 latents
- The **CLIP encoder** converts token sequences (max 77 tokens) into 768-dim embeddings
- The **U-Net** operates entirely in latent space, conditioned on text via cross-attention
- The **DDPM sampler** runs 50 denoising steps with linear beta schedule from 0.00085 to 0.012
- **Classifier-free guidance** blends conditional and unconditional predictions for quality improvement

## References

- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — Ho et al., 2020
- [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) — Rombach et al., 2022
- [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) — Radford et al., 2021

Credits & Acknowledgments
I would like to express my sincere gratitude to Umar Jamil for his exceptional educational content. This implementation was built following his deep-dive tutorial, which was instrumental in understanding the complex interplay between the mathematics and the code in diffusion models.

Tutorial: Coding Stable Diffusion from scratch in PyTorch by Umar Jamil.


