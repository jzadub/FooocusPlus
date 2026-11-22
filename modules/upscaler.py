import numpy as np
import torch
from collections import OrderedDict
from PIL import Image

import modules.core as core
import modules.loader as loader
from enhanced.translator import interpret
from ldm_patched.contrib.external_upscale_model import ImageUpscaleWithModel
from ldm_patched.pfn.architecture.RRDB import RRDBNet as ESRGAN

opImageUpscaleWithModel = ImageUpscaleWithModel()
model = None


def perform_upscale(img):
    global model

    interpret('[Upscaler] Upscaling image with shape', f'{str(img.shape)}...')

    if model is None:
        model_filename = loader.download_upscale_model()
        sd = torch.load(model_filename, weights_only=True)
        sdo = OrderedDict()
        for k, v in sd.items():
            sdo[k.replace('residual_block_', 'RDB')] = v
        del sd
        model = ESRGAN(sdo)
        model.cpu()
        model.eval()

    img = core.numpy_to_pytorch(img)
    img = opImageUpscaleWithModel.upscale(model, img)[0]
    img = core.pytorch_to_numpy(img)[0]

    return img


def execute_fast_upscale(image_array: np.ndarray, scale_factor: float) -> np.ndarray:
    """
    High-speed AI Upscale: UltraSharp neural pass
    + Lanczos dimension alignment. Executes in ~1.5
    seconds without invoking heavy diffusion samplers.
    """

    orig_h, orig_w = image_array.shape[:2]
    target_w = int(orig_w * scale_factor)
    target_h = int(orig_h * scale_factor)

    interpret(f'[Upscaler] Running Fast Upscale: {orig_w}×{orig_h} → {target_w}×{target_h} ({scale_factor}x)...')

    # Step 1: Neural super-resolution pass
    # (4x-UltraSharp via ESRGAN engine)
    ai_upscaled = perform_upscale(image_array)

    # Step 2: High-quality Lanczos alignment
    # to the exact user multiplier
    pil_img = Image.fromarray(ai_upscaled)
    if pil_img.size != (target_w, target_h):
        pil_img = pil_img.resize((target_w, target_h), resample=Image.Resampling.LANCZOS)

    return np.array(pil_img)
