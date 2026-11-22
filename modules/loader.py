import httpx
import json
import os  # Strictly used only for os.environ environment variable checks
import shutil
import time

from pathlib import Path
from tqdm import tqdm
from urllib.parse import urlparse
from typing import Optional

import common
import modules.flags as flags
import modules.user_structure as US
from args_manager import args
from enhanced.translator import interpret, \
    interpret_info, interpret_warn


current_dir = Path.cwd().resolve()
base_model_name = ''
lora_filenames = []
model_filenames = []
path_wildcards = Path(current_dir / 'wildcards')
sd15_model_path = 'SD1.5/realisticVisionV60B1_v51VAE.safetensors'
vae_filenames = []
wildcard_filenames = []
_verified_download_paths = set()


def is_comfy_checkpoint(file_path_str: str) -> bool:
    """Helper to check if a checkpoint resides inside a Comfy-only folder."""
    parts = Path(file_path_str).parts
    if parts:
        first_part = parts[0].lower()

        # --- MIXED ALTERNATIVE FOLDER GATEWAY ---
        # If the file is inside 'Alternative/', only allow it through if it is the
        # Playground model. Exclude all other Comfy-specific contents from standard view.
        if first_part == 'alternative':
            return 'playground' not in file_path_str.lower()

        # Exact folder name matches from the checkpoints directory
        comfy_checkpoint_folders = {
            'fluxdev', 'fluxkrea', 'fluxschnell', 'krea2', 'sd1.5', 'sd3x', 'sdxl_gguf',
            'z-image'
        }
        return first_part in comfy_checkpoint_folders
    return False


def get_base_model_list(engine='Fooocus', task_method=None, for_import=False):
    global base_model_name

    # If called by the metadata importer,
    # bypass all UI filters and return the raw list
    if for_import:
        return common.MODELS_INFO.get_model_names('checkpoints')

    # 1. Fetch the default filter and get the matching model list
    file_filter = flags.model_file_filter.get(engine, [])
    base_model_list = common.MODELS_INFO.get_model_names('checkpoints', file_filter)

    # Inspect the currently selected model name
    # to determine the dynamic filter mode
    model_lower = base_model_name.lower() if base_model_name else ''
    method_lower = str(task_method).lower() if task_method else ''

    # 2. For SD1.5 (Comfy SD1.5) engine modes:
    # Check if the model name or engine indicates SD1.5
    if 'sd1.5' in model_lower or engine == 'SD1':
        return [f for f in base_model_list if 'sd1.5' in f.replace('\\', '/').lower()]

    # 2.5. For SDXL / Pony GGUF Models
    is_sdxl_gguf = (
        'sdxl_base_gguf' in method_lower or
        'sdxl_pony_gguf' in method_lower or
        'sdxl_gguf' in model_lower or
        '2gb_' in model_lower
    )

    if engine == 'Comfy' and is_sdxl_gguf:
        # Fetch all models located inside the SDXL_GGUF folder
        gguf_models = [
            f for f in base_model_list
            if 'sdxl_gguf' in f.replace('\\', '/').lower() and f.endswith('gguf')
        ]

        is_pony = 'pony' in model_lower
        # Check whether the active model/preset is 4-step / Turbo
        is_turbo = any(k in model_lower for k in ['4step', '4-step', 'turbo'])

        if is_pony:
            # Show only Pony GGUF models
            return [f for f in gguf_models if 'pony' in f.lower()]
        elif is_turbo:
            # Show only 4-step / turbo GGUF models
            return [f for f in gguf_models if any(k in f.lower() for k in ['4step', '4-step', 'turbo', 'lightning'])]
        else:
            # Show only standard 20-step base GGUF models
            return [f for f in gguf_models if not any(k in f.lower() for k in ['4step', '4-step', 'turbo', 'pony'])]

    # 3. For standard Fooocus or Comfy
    # (SDXL / SD1.5) modes:
    # Exclude specialized models (SD3, Flux, HyDiT)
    # and any models inside subdirectories
    if engine in ['Fooocus', 'Comfy']:
        base_model_list = common.MODELS_INFO.get_model_names('checkpoints', flags.model_file_filter['Fooocus'], reverse=True)
        valid_models = [f for f in base_model_list if not is_comfy_checkpoint(f)]

        # Check active preset names and
        # model flags for Pony and LowVRAM
        preset_name = str(getattr(args, 'preset', '')).lower() if hasattr(args, 'preset') else ''
        is_pony = 'pony' in model_lower or 'pony' in method_lower or 'pony' in preset_name
        is_lowvram = 'lowvram' in model_lower or 'lowvram' in preset_name or '4gb_' in preset_name or '4gb' in model_lower

        if is_lowvram:
            # Filter models residing specifically
            # inside the LowVRAM/ directory
            lowvram_models = [f for f in valid_models if f.replace('\\', '/').lower().startswith('lowvram/')]
            if is_pony:
                # 4GB_Pony: Show ONLY Pony
                # models inside LowVRAM/
                return [f for f in lowvram_models if 'pony' in f.lower()]
            else:
                # 4GB_Default, 4GB_Realistic, etc,
                # Show ONLY non-Pony models
                # inside LowVRAM/
                return [f for f in lowvram_models if 'pony' not in f.lower()]

        if is_pony:
            # Standard Pony Presets: Show strictly
            # models residing inside Pony/
            return [f for f in valid_models if f.replace('\\', '/').lower().startswith('pony/')]
        else:
            # Standard SDXL Presets: Exclude Pony/
            # and LowVRAM/ directories completely
            return [
                f for f in valid_models
                if not f.replace('\\', '/').lower().startswith('pony/')
                and not f.replace('\\', '/').lower().startswith('lowvram/')
            ]

    # 4. For Flux engine modes:
    elif engine == 'Flux':

        is_aio_mode = 'aio' in model_lower or 'aio' in method_lower or 'all_in_one' in method_lower
        is_gguf_mode = '.gguf' in model_lower or 'gguf' in method_lower
        is_4gguf_workflow = '4gguf' in method_lower or '_4' in method_lower
        is_hyperflux = 'hyperflux' in model_lower or '-hyp' in model_lower

        is_krea2 = (
            any(k in model_lower for k in ['krea2', 'krea-2', 'krea_2']) or
            any(k in method_lower for k in ['krea2', 'krea-2', 'krea_2'])
        )
        is_schnell = 'schnell' in model_lower or 'schnell' in method_lower

        # Base / Turbo sub-family flags
        is_turbo = (
            any(kw in model_lower for kw in flags.TURBO_MODEL_KEYWORDS) or
            any(kw in method_lower for kw in flags.TURBO_METHOD_KEYWORDS)
        )

        is_z_image = (
            any(kw in model_lower for kw in flags.Z_IMAGE_MODEL_KEYWORDS) or
            any(kw in method_lower for kw in flags.Z_IMAGE_METHOD_KEYWORDS)
        )

        # ----------------------------------------
        # Krea 2 Sub-Family
        # ----------------------------------------
        if is_krea2:
            krea2_all = [
                f for f in base_model_list
                if any(k in f.lower() for k in ['krea2', 'krea-2', 'krea_2'])
                and 'flux1-krea' not in f.lower()
            ]
            # Differentiate between Turbo
            # and Base (Raw) sub-families
            is_krea_turbo = 'turbo' in model_lower or 'turbo' in method_lower

            if is_krea_turbo:
                krea2_files = [f for f in krea2_all if 'turbo' in f.lower()]
            else:
                krea2_files = [f for f in krea2_all if 'turbo' not in f.lower()]

            if is_gguf_mode:
                return [f for f in krea2_files if f.endswith('gguf')]
            elif is_aio_mode:
                return [f for f in krea2_files if not f.endswith('gguf') and 'aio' in f.lower()]
            else:
                return [f for f in krea2_files if not f.endswith('gguf') and 'aio' not in f.lower()]

        # ----------------------------------------------------
        # A. Z-Image Sub-Family (Requirements 1 & 2)
        # ----------------------------------------------------
        if is_z_image:
            z_files = [f for f in base_model_list if any(k in f.lower() for k in ['z-image', 'z_image', 'z-img', 'z_img'])]
            is_z_4gguf = is_4gguf_workflow or 'q4' in model_lower

            if is_turbo:
                turbo_files = [f for f in z_files if 'turbo' in f.lower()]
                if is_gguf_mode:
                    # ZI-Turbo 4GGUF shows ONLY Q4 GGUF
                    # (excludes Q5 and Q8)
                    if is_z_4gguf:
                        return [f for f in turbo_files if f.endswith('gguf') and 'q4' in f.lower()]
                    else:
                        return [f for f in turbo_files if f.endswith('gguf') and 'q4' not in f.lower()]
                elif is_aio_mode:
                    return [f for f in turbo_files if not f.endswith('gguf') and 'aio' in f.lower()]
                else:
                    return [f for f in turbo_files if not f.endswith('gguf') and 'aio' not in f.lower()]
            else:
                base_files = [f for f in z_files if 'turbo' not in f.lower()]
                if is_gguf_mode:
                    # ZI-Base 4GGUF shows ONLY Q4 GGUF
                    # (excludes Q5 and Q8)
                    if is_z_4gguf:
                        return [f for f in base_files if f.endswith('gguf') and 'q4' in f.lower()]
                    else:
                        return [f for f in base_files if f.endswith('gguf') and 'q4' not in f.lower()]
                elif is_aio_mode:
                    return [f for f in base_files if not f.endswith('gguf') and 'aio' in f.lower()]
                else:
                    return [f for f in base_files if not f.endswith('gguf') and 'aio' not in f.lower()]

        # ----------------------------------------------------
        # B. LowVRAM / 4GGUF Flux Family (Requirement 3)
        # ----------------------------------------------------
        if is_4gguf_workflow and is_gguf_mode:
            # Match 4-bit quants AND q2k
            low_quants = ['q4_k_s', 'q4_k_m', 'q4ks', 'q4_0', 'q4_1', 'q2k', 'q2_k', 'q3k', 'q3_k']
            low_vram_ggufs = [
                f for f in base_model_list
                if f.endswith('gguf')
                and any(q in f.lower() for q in low_quants)
                and 'z-image' not in f.lower()
                and 'z_image' not in f.lower()
                and 'krea2' not in f.lower()
            ]

            # HyperFlux4 shows hyperflux Diversity q4KS AND q2K
            if is_hyperflux:
                return [f for f in low_vram_ggufs if 'hyperflux' in f.lower() or '-hyp' in f.lower()]
            else:
                return [f for f in low_vram_ggufs if 'hyperflux' not in f.lower() and '-hyp' not in f.lower()]

        # ----------------------------------------------------
        # C. Standard Flux GGUF Workflows (flux_base_gguf)
        # ----------------------------------------------------
        if is_gguf_mode:
            flux_ggufs = [
                f for f in base_model_list
                if f.endswith('gguf')
                and 'z-image' not in f.lower()
                and 'z_image' not in f.lower()
                and 'krea2' not in f.lower()
            ]

            # Exclude Q2_K, Q3_K and Q4_K
            # from standard GGUF workflows
            flux_ggufs = [
                f for f in flux_ggufs
                if 'q2k' not in f.lower()
                and 'q2_k' not in f.lower()
                and 'q3k' not in f.lower()
                and 'q3_k' not in f.lower()
                and 'q4k' not in f.lower()
                and 'q4_k' not in f.lower()
            ]

            # HyperFlux presets
            if is_hyperflux:
                # Issue 2 Fix: Distinguish HyperFlux Schnell from HyperFlux Dev
                if is_schnell:
                    # HyperFlux1S5 / HyperFlux1S8 show ONLY Schnell HyperFlux models
                    return [f for f in flux_ggufs if ('hyperflux' in f.lower() or '-hyp' in f.lower()) and 'schnell' in f.lower()]
                else:
                    # HyperFluxFast shows ONLY Dev HyperFlux models (Diversity)
                    return [f for f in flux_ggufs if ('hyperflux' in f.lower() or '-hyp' in f.lower()) and 'schnell' not in f.lower()]

            # Standard Non-HyperFlux GGUFs
            if is_schnell:
                return [f for f in flux_ggufs if 'schnell' in f.lower() and 'hyperflux' not in f.lower() and '-hyp' not in f.lower()]
            else:
                return [f for f in flux_ggufs if 'schnell' not in f.lower() and 'hyperflux' not in f.lower() and '-hyp' not in f.lower()]

        # ----------------------------------------------
        # D. Standard Flux FP8 / Safetensors (Non-GGUF)
        # ----------------------------------------------
        safetensors_models = [
            f for f in base_model_list
            if not f.endswith('gguf')
            and 'z-image' not in f.lower()
            and 'z_image' not in f.lower()
            and 'krea2' not in f.lower()
        ]

        # Distinguish Flux Schnell FP8 (Flux1S_FP8)
        # from Flux Dev/Krea FP8
        if is_schnell:
            return [f for f in safetensors_models if 'schnell' in f.lower()]
        else:
            return [f for f in safetensors_models if 'schnell' not in f.lower()]

    # 5. For SD3x engine modes:
    elif engine == 'SD3x':

        if '.gguf' in model_lower:
            return [f for f in base_model_list if f.endswith('gguf')]
        else:
            return [f for f in base_model_list if not f.endswith('gguf')]

    return base_model_list


def is_comfy_lora(file_path_str: str) -> bool:
    """Helper to check if a LoRA resides inside a Comfy-only folder."""
    parts = Path(file_path_str).parts
    if parts:
        first_part = parts[0].lower()
        # Exact folder name matches from the LoRAs directory
        # These folder names must be listed in lower case
        comfy_lora_folders = {'flux', 'krea2', 'sd1.5', 'sd3x', 'z-image'}
        return first_part in comfy_lora_folders
    return False


def get_lora_model_list(engine='Fooocus', task_method=None, for_import=False) -> list:
    """
    Recursively filters LoRA files based on the active preset
    - Alternative (Kolors, HyDiT, Playground): Shows flat root LoRAs + 'Alternative/' subfolder
      (except the HyDiT preset only shows the 'Alternative/' subfolder)
    - Flux: Shows only 'Flux/' folder LoRAs (excluding Z-Image)
    - SD3.5: Shows only 'SD3x/' folder LoRAs
    - SD1.5: Shows only 'SD1.5/' folder LoRAs
    - Pony: Shows only 'Pony/' folder LoRAs
    - Z-Image: Shows only 'Z-Image' folder LoRAs
    - SDXL (Standard): Shows flat root LoRAs + any custom user subdirectories, hiding Comfy-only folders
    """
    global base_model_name

    # Fetch all raw LoRA files (recursive list)
    raw_loras = common.MODELS_INFO.get_model_names('loras')

    # Bypassed by metadata importer to ensure
    # paths are resolved regardless of UI state
    if for_import:
        return raw_loras

    # Extract state strings for checking
    model_lower = base_model_name.lower() if base_model_name else ''
    method_lower = str(task_method).lower() if task_method else ''

    # 0. Krea 2 Sub-Family check
    is_krea2 = any(x in model_lower for x in ['krea2', 'krea-2', 'krea_2']) or \
               any(x in method_lower for x in ['krea2', 'krea-2', 'krea_2'])
    if is_krea2:
        return [
            f for f in raw_loras
            if any(k in f.lower() for k in ['krea2', 'krea-2', 'krea_2'])
        ]

    # 1. Z-Image Engine check
    is_z_image = any(x in model_lower for x in ['z-image', 'z_image']) or \
                 any(x in method_lower for x in ['zit', 'zib'])
    if is_z_image:
        return [f for f in raw_loras if 'z-image' in f.lower() or 'z_image' in f.lower()]

    # 2. Flux Engine check
    if engine == 'Flux':
        return [f for f in raw_loras if 'flux' in f.lower() and 'z-image' not in f.lower() and 'z_image' not in f.lower()]

    # 3. SD3x Engine check
    if engine == 'SD3x':
        return [f for f in raw_loras if 'sd3x' in f.lower()]

    # 4. SD1 (SD1.5) Engine check
    if engine == 'SD1' or 'sd15' in method_lower:
        return [f for f in raw_loras if 'sd1.5' in f.replace('\\', '/').lower()]

    # 5. Pony Sub-Family (SDXL Checkpoints)
    if 'pony' in model_lower:
        return [f for f in raw_loras if 'pony' in f.lower()]

    # 6. Alternative Engines Filter
    # (Kolors and Playground Only)
    # Restricts the choices to flat root-level LoRAs plus those stored in the 'Alternative' subfolder
    if engine == 'Kolors+' or (model_lower and 'playground' in model_lower):
        return [
            f for f in raw_loras
            if '/' not in f and '\\' not in f or 'alternative' in f.lower()
        ]

    # 7. HyDiT+ Engine Filter (Strict Lockout)
    # Strictly protects HyDiT from standard SDXL
    # LoRAs by showing ONLY specialized LoRAs
    # stored inside the 'Alternative/' directory
    if engine == 'HyDiT+':
        return [f for f in raw_loras if 'alternative' in f.lower()]

    # 8. Standard SDXL (Fooocus / Comfy)
    # Include standard subfolders,
    # but exclude Comfy-only and Pony LoRA folders
    if engine in ['Fooocus', 'Comfy']:
        return [f for f in raw_loras if not is_comfy_lora(f) and 'pony' not in f.replace('\\', '/').lower()]

    # Fallback: return everything
    return raw_loras


def get_vae_list(engine='Fooocus', task_method=None) -> list:
    # Returns the filtered VAE choices for the dropdown.
    # Merges curated in-app catalogs with any
    # user-downloaded custom VAEs.
    global base_model_name

    raw_vaes = common.MODELS_INFO.get_model_names('vae')
    model_lower = base_model_name.lower() if base_model_name else ''
    method_lower = str(task_method).lower() if task_method else ''
    engine_lower = str(engine).lower()

    is_aio = 'aio' in method_lower or 'all_in_one' in method_lower or 'aio' in model_lower
    is_gguf = 'gguf' in method_lower or '.gguf' in model_lower

    def is_flux_vae(name):
        n = Path(name).name.lower()
        return 'flux' in n or 'ultraflux' in n or n.startswith('ae.') or 'z-image' in n or 'z_image' in n

    def is_sd3_vae(name):
        n = Path(name).name.lower()
        return 'sd3' in n

    def is_sd15_vae(name):
        n = Path(name).name.lower()
        p = str(name).lower()
        if 'sdxl' in n or 'xl' in n or 'pony' in n:
            return False
        keywords = [
            'sd15', 'sd1.5', 'sd1-5', 'v1-5', '1.5',
            'mse', 'ema', '560000', '840000',
            'kl-f8', 'clearvae', 'blessed', 'orangemix', 'anything'
        ]
        return any(k in n for k in keywords) or 'sd1.5' in p or 'sd15' in p

    def is_sdxl_vae(name):
        n = Path(name).name.lower()
        if 'ponydiffusionv6xl_vae' in n or n.startswith('sdxl_vae.'):
            return False
        return not (is_flux_vae(name) or is_sd3_vae(name) or is_sd15_vae(name))

    # 1. SD3.5 Engine: Curated SD3 list
    # plus any custom SD3 VAEs on disk
    if 'sd3' in engine_lower:
        curated_sd3_files = {flags.extract_vae_filename(k): k for k in flags.SD3_VAES.keys()}
        sd3_list = list(flags.SD3_VAES.keys())
        for f in raw_vaes:
            if is_sd3_vae(f) and Path(f).name not in curated_sd3_files:
                sd3_list.append(f)
        return sd3_list

    # 2. Check for Krea 2 sub-family
    if 'krea' in model_lower or 'krea' in method_lower:
        curated_krea_files = {flags.extract_vae_filename(k): k for k in flags.KREA_VAES.keys()}
        krea_list = list(flags.KREA_VAES.keys())
        for f in raw_vaes:
            if 'qwen' in f.lower() and Path(f).name not in curated_krea_files:
                krea_list.append(f)
        return krea_list

    # 3. Flux / Z-Image Engine: Curated Flux list
    # plus any custom Flux VAEs on disk
    if 'flux' in engine_lower:
        curated_flux_files = {flags.extract_vae_filename(k): k for k in flags.FLUX_VAES.keys()}
        flux_list = list(flags.FLUX_VAES.keys())
        for f in raw_vaes:
            if is_flux_vae(f) and Path(f).name not in curated_flux_files:
                flux_list.append(f)
        return flux_list

    # 4. SD1.5 Mode: Curated SD1.5 list
    # plus any custom 1.5 VAEs on disk
    if 'sd1' in engine_lower or 'sd1.5' in model_lower or 'sd15' in method_lower:
        curated_15_files = {flags.extract_vae_filename(k): k for k in flags.SD15_VAES.keys()}
        sd15_list = list(flags.SD15_VAES.keys())
        for f in raw_vaes:
            if is_sd15_vae(f) and Path(f).name not in curated_15_files:
                sd15_list.append(f)
        return sd15_list

    # 5. Standard SDXL: Curated SDXL list
    # plus any custom SDXL VAEs on disk
    curated_xl_files = {flags.extract_vae_filename(k): k for k in flags.SDXL_VAES.keys()}
    sdxl_list = list(flags.SDXL_VAES.keys())
    for f in raw_vaes:
        if is_sdxl_vae(f) and Path(f).name not in curated_xl_files:
            sdxl_list.append(f)
    return sdxl_list


def update_files(engine='Fooocus', task_method=None):
    global model_filenames, lora_filenames, \
        path_wildcards, vae_filenames, \
        wildcard_filenames
    common.MODELS_INFO.refresh_from_path()
    model_filenames = get_base_model_list(engine, task_method)
    lora_filenames = get_lora_model_list(engine, task_method)
    vae_filenames = get_vae_list(engine, task_method)
    wildcard_filenames = US.list_files_by_patterns(path_wildcards, ['*.txt'])
    return model_filenames, lora_filenames, vae_filenames


# --- Model Download Section ---
def get_remote_file_size(url: str, timeout: float = 10.0) -> Optional[int]:
    # Queries the server for the expected
    # byte size of the remote file:
    try:
        with httpx.Client(follow_redirects=True, timeout=timeout) as client:
            response = client.head(url)
            for hist in response.history:
                linked_size = hist.headers.get('x-linked-size')
                if linked_size and linked_size.isdigit():
                    return int(linked_size)

            if 200 <= response.status_code < 400:
                linked_size = response.headers.get('x-linked-size')
                if linked_size and linked_size.isdigit():
                    return int(linked_size)
                encoding = response.headers.get('content-encoding', '').lower()
                content_length = response.headers.get('content-length')
                if content_length and content_length.isdigit() and encoding in ('', 'identity'):
                    return int(content_length)

            # Fallback for servers (such as S3 presigned URLs) that reject HEAD
            with client.stream('GET', url, headers={'Range': 'bytes=0-0'}) as range_resp:
                if range_resp.status_code == 206:
                    content_range = range_resp.headers.get('content-range', '')
                    if '/' in content_range:
                        total_str = content_range.rsplit('/', 1)[-1].strip()
                        if total_str.isdigit():
                            return int(total_str)
                elif range_resp.status_code == 200:
                    encoding = range_resp.headers.get('content-encoding', '').lower()
                    content_length = range_resp.headers.get('content-length')
                    if content_length and content_length.isdigit() and encoding in ('', 'identity'):
                        return int(content_length)
    except Exception:
        pass
    return None


def verify_safetensors_completeness(file_path: Path) -> bool:
    # Validates a .safetensors header and confirms
    # the file contains all declared tensor bytes.
    if not file_path.is_file():
        return False
    try:
        file_size = file_path.stat().st_size
        if file_size < 10:
            return False
        with file_path.open('rb') as f:
            header_len = int.from_bytes(f.read(8), 'little')
            if header_len <= 0 or header_len > 100_000_000 or 8 + header_len > file_size:
                return False
            header_data = json.loads(f.read(header_len).decode('utf-8'))

        max_end = 0
        for key, value in header_data.items():
            if key == '__metadata__':
                continue
            if isinstance(value, dict) and 'data_offsets' in value:
                offsets = value['data_offsets']
                if isinstance(offsets, list) and len(offsets) == 2 and offsets[1] > max_end:
                    max_end = offsets[1]

        return file_size >= (8 + header_len + max_end)
    except Exception:
        return False


def is_file_complete(file_path: Path, url: Optional[str] = None) -> bool:
    # Checks whether a model file on disk is
    # complete and uncorrupted:
    if not file_path.is_file():
        return False

    actual_size = file_path.stat().st_size
    if actual_size == 0:
        return False

    resolved_key = str(file_path.resolve())
    if resolved_key in _verified_download_paths:
        return True

    if file_path.suffix.lower() == '.safetensors':
        if not verify_safetensors_completeness(file_path):
            return False

    if url:
        domain = os.environ.get('HF_MIRROR', 'https://huggingface.co').rstrip('/')
        resolved_url = str.replace(url, 'https://huggingface.co', domain, 1)
        expected_size = get_remote_file_size(resolved_url, timeout=5.0)
        if expected_size is not None and expected_size > 0 and actual_size < expected_size:
            return False

    _verified_download_paths.add(resolved_key)
    return True


def parse_hf_url(url: str, domain: str) -> Optional[tuple]:
    """Extracts (repo_id, revision, repo_filename) from a Hugging Face resolve/blob URL."""
    clean_url = url.split('?', 1)[0]
    parts = clean_url.split(domain + '/', 1)
    if len(parts) < 2:
        return None

    separator = '/resolve/' if '/resolve/' in parts[1] else '/blob/'
    if separator not in parts[1]:
        return None

    sub_parts = parts[1].split(separator, 1)
    repo_id = sub_parts[0]
    rev_and_path = sub_parts[1].split('/', 1)
    if len(rev_and_path) < 2:
        return None

    revision, repo_filename = rev_and_path[0], rev_and_path[1]
    return repo_id, revision, repo_filename


def download_with_resume_and_verify(
        url: str,
        cached_file: Path,
        expected_size: Optional[int],
        progress: bool = True,
        max_retries: int = 3
) -> bool:
    # Downloads to a .part file with Range resume,
    # byte-count verification and automatic retries
    part_file = cached_file.with_name(cached_file.name + '.part')

    for attempt in range(1, max_retries + 1):
        try:
            existing_bytes = part_file.stat().st_size if part_file.exists() else 0
            if expected_size is not None and existing_bytes > expected_size:
                part_file.unlink()
                existing_bytes = 0

            if expected_size is not None and expected_size > 0 and existing_bytes == expected_size:
                if cached_file.suffix.lower() != '.safetensors' or verify_safetensors_completeness(part_file):
                    part_file.replace(cached_file)
                    _verified_download_paths.add(str(cached_file.resolve()))
                    return True
                part_file.unlink()
                existing_bytes = 0

            headers = {}
            if existing_bytes > 0:
                headers['Range'] = f'bytes={existing_bytes}-'

            timeout = httpx.Timeout(30.0, read=60.0)
            with httpx.Client(follow_redirects=True, timeout=timeout) as client:
                with client.stream('GET', url, headers=headers) as response:
                    if existing_bytes > 0 and response.status_code == 416:
                        part_file.unlink()
                        continue

                    if existing_bytes > 0 and response.status_code == 200:
                        existing_bytes = 0

                    response.raise_for_status()

                    total_size = expected_size
                    if total_size is None:
                        encoding = response.headers.get('content-encoding', '').lower()
                        if response.status_code == 206:
                            content_range = response.headers.get('content-range', '')
                            if '/' in content_range:
                                total_str = content_range.rsplit('/', 1)[-1].strip()
                                if total_str.isdigit():
                                    total_size = int(total_str)
                        elif response.status_code == 200 and encoding in ('', 'identity'):
                            content_length = response.headers.get('content-length')
                            if content_length and content_length.isdigit():
                                total_size = int(content_length)

                    file_mode = 'ab' if (existing_bytes > 0 and response.status_code == 206) else 'wb'
                    with part_file.open(file_mode) as f:
                        with tqdm(
                            total=total_size,
                            initial=existing_bytes,
                            unit='B',
                            unit_scale=True,
                            unit_divisor=1024,
                            disable=not progress,
                            desc=cached_file.name
                        ) as pbar:
                            for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                                if chunk:
                                    f.write(chunk)
                                    pbar.update(len(chunk))

            actual_size = part_file.stat().st_size if part_file.exists() else 0
            if actual_size == 0:
                interpret_warn('[Loader] Downloaded file is empty:', cached_file.name)
                if part_file.exists():
                    part_file.unlink()
                if attempt < max_retries:
                    interpret_info('[Loader] Retrying download:', f'{attempt + 1}/{max_retries}...')
                    time.sleep(2 * attempt)
                continue

            if total_size is not None and total_size > 0 and actual_size != total_size:
                interpret_warn('[Loader] Download incomplete:', f'{cached_file.name} ({actual_size:,} / {total_size:,} bytes)')
                if attempt < max_retries:
                    interpret_info('[Loader] Retrying download:', f'{attempt + 1}/{max_retries}...')
                    time.sleep(2 * attempt)
                continue

            if cached_file.suffix.lower() == '.safetensors' and not verify_safetensors_completeness(part_file):
                interpret_warn('[Loader] Downloaded safetensors file failed verification:', cached_file.name)
                part_file.unlink()
                if attempt < max_retries:
                    interpret_info('[Loader] Retrying download:', f'{attempt + 1}/{max_retries}...')
                    time.sleep(2 * attempt)
                continue

            part_file.replace(cached_file)
            _verified_download_paths.add(str(cached_file.resolve()))
            interpret_info('[Loader] Successfully downloaded:', cached_file.name)
            print()
            return True

        except Exception as e:
            interpret_warn('[Loader] Download interrupted:', f'{cached_file.name}: {e}')
            if attempt < max_retries:
                interpret_info('[Loader] Retrying download:', f'{attempt + 1}/{max_retries}...')
                time.sleep(2 * attempt)

    return False


def load_file_from_url(
        url: str,
        *,
        model_dir: str,
        progress: bool = True,
        file_name: Optional[str] = None,
) -> str:
    # Download a file from `url` into `model_dir`,
    # verifying completeness & retrying if truncated.
    # 1. Resolve Hugging Face Mirror environment
    domain = os.environ.get('HF_MIRROR', 'https://huggingface.co').rstrip('/')
    url = str.replace(url, 'https://huggingface.co', domain, 1)

    # 2. Resolve paths using pathlib.Path
    model_path_dir = Path(model_dir)
    model_path_dir.mkdir(parents=True, exist_ok=True)

    if not file_name:
        parts = urlparse(url)
        file_name = Path(parts.path).name

    cached_file = (model_path_dir / file_name).resolve()

    need_download = False
    if not cached_file.exists():
        need_download = True
    elif not is_file_complete(cached_file, url):
        interpret_warn('[Loader] Incomplete or corrupted model detected:', f'{file_name}. Re-downloading...')
        _verified_download_paths.discard(str(cached_file))
        cached_file.unlink()
        need_download = True

    # 3. If the file is missing or incomplete,
    # initiate verified download
    if need_download:
        # Print the UI popup and console warnings EXACTLY ONCE here
        interpret_info('[Loader] Downloading:', url + ' → ' + str(cached_file))
        interpret('Please wait for the download to complete.')
        interpret_warn('Please wait for the download to complete. Progress can be checked in the console window.', silent=True)

        expected_size = get_remote_file_size(url)
        is_hf = 'huggingface.co' in url or (domain != 'https://huggingface.co' and domain in url)
        download_success = False

        # --- AUTOMATED HUGGING FACE LFS/XET INTERCEPTOR ---
        if is_hf:
            hf_parts = parse_hf_url(url, domain)
            if hf_parts is not None:
                repo_id, revision, repo_filename = hf_parts
                staging_dir = model_path_dir / f'.hf_tmp_{cached_file.stem}'
                try:
                    from huggingface_hub import hf_hub_download

                    hf_kwargs = {
                        'repo_id': repo_id,
                        'filename': repo_filename,
                        'revision': revision,
                        'local_dir': str(staging_dir)
                    }
                    if domain != 'https://huggingface.co':
                        hf_kwargs['endpoint'] = domain

                    dl_path = Path(hf_hub_download(**hf_kwargs))
                    actual_size = dl_path.stat().st_size if dl_path.is_file() else 0
                    size_ok = (actual_size > 0) and (expected_size is None or actual_size == expected_size)
                    st_ok = (cached_file.suffix.lower() != '.safetensors') or verify_safetensors_completeness(dl_path)

                    if size_ok and st_ok:
                        dl_path.replace(cached_file)
                        _verified_download_paths.add(str(cached_file))
                        interpret_info('[Loader] Successfully downloaded:', file_name)
                        print()
                        download_success = True
                    else:
                        interpret_warn('[Loader] Hugging Face download incomplete:', f'{file_name} ({actual_size:,} / {expected_size} bytes). Retrying...')
                except Exception:
                    interpret_info('[Loader] Using the standard downloader...')
                finally:
                    if staging_dir.exists():
                        shutil.rmtree(staging_dir, ignore_errors=True)

        # --- RESUMABLE FALLBACK DOWNLOADER ---
        if not download_success:
            download_success = download_with_resume_and_verify(
                url=url,
                cached_file=cached_file,
                expected_size=expected_size,
                progress=progress,
                max_retries=3
            )

        if not download_success:
            interpret('[Loader] Could not complete download:', str(cached_file))
            interpret('Download failed after 3 attempts. Please try again later or download manually from', url)
            print()
        else:
            # Rescan models on disk and update
            # filename list immediately after
            # the newdownload
            common.MODELS_INFO.refresh_from_path()
            update_files()

    return str(cached_file)


def get_write_directory(paths) -> Path:
    """Safely extracts the primary writable directory, falling back to current working dir if empty."""
    if isinstance(paths, list) and len(paths) > 0:
        return Path(paths[0])
    elif isinstance(paths, (str, Path)) and paths:
        return Path(paths)
    return Path.cwd()


def download_antelope_models():
    # load the Comfy  insightface directory,
    # used by the Flux PuLID and InstantID custom nodes
    # not currently used in FooocusPlus
    insightface_root = Path(getattr(common, 'path_insightface', Path.cwd() / 'models' / 'insightface'))
    model_dir = insightface_root / 'models' / 'antelopev2'

    antelope_files = [
        '1k3d68.onnx',
        '2d106det.onnx',
        'genderage.onnx',
        'glintr100.onnx',
        'scrfd_10g_bnkps.onnx'
    ]

    # Download all 5 files sequentially
    for file_name in antelope_files:
        load_file_from_url(
            url=f'https://huggingface.co/monas/InstantID/resolve/main/models/antelopev2/{file_name}',
            model_dir=str(model_dir),
            file_name=file_name
        )
    return str(model_dir)


def download_base_sd15_model():
    # Resolve the relative path dynamically from the global string
    global sd15_model_path
    rel_path = Path(sd15_model_path)

    # rel_path.parent -> 'SD1.5'
    # rel_path.name   -> 'realisticVisionV60B1_v51VAE.safetensors'
    model_path = get_write_directory(common.paths_checkpoints) / rel_path.parent
    model_file_name = rel_path.name

    # Download the base SD1.5 checkpoint
    # Required for both Fooocus and
    # Comfy IC-Light modes
    load_file_from_url(
        url='https://huggingface.co/moiu2998/mymo/resolve/3c3093fa083909be34a10714c93874ce5c9dabc4/realisticVisionV60B1_v51VAE.safetensors?download=true',
        model_dir=str(model_path),
        file_name=model_file_name
    )

    # Conditionally download the IC-Light
    # LDM UNet weights if Comfy is active
    if common.comfy_active:
        unet_dir = Path(common.path_unet)
        ic_light_unet_name = 'iclight_sd15_fc_unet_ldm.safetensors'

        load_file_from_url(
            url='https://huggingface.co/huchenlei/IC-Light-ldm/resolve/main/iclight_sd15_fc_unet_ldm.safetensors',
            model_dir=str(unet_dir),
            file_name=ic_light_unet_name
        )

    return str(model_path / model_file_name)


def download_bert_model():
    llms_root = Path(getattr(common, 'path_llms', Path.cwd() / 'models' / 'llms'))
    model_dir = llms_root / 'bert-base-uncased'
    file_name = 'model.safetensors'

    load_file_from_url(
        url='https://huggingface.co/google-bert/bert-base-uncased/resolve/main/model.safetensors',
        model_dir=str(model_dir),
        file_name=file_name
    )
    return str(model_dir / file_name)


def download_controlnet_canny():
    model_dir = get_write_directory(common.paths_controlnet)
    file_name = 'control-lora-canny-rank128.safetensors'
    load_file_from_url(
        url='https://huggingface.co/lllyasviel/misc/resolve/main/control-lora-canny-rank128.safetensors',
        model_dir=str(model_dir),
        file_name=file_name
    )
    return str(model_dir / file_name)


def download_controlnet_cpds():
    model_dir = get_write_directory(common.paths_controlnet)
    file_name = 'fooocus_xl_cpds_128.safetensors'
    load_file_from_url(
        url='https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_xl_cpds_128.safetensors',
        model_dir=str(model_dir),
        file_name=file_name
    )
    return str(model_dir / file_name)


def download_eva_clip_model():
    model_dir = get_write_directory(common.path_clip)
    file_name = 'EVA02_CLIP_L_336_psz14_s6B.pt'
    load_file_from_url(
        url='https://huggingface.co/QuanSun/EVA-CLIP/resolve/main/EVA02_CLIP_L_336_psz14_s6B.pt',
        model_dir=str(model_dir),
        file_name=file_name
    )
    return str(model_dir / file_name)


def download_faceid_lora():
    model_dir = get_write_directory(common.paths_loras)
    file_name = 'ip-adapter-faceid-plusv2_sdxl_lora.safetensors'

    load_file_from_url(
        url='https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plusv2_sdxl_lora.safetensors',
        model_dir=str(model_dir),
        file_name=file_name
    )
    return str(model_dir / file_name)


def download_groundingdino_model():
    # Resolve the physical destination directory
    model_dir = get_write_directory(common.paths_inpaint)
    file_name = 'groundingdino_swint_ogc.pth'

    load_file_from_url(
        url='https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth',
        model_dir=str(model_dir),
        file_name=file_name
    )
    return str(model_dir / file_name)


def download_inpaint_models(v):
    if not v:
        v = 'v2.6'

    model_dir = get_write_directory(common.paths_inpaint)

    load_file_from_url(
        url='https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/fooocus_inpaint_head.pth',
        model_dir=str(model_dir),
        file_name='fooocus_inpaint_head.pth'
    )
    head_file = str(model_dir / 'fooocus_inpaint_head.pth')
    patch_file = None

    if v == 'v2.5':
        patch_name = 'inpaint_v25.fooocus.patch'
        load_file_from_url(
            url='https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/inpaint_v25.fooocus.patch',
            model_dir=str(model_dir),
            file_name=patch_name
        )
        patch_file = str(model_dir / patch_name)
    else:   # i.e. v=='v2.6'
        patch_name = 'inpaint_v26.fooocus.patch'
        load_file_from_url(
            url='https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/inpaint_v26.fooocus.patch',
            model_dir=str(model_dir),
            file_name=patch_name
        )
        patch_file = str(model_dir / patch_name)

    return head_file, patch_file


def download_ip_adapters(v):
    assert v in ['ip', 'face']

    results = []
    clip_dir = get_write_directory(common.path_clip_vision)
    control_dir = get_write_directory(common.paths_controlnet)

    load_file_from_url(
        url='https://huggingface.co/lllyasviel/misc/resolve/main/clip_vision_vit_h.safetensors',
        model_dir=str(clip_dir),
        file_name='clip_vision_vit_h.safetensors'
    )
    results.append(str(clip_dir / 'clip_vision_vit_h.safetensors'))

    load_file_from_url(
        url='https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_ip_negative.safetensors',
        model_dir=str(control_dir),
        file_name='fooocus_ip_negative.safetensors'
    )
    results.append(str(control_dir / 'fooocus_ip_negative.safetensors'))

    if v == 'ip':
        file_name = 'ip-adapter-plus_sdxl_vit-h.bin'
        load_file_from_url(
            url='https://huggingface.co/lllyasviel/misc/resolve/main/ip-adapter-plus_sdxl_vit-h.bin',
            model_dir=str(control_dir),
            file_name=file_name
        )
        results.append(str(control_dir / file_name))

    if v == 'face':
        file_name = 'ip-adapter-plus-face_sdxl_vit-h.bin'
        load_file_from_url(
            url='https://huggingface.co/lllyasviel/misc/resolve/main/ip-adapter-plus-face_sdxl_vit-h.bin',
            model_dir=str(control_dir),
            file_name=file_name
        )
        results.append(str(control_dir / file_name))

    return results


def download_pulid_flux_model():
    # Resolve the physical destination directory
    model_dir = get_write_directory(common.paths_pulid)
    file_name = 'pulid_flux_v0.9.1.safetensors'

    load_file_from_url(
        url='https://huggingface.co/guozinan/PuLID/resolve/main/pulid_flux_v0.9.1.safetensors',
        model_dir=str(model_dir),
        file_name=file_name
    )
    return str(model_dir / file_name)


def download_safety_checker_model():
    model_dir = get_write_directory(common.path_safety_checker)
    file_name = 'stable-diffusion-safety-checker.bin'
    load_file_from_url(
        url='https://huggingface.co/mashb1t/misc/resolve/main/stable-diffusion-safety-checker.bin',
        model_dir=str(model_dir),
        file_name=file_name
    )
    return str(model_dir / file_name)


def download_sam_model(sam_model: str) -> str:
    match sam_model:
        case 'vit_b':
            return download_sam_vit_b()
        case 'vit_l':
            return download_sam_vit_l()
        case 'vit_h':
            return download_sam_vit_h()
        case _:
            raise ValueError(f'sam model {sam_model} does not exist.')


def download_sam_vit_b():
    model_dir = get_write_directory(common.path_sam)
    file_name = 'sam_vit_b_01ec64.pth'
    load_file_from_url(
        url='https://huggingface.co/mashb1t/misc/resolve/main/sam_vit_b_01ec64.pth',
        model_dir=str(model_dir),
        file_name=file_name
    )
    return str(model_dir / file_name)


def download_sam_vit_l():
    model_dir = get_write_directory(common.path_sam)
    file_name = 'sam_vit_l_0b3195.pth'
    load_file_from_url(
        url='https://huggingface.co/mashb1t/misc/resolve/main/sam_vit_l_0b3195.pth',
        model_dir=str(model_dir),
        file_name=file_name
    )
    return str(model_dir / file_name)


def download_sam_vit_h():
    model_dir = get_write_directory(common.path_sam)
    file_name = 'sam_vit_h_4b8939.pth'
    load_file_from_url(
        url='https://huggingface.co/mashb1t/misc/resolve/main/sam_vit_h_4b8939.pth',
        model_dir=str(model_dir),
        file_name=file_name
    )
    return str(model_dir / file_name)


def download_sdxl_lcm_lora():
    model_dir = get_write_directory(common.paths_loras)
    load_file_from_url(
        url='https://huggingface.co/lllyasviel/misc/resolve/main/sdxl_lcm_lora.safetensors',
        model_dir=str(model_dir),
        file_name=flags.PerformanceLoRA.Extreme_Speed.value
    )
    return flags.PerformanceLoRA.Extreme_Speed.value


def download_sdxl_lightning_lora():
    model_dir = get_write_directory(common.paths_loras)
    load_file_from_url(
        url='https://huggingface.co/mashb1t/misc/resolve/main/sdxl_lightning_4step_lora.safetensors',
        model_dir=str(model_dir),
        file_name=flags.PerformanceLoRA.Lightning.value
    )
    return flags.PerformanceLoRA.Lightning.value


def download_sdxl_hyper_sd_lora():
    model_dir = get_write_directory(common.paths_loras)
    load_file_from_url(
        url='https://huggingface.co/mashb1t/misc/resolve/main/sdxl_hyper_sd_4step_lora.safetensors',
        model_dir=str(model_dir),
        file_name=flags.PerformanceLoRA.Hyper_SD.value
    )
    return flags.PerformanceLoRA.Hyper_SD.value


def download_sdxl_vae() -> str:
    # Download the official SAI SDXL VAE
    # for Fooocus and Comfy workflows.
    model_dir = get_write_directory(common.path_vae)
    file_name = 'sdxl_vae.safetensors'
    load_file_from_url(
        url='https://huggingface.co/stabilityai/sdxl-vae/resolve/main/sdxl_vae.safetensors',
        model_dir=str(model_dir),
        file_name=file_name
    )
    return str(Path(model_dir) / file_name)


def download_siglip_vision_model():
    # Resolve the local clip_vision path
    # This file is currently not used but
    # is for Flux/SD3.5 Image Prompt
    model_dir = get_write_directory(common.path_clip_vision)
    file_name = 'sigclip_vision_patch14_384.safetensors'

    load_file_from_url(
        url='https://huggingface.co/Comfy-Org/siglip_vision_patch14_384/resolve/main/sigclip_vision_patch14_384.safetensors',
        model_dir=str(model_dir),
        file_name=file_name
    )
    return str(model_dir / file_name)


def download_superprompter_model():
    path_superprompter = get_write_directory(common.paths_llms) / 'superprompt-v1'
    load_file_from_url(
        url='https://huggingface.co/roborovski/superprompt-v1/resolve/main/model.safetensors',
        model_dir=str(path_superprompter),
        file_name='model.safetensors'
    )
    load_file_from_url(
        url='https://huggingface.co/roborovski/superprompt-v1/resolve/main/config.json',
        model_dir=str(path_superprompter),
        file_name='config.json'
    )
    load_file_from_url(
        url='https://huggingface.co/roborovski/superprompt-v1/resolve/main/generation_config.json',
        model_dir=str(path_superprompter),
        file_name='generation_config.json'
    )
    load_file_from_url(
        url='https://huggingface.co/roborovski/superprompt-v1/resolve/main/README.md',
        model_dir=str(path_superprompter),
        file_name='README.md'
    )
    load_file_from_url(
        url='https://huggingface.co/roborovski/superprompt-v1/resolve/main/spiece.model',
        model_dir=str(path_superprompter),
        file_name='spiece.model'
    )
    load_file_from_url(
        url='https://huggingface.co/roborovski/superprompt-v1/resolve/main/tokenizer.json',
        model_dir=str(path_superprompter),
        file_name='tokenizer.json'
    )
    load_file_from_url(
        url='https://huggingface.co/roborovski/superprompt-v1/resolve/main/tokenizer_config.json',
        model_dir=str(path_superprompter),
        file_name='tokenizer_config.json'
    )
    return str(path_superprompter / 'model.safetensors')


def download_ultrasharp_model():
    model_dir = get_write_directory(common.path_upscale_models)
    file_name = '4x-UltraSharp.pth'

    load_file_from_url(
        url='https://huggingface.co/lokcx/4x-Ultrasharp/resolve/main/4x-UltraSharp.pth',
        model_dir=str(model_dir),
        file_name=file_name
    )
    return str(model_dir / file_name)


def download_upscale_model():
    model_dir = get_write_directory(common.path_upscale_models)
    file_name = 'fooocus_upscaler_s409985e5.bin'
    load_file_from_url(
        url='https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_upscaler_s409985e5.bin',
        model_dir=str(model_dir),
        file_name=file_name
    )
    return str(model_dir / file_name)
