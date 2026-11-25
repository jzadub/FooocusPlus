"""
LoRA trigger word lookup for the LoRA panel in the Models tab.

Where trigger words are looked for, in order:
  1. <lora>.txt              a plain text file you write yourself (manual override)
  2. <lora>.civitai.info     written by Civitai Helper / Forge / A1111 extensions
  3. <lora>.metadata.json    written by Stability Matrix / LoRA Manager
  4. <lora>.json             A1111 "activation text"
  5. "modelspec.trigger_phrase" stored inside the .safetensors file
  6. Earlier Civitai lookups, cached in UserDir/lora_trigger_cache.json
  7. Civitai lookup by file hash (one time per file, then cached)
  8. Most frequent training tags stored inside the .safetensors file
"""

import json
import os
import struct
import threading

import common
import modules.util as util

ENABLE_CIVITAI_LOOKUP = True
CIVITAI_URL = 'https://civitai.com/api/v1/model-versions/by-hash/'
MAX_TRAINING_TAGS = 8

_lock = threading.Lock()
_cache = None


def _cache_path():
    from modules.config import user_dir
    return os.path.join(str(user_dir), 'lora_trigger_cache.json')


def _load_cache():
    global _cache
    if _cache is None:
        try:
            with open(_cache_path(), 'r', encoding='utf-8') as f:
                _cache = json.load(f)
        except Exception:
            _cache = {}
    return _cache


def _save_cache():
    try:
        with open(_cache_path(), 'w', encoding='utf-8') as f:
            json.dump(_cache, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f'[LoRA Triggers] Could not save cache: {e}')


def _file_key(path):
    st = os.stat(path)
    return f'{path}|{st.st_size}|{int(st.st_mtime)}'


def _join(words):
    if isinstance(words, str):
        words = [words]
    cleaned = []
    for w in words or []:
        w = str(w).strip().strip(',').strip()
        if w and w not in cleaned:
            cleaned.append(w)
    return ', '.join(cleaned)


def _read_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def _sidecar_words(lora_path):
    stem = os.path.splitext(lora_path)[0]

    txt = stem + '.txt'
    if os.path.isfile(txt):
        try:
            with open(txt, 'r', encoding='utf-8', errors='ignore') as f:
                text = ' '.join(line.strip() for line in f if line.strip())[:500]
            if text:
                return text, 'your .txt file'
        except Exception:
            pass

    data = _read_json(stem + '.civitai.info')
    if isinstance(data, dict) and _join(data.get('trainedWords')):
        return _join(data.get('trainedWords')), 'Civitai info file'

    data = _read_json(stem + '.metadata.json')
    if isinstance(data, dict):
        words = data.get('trainedWords') or (data.get('civitai') or {}).get('trainedWords')
        if _join(words):
            return _join(words), 'metadata file'

    data = _read_json(stem + '.json')
    if isinstance(data, dict) and str(data.get('activation text', '')).strip():
        return str(data['activation text']).strip(), 'A1111 json file'

    return None, None


def _safetensors_metadata(lora_path):
    if not lora_path.lower().endswith('.safetensors'):
        return {}
    try:
        with open(lora_path, 'rb') as f:
            header_len = struct.unpack('<Q', f.read(8))[0]
            if header_len > 100 * 1024 * 1024:
                return {}
            header = json.loads(f.read(header_len))
        return header.get('__metadata__', {}) or {}
    except Exception:
        return {}


def _training_tags(meta):
    raw = meta.get('ss_tag_frequency')
    if not raw:
        return None
    try:
        freq = json.loads(raw) if isinstance(raw, str) else raw
        totals = {}
        for tags in freq.values():
            for tag, count in tags.items():
                tag = tag.strip()
                if tag:
                    totals[tag] = totals.get(tag, 0) + int(count)
        top = sorted(totals.items(), key=lambda kv: -kv[1])[:MAX_TRAINING_TAGS]
        return _join([t for t, _ in top]) or None
    except Exception:
        return None


def _civitai_words(lora_path):
    import requests
    file_hash = util.sha256(lora_path)  # AutoV2 hash, accepted by Civitai
    try:
        r = requests.get(CIVITAI_URL + file_hash, timeout=10)
        if r.status_code == 404:
            return ''
        r.raise_for_status()
        return _join(r.json().get('trainedWords'))
    except Exception as e:
        print(f'[LoRA Triggers] Civitai lookup failed for {os.path.basename(lora_path)}: {e}')
        return None  # network problem: don't cache, try again next time


def get_trigger_words(lora_name, allow_network=True):
    """Returns (text, source). text is '' when nothing was found."""
    if not lora_name or lora_name == 'None':
        return '', ''

    lora_path = util.get_file_from_folder_list(lora_name, common.paths_loras)
    if not os.path.isfile(lora_path):
        return '', 'file not found'

    words, source = _sidecar_words(lora_path)
    if words:
        return words, source

    meta = _safetensors_metadata(lora_path)
    if str(meta.get('modelspec.trigger_phrase', '')).strip():
        return meta['modelspec.trigger_phrase'].strip(), 'embedded in file'

    with _lock:
        cache = _load_cache()
        key = _file_key(lora_path)
        if key in cache:
            words = cache[key]
        elif allow_network and ENABLE_CIVITAI_LOOKUP:
            words = _civitai_words(lora_path)
            if words is not None:
                cache[key] = words
                _save_cache()
        else:
            words = None

    if words:
        return words, 'Civitai'

    tags = _training_tags(meta)
    if tags:
        return tags, 'most used training tags'

    return '', 'none found'


def trigger_box_update(lora_name, allow_network=True):
    import gradio as gr
    if not lora_name or lora_name == 'None':
        return gr.update(value='', visible=False)
    words, source = get_trigger_words(lora_name, allow_network)
    label = f'Trigger words ({source})' if source else 'Trigger words'
    return gr.update(value=words or 'No trigger words found', label=label, visible=True)


def initial_value(lora_name):
    """Startup value: local sources and cache only, no network."""
    if not lora_name or lora_name == 'None':
        return '', 'Trigger words', False
    words, source = get_trigger_words(lora_name, allow_network=False)
    label = f'Trigger words ({source})' if source else 'Trigger words'
    return words or 'No trigger words found', label, True
