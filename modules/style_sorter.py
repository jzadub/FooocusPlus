import gradio as gr
import json
from pathlib import Path

import modules.localization as localization
from args_manager import args as cli_args
from enhanced.translator import interpret

all_styles = []
SORT_FILE = Path('sorted_styles.json')
V2_STYLE = "Fooocus V2"


def _anchor_v2(styles_list):
    # Internal helper to ensure Fooocus V2
    # is always at index 0
    if V2_STYLE in styles_list:
        styles_list.remove(V2_STYLE)
        return [V2_STYLE] + styles_list
    return styles_list


def try_load_sorted_styles(style_names, default_selected):
    global all_styles
    all_styles = style_names

    try:
        if SORT_FILE.exists():
            with SORT_FILE.open('r', encoding='utf-8') as fp:
                loaded_data = json.load(fp)
                sorted_styles = [x for x in loaded_data if x in all_styles]

                # Append any new styles found in all_styles that weren't in the JSON
                for x in all_styles:
                    if x not in sorted_styles:
                        sorted_styles.append(x)

                all_styles = sorted_styles
    except Exception as e:
        interpret('[Sorter] Load style sorting failed:',  e)

    # Move defaults to top, but then force V2 to the absolute top
    unselected = [y for y in all_styles if y not in default_selected]
    all_styles = _anchor_v2(default_selected + unselected)
    return


def sort_styles(selected):
    global all_styles

    # Maintain the user's "selection-first" sorting
    unselected = [y for y in all_styles if y not in selected]

    # Anchor Fooocus V2 to the top of the entire list
    sorted_styles = _anchor_v2(selected + unselected)

    try:
        with SORT_FILE.open('w', encoding='utf-8') as fp:
            json.dump(sorted_styles, fp, indent=4)
    except Exception as e:
        interpret('[Sorter] Failed to save style sort:', e)

    all_styles = sorted_styles
    return gr.CheckboxGroup.update(choices=sorted_styles)


def localization_key(x):
    return x + localization.current_translation.get(x, '')


def search_styles(selected, query):
    unselected = [y for y in all_styles if y not in selected]
    matched = [y for y in unselected if query.lower() in localization_key(y).lower()] if len(query.replace(' ', '')) > 0 else []
    unmatched = [y for y in unselected if y not in matched]
    sorted_styles = matched + selected + unmatched
    return gr.CheckboxGroup.update(choices=sorted_styles)


def get_hidden_styles_path():
    user_dir = Path(cli_args.user_dir)
    user_dir.mkdir(parents=True, exist_ok=True)
    return user_dir / 'hidden_styles.json'


def load_hidden_styles():
    file_path = get_hidden_styles_path()
    if not file_path.exists():
        return []
    try:
        with file_path.open('r', encoding='utf-8') as fp:
            data = json.load(fp)
            if isinstance(data, list):
                return [str(x) for x in data if isinstance(x, str)]
    except Exception as e:
        interpret('[Sorter] Failed to load hidden styles:', e)
    return []


def save_hidden_styles(hidden_list_json):
    if not hidden_list_json:
        return
    try:
        hidden_data = json.loads(hidden_list_json)
        if not isinstance(hidden_data, list):
            return
        file_path = get_hidden_styles_path()
        with file_path.open('w', encoding='utf-8') as fp:
            json.dump(sorted(list(set(hidden_data))), fp, indent=4, ensure_ascii=False)
    except Exception as e:
        interpret('[Sorter] Failed to save hidden styles:', e)
    return
