import json
import os
from typing import Optional, Tuple, List, Dict, Union

import argparse
# import clip
import open_clip as clip
import numpy as np
import openai_api
import pickle
import torch
import tqdm
import data_utils
import prompt
from json.decoder import JSONDecodeError
import string


if torch.cuda.is_available():
    dtype = torch.float16
else:
    dtype = torch.float32

@torch.no_grad()
def extract_image_features(device: torch.device, args: argparse.Namespace, dataset: torch.utils.data.Dataset, clip_model: clip.model.CLIP, batch_size: Optional[int] = 32,
                           num_workers: Optional[int] = 8, preload: str = None, **kwargs) -> Tuple[torch.Tensor, List[str]]:
    """
    Extracts image features from a dataset using a CLIP model.
    """
    if preload is not None and os.path.exists(preload):
        print(f'Loading precomputed image features from {preload}!')
        extracted_data = pickle.load(open(preload, 'rb'))
        index_features, index_names = extracted_data['index_features'], extracted_data['index_names']
        index_ranks = [] if 'index_ranks' not in extracted_data else extracted_data['index_ranks']
        aux_data = {} if 'aux_data' not in extracted_data else extracted_data['aux_data']
    else:
        loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size,
                                             num_workers=num_workers, pin_memory=True, collate_fn=data_utils.collate_fn)

        index_features, index_names, index_ranks, aux_data = [], [], [], []
        if 'genecis' in args.dataset:
            aux_data = {'ref_features': [], 'instruct_features': []}

        try:
            print(f"Extracting image features {dataset.__class__.__name__} - {dataset.split}")
        except Exception as e:
            pass

        # Extract features
        index_rank = None
        for batch in tqdm.tqdm(loader):
            if 'genecis' in args.dataset:      # reference, caption, blip_ref_img(-), gallery_and_target, 0
                _, n_gallery, _, h, w = batch[3].size()      # ([32, 15, 3, 224, 224])
                images = batch[3].view(-1, 3, h, w)
                names, index_rank = batch[1], batch[4]
                ref_images = batch[0]
                instructions = batch[1]
            else:
                images = batch.get('image')
                names = batch.get('image_name')
                if images is None: images = batch.get('reference_image')   # ？？
                if names is None: names = batch.get('reference_name')

            images = images.to(device)
            with torch.no_grad(), torch.cuda.amp.autocast():
                # if args.clip in ['ViT-bigG-14', 'ViT-B-32', 'ViT-B-16', 'ViT-L-14', 'ViT-H-14', 'ViT-g-14']:
                batch_features = clip_model.encode_image(images)
                # else:
                #     batch_features = extract_blip_image_features(images, processor, clip_model, device)
                index_features.append(batch_features.cpu())
                index_names.extend(names)
                if index_rank is not None:
                    index_ranks.extend(index_rank)
                if len(aux_data):
                    aux_data['ref_features'].append(clip_model.encode_image(ref_images.to(device)).cpu())
                    if hasattr(clip_model, 'tokenizer'):
                        aux_data['instruct_features'].append(clip_model.encode_text(
                            clip_model.tokenizer(instructions, context_length=77).to(device)).cpu())
                    else:
                        aux_data['instruct_features'].append(
                            clip_model.encode_text(clip.tokenize(instructions, context_length=77).to(device)).cpu())

        index_features = torch.vstack(index_features)

        if 'genecis' in args.dataset:
            # Reshape features into gallery-set for GeneCIS-style problems.
            index_features = index_features.view(-1, n_gallery, batch_features.size()[-1])
            index_ranks = torch.stack(index_ranks)
            aux_data['ref_features'] = torch.vstack(aux_data['ref_features'])
            aux_data['instruct_features'] = torch.vstack(aux_data['instruct_features'])

        if preload is not None:
            pickle.dump({'index_features': index_features, 'index_names': index_names, 'index_ranks': index_ranks,
                         'aux_data': aux_data}, open(preload, 'wb'))

    return index_features, index_names, index_ranks, aux_data


@torch.no_grad()
def generate_predictions(
        device: torch.device, args: argparse.Namespace, clip_model: clip.model.CLIP,
        query_dataset: torch.utils.data.Dataset, preload_dict: Dict[str, Union[str, None]], **kwargs
) -> Tuple[torch.Tensor, List[str], list]:
    """
    Generates features predictions for the validation set of CIRCO
    """
    ### Generate BLIP Image Captions.
    torch.cuda.empty_cache()
    batch_size = 32

    if preload_dict['captions'] is None or not os.path.exists(preload_dict['captions']):
        all_mod_captions, relative_captions = {}, []
        gt_img_ids, query_ids = [], []
        target_names, reference_names = [], []
        all_images, all_images_path = [], []

        query_loader = torch.utils.data.DataLoader(
            dataset=query_dataset, batch_size=batch_size, num_workers=8,
            pin_memory=False, collate_fn=data_utils.collate_fn, shuffle=False)
        query_iterator = tqdm.tqdm(query_loader, position=0, desc='Generating queries...')

        for batch in query_iterator:
            if 'genecis' in args.dataset:         # batch: reference, caption, ref_path, gallery_and_target, 0
                batch_image = batch[0]
                relative_captions.extend(batch[1])
                all_images.extend(batch[0].to(device))
                all_images_path.extend(batch[2])
            else:
                batch_image = batch['reference_image']
                all_images.extend(batch['reference_image'].to(device))
                all_images_path.extend(batch['reference_image_path'])
                reference_names.extend(batch['reference_name'])
                if 'fashioniq' not in args.dataset:
                    relative_captions.extend(batch['relative_caption'])
                else:
                    correction_dict = json.load(open(os.path.join('precomputed', 'correction_dict_{}.json'.format(args.dataset)), 'r'))
                    rel_caps = batch['relative_captions']
                    rel_caps = np.array(rel_caps).T.flatten().tolist()
                    relative_captions.extend([
                        f"{correct_text(rel_caps[i], correction_dict).strip('.?, ')} and {rel_caps[i + 1].strip('.?, ')}" for i in
                        range(0, len(rel_caps), 2)
                    ])

                if 'target_name' in batch:
                    target_names.extend(batch['target_name'])

                gt_key = 'gt_img_ids'
                if 'group_members' in batch:
                    gt_key = 'group_members'
                if gt_key in batch:
                    gt_img_ids.extend(np.array(batch[gt_key]).T.tolist())

                query_key = 'query_id'
                if 'pair_id' in batch:
                    query_key = 'pair_id'
                if query_key in batch:
                    query_ids.extend(batch[query_key])

            query_iterator.set_postfix_str(f'Shape: {batch_image.size()}')



        # ===================================== Query Generation by LLM =====================================
        print("LLM working ...")
        img_captions = json.load(open(preload_dict['image_captions'], 'r', encoding="utf-8"))
        query_phrase_list = []
        for i in tqdm.trange(len(all_images), position=1, desc='LLM working ...', leave=False):
            query_prompt = prompt.query_generation_prompt + '\n' + "Reference Image:" + img_captions[str(reference_names[i])]
            query_prompt = query_prompt + '\n' + "Modification Text:" + relative_captions[i]
            resp_query = openai_api.openai_completion(prompt=query_prompt.strip())
            if resp_query:
                resp_query_lines = resp_query.split('\n')
                for line in resp_query_lines:
                    if line.strip().startswith('Edited Description:'):
                        query_content = line.split(':')[1].strip()
                        query_item = {
                            "candidate": str(reference_names[i]),
                            "query": query_content
                        }
                        query_phrase_list.append(query_item)
                        print(f"Processing completed: {reference_names[i]} - ")
                        break
            else:
                query_item = {
                    "candidate": str(reference_names[i]),
                    "query": 'None'
                }
                query_phrase_list.append(query_item)
                print(f"Processing completed: {reference_names[i]} - ")

        json.dump(query_phrase_list, open(preload_dict['phrase'], 'w'), indent=2)

    else:
        print(f'Loading precomputed image captions from {preload_dict["captions"]}!')
        res_dict = json.load(open(preload_dict['captions'], 'r', encoding="utf-8"))
        reference_names, target_names, all_mod_captions, gt_img_ids, relative_captions, query_ids = res_dict.values()

    # ===================================== Decompose MLLM outputs =====================================
    if preload_dict['mods'] is None or not os.path.exists(preload_dict['mods']):
        mod_captions = []
        for i in tqdm.trange(len(all_mod_captions), position=1, desc=f'Disentangle Target Image Description ...', leave=False):
            mod_caption = all_mod_captions[i]
            description = extract_target_description(mod_caption)
            if description == "":
                mod_captions.append(relative_captions[i])
            else:
                mod_captions.append(description)

        if preload_dict['mods'] is not None:
            dump_dict = {'queries': mod_captions}
            json.dump(dump_dict, open(preload_dict['mods'], 'w'), indent=2)
    else:
        print(f'Loading precomputed caption modifiers from {preload_dict["mods"]}!')
        mod_captions = json.load(open(preload_dict['mods'], 'r'))['queries']

    ### Perform text-to-image retrieval based on the modified captions.
    predicted_features = text_encoding(device, clip_model, mod_captions, batch_size=batch_size)
    # print("predicted_features.size()", predicted_features.size())

    print('Loading query phrase file ...')
    mod_phrases = get_all_queries(preload_dict['phrase'])
    phrase_features = text_encoding(device, clip_model, mod_phrases, batch_size=batch_size)


    return {
        'reference_names': reference_names,
        'target_names': target_names,
        'predicted_features': predicted_features,
        'phrase_features': phrase_features,
        'targets': gt_img_ids,
        'query_ids': query_ids,
        'mod_captions': mod_captions
    }


def text_encoding(device, clip_model, input_captions, batch_size=32):
    n_iter = int(np.ceil(len(input_captions) / batch_size))
    predicted_features = []

    for i in tqdm.trange(n_iter, position=0, desc='Encoding captions...'):
        captions_to_use = input_captions[i * batch_size:(i + 1) * batch_size]

        # if args.clip in ['ViT-bigG-14', 'ViT-B-32', 'ViT-B-16', 'ViT-L-14', 'ViT-H-14', 'ViT-g-14']:
        if hasattr(clip_model, 'tokenizer'):
            tokenized_input_captions = clip_model.tokenizer(captions_to_use, context_length=77).to(device)
        else:
            tokenized_input_captions = clip.tokenize(captions_to_use, context_length=77, truncate=True).to(device)
        clip_text_features = clip_model.encode_text(tokenized_input_captions)

        predicted_features.append(clip_text_features)
    predicted_features = torch.vstack(predicted_features)

    return torch.nn.functional.normalize(predicted_features, dim=-1)



def extract_target_description(json_str):
    try:
        clean_str = json_str.strip().lstrip('\ufeff').rstrip('\0')

        if clean_str.startswith("```json"):
            clean_str = clean_str.replace("```json", "").replace("```", "").strip()

        if "{" in clean_str and "}" in clean_str:
            start_idx = clean_str.find("{")
            end_idx = clean_str.rfind("}") + 1
            clean_str = clean_str[start_idx:end_idx].strip()

        data = json.loads(clean_str)
        target_desc = data.get("Target Image Description", "")

        return target_desc.strip()

    except JSONDecodeError as e:
        return ""
    except Exception as e:
        return ""


def get_all_queries(file_path="A.json"):
    query_list = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            phrase_dict = json.load(f)

        if isinstance(phrase_dict, list):
            for item in phrase_dict:
                if isinstance(item, dict) and "query" in item:
                    query_list.append(item["query"])
        return query_list

    except FileNotFoundError:
        print(f"Cannot find file {file_path}")
        return []
    except json.JSONDecodeError:
        print("A.json illegal")
        return []
    except Exception as e:
        print(f"Error: {str(e)}")
        return []


def correct_text(text, correction_dict):
    trans = str.maketrans({key: ' ' for key in string.punctuation})
    tokens = str(text).lower().translate(trans).strip().split()
    text = " ".join([correction_dict.get(word) if word in correction_dict else word for word in tokens])

    return text
