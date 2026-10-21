import os
from typing import List, Dict

import argparse
# import clip
import open_clip as clip
import open_clip
# import lavis
# import numpy as np
import termcolor
import torch
torch.multiprocessing.set_sharing_strategy('file_system')

# import compute_results
import data_utils
import datasets
# import prompts
import utils

from transformers import Blip2Processor, Blip2ForConditionalGeneration


def main():
    ### Load Input Arguments.
    parser = argparse.ArgumentParser()
    # Base Arguments
    parser.add_argument("--exp-name", type=str, help="Experiment to evaluate")
    parser.add_argument("--device", type=int, default=0, help="GPU ID to use.")
    parser.add_argument("--preload", nargs='*', type=str, default=['img_features','captions','mods','phrase','image_captions'],
                        help='List of properties to preload is computed once before.')
    # Base Model Choices
    parser.add_argument("--clip", type=str, default='ViT-B/32',
                        choices=['ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'RN50x4', 'ViT-bigG-14',
                                 'ViT-B-32','ViT-B-16','ViT-L-14','ViT-H-14','ViT-g-14',
                                 'blip-itm-base-coco', 'blip-itm-large-coco', 'blip-image-captioning-large', 'blip-itm-large-flicker'],
                        help="Which CLIP text-to-image retrieval model to use"),
    parser.add_argument("--blip", type=str, default='gpt-4o', choices=['gpt-4o-mini', 'gpt-4o', 'o4-mini'], help="MLLM to use.")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=['cirr', 'circo',
                                 'fashioniq_dress', 'fashioniq_toptee', 'fashioniq_shirt',
                                 'genecis_change_attribute', 'genecis_change_object', 'genecis_focus_attribute', 'genecis_focus_object'],
                        help="Dataset to use")
    parser.add_argument("--split", type=str, default='val', choices=['val', 'test'],
                        help='Dataset split to evaluate on. Some datasets require special testing protocols s.a. cirr/circo.')
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--preprocess-type", default="targetpad", type=str, choices=['clip', 'targetpad'], help="Preprocess pipeline to use")
    # MLLM Prompt Arguments.
    # available_prompts = [f'prompts.{x}' for x in prompts.__dict__.keys() if '__' not in x]
    # parser.add_argument("--llm_prompt", default='prompts.simple_modifier_prompt', type=str, choices=available_prompts,
    #                     help='Denotes the base prompt to use to probe the LLM. Has to be available in prompts.py')
    parser.add_argument("--batch_size", default=32, type=int, help='Batch size to use.')
    args = parser.parse_args()

    ### Set Device.
    termcolor.cprint(f'Starting evaluation on {args.dataset.upper()} (split: {args.split})\n', color='green', attrs=['bold'])
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    ### Argument Checks.
    preload_dict = {key: None for key in ['img_features', 'captions', 'mods', 'phrase', 'image_captions']}      # Build loading and saving paths
    preload_str = f'{args.dataset}_{args.blip}_{args.clip}_{args.split}'.replace('/', '-')

    if len(args.preload):
        os.makedirs('precomputed', exist_ok=True)
    if 'img_features' in args.preload:
        # # CLIP embeddings only have to be computed when CLIP model changes.
        preload_dict['img_features'] = os.path.join('precomputed', preload_str + '_img_features.pkl')   # to be retrieved

    if 'captions' in args.preload:
        # # BLIP captions only have to be computed when BLIP model or BLIP prompt changes.
        caption_load_str = f'{args.dataset}_{args.blip}_{args.split}'.replace('/', '-')
        preload_dict['captions'] = os.path.join('precomputed', caption_load_str + '_captions.json')

    if 'mods' in args.preload:
        # # LLM-based caption modifications have to be queried only when BLIP model or BLIP prompt changes.
        mod_load_str = f'{args.dataset}_{args.blip}_{args.split}'.replace('/', '-')
        preload_dict['mods'] = os.path.join('precomputed', mod_load_str + f'_mods.json')

    phrase_load_str = f'query_phrase_{args.dataset}_{args.split}'
    preload_dict['phrase'] = os.path.join('precomputed', phrase_load_str + '.json')

    img_caption_load_str = f'{args.dataset}_{args.blip}_{args.split}'.replace('/', '-')
    preload_dict['image_captions'] = os.path.join('precomputed', img_caption_load_str + '_image_captions.json')


    if args.split == 'test':
        preload_dict['test'] = preload_str + f'_test_submission.json'

    print("Preload_dicts have been completed ...")

    ### Load CLIP model, BLIP model & Preprocessing.
    print(f'Loading CLIP {args.clip}... ', end='')

    if args.clip in ['ViT-bigG-14', 'ViT-B-32', 'ViT-B-16', 'ViT-L-14', 'ViT-H-14', 'ViT-g-14']:
        # import open_clip
        pretraining_path = {
            'ViT-B-32': '/root/autodl-tmp/Models/laionCLIP-ViT-B-32-laion2B-s34B-b79K/open_clip_pytorch_model.bin',
            'ViT-L-14': '/root/autodl-tmp/Models/laionCLIP-ViT-L-14-laion2B-s32B-b82K/open_clip_pytorch_model.bin',
            'ViT-g-14': '/root/autodl-tmp/Models/laionCLIP-ViT-g-14-laion2B-s34B-b88K/open_clip_pytorch_model.bin',
            'ViT-B-16': 'laion2b_s34b_b88k',
            'ViT-H-14': 'laion2b_s32b_b79k',
            'ViT-bigG-14': 'laion2b_s39b_b160k'
        }
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(args.clip, pretrained=pretraining_path[args.clip])
        clip_model = clip_model.eval().requires_grad_(False).to(device)
        tokenizer = open_clip.get_tokenizer(args.clip)
        clip_model.tokenizer = tokenizer
    else:
        clip_model, clip_preprocess = clip.load(args.clip, device=device, jit=False)
        clip_model = clip_model.float().eval().requires_grad_(False).to(device)


    print('Done.')

    print('Loading BLIP-2 ...')

    model_path = "/root/autodl-tmp/Models/salesforce-blip2-flan-t5-xl"

    blip_processor = Blip2Processor.from_pretrained(
        model_path,
        local_files_only=True,
        use_fast=False
    )

    blip_model = Blip2ForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map="auto",  # recommended for large BLIP-2
        local_files_only=True
    )

    blip_model.eval()



    # Data preprocessing
    if args.preprocess_type == 'targetpad':
        print('Target pad preprocess pipeline is used.')
        preprocess = data_utils.targetpad_transform(1.25, clip_preprocess.transforms[0].size)
    elif args.preprocess_type == 'clip':
        print('CLIP preprocess pipeline is used.')
        preprocess = clip_preprocess


    ### Load Evaluation Datasets.
    target_datasets, query_datasets, pairings = [], [], []

    if 'fashioniq' in args.dataset.lower():
        dress_type = args.dataset.split('_')[-1]
        target_datasets.append(datasets.FashionIQDataset(args.dataset_path, args.split, [dress_type], 'classic', preprocess, blip_processor))
        query_datasets.append(datasets.FashionIQDataset(args.dataset_path, args.split, [dress_type], 'relative', preprocess, blip_processor))
        pairings.append(dress_type)
        # compute_results_function = compute_results.fiq

    elif args.dataset.lower() == 'cirr':
        split = 'test1' if args.split == 'test' else args.split
        target_datasets.append(datasets.CIRRDataset(args.dataset_path, split, 'classic', preprocess))
        query_datasets.append(datasets.CIRRDataset(args.dataset_path, split, 'relative', preprocess))
        # compute_results_function = compute_results.cirr
        pairings.append('default')

    elif args.dataset.lower() == 'circo':
        target_datasets.append(datasets.CIRCODataset(args.dataset_path, args.split, 'classic', preprocess, blip_processor))
        query_datasets.append(datasets.CIRCODataset(args.dataset_path, args.split, 'relative', preprocess, blip_processor))
        # compute_results_function = compute_results.circo
        pairings.append('default')

    elif 'genecis' in args.dataset.lower():
        # prop_file = '_'.join(args.dataset.lower().split('_')[1:])
        prop_file = 'supplement'
        prop_file = os.path.join(args.dataset_path, 'genecis', prop_file + '.json')
        if 'object' in args.dataset.lower():
            datapath = os.path.join(args.dataset_path, 'coco2017', 'val2017')
            genecis_dataset = datasets.COCOValSubset(root_dir=datapath, val_split_path=prop_file, transform=preprocess)
        elif 'attribute' in args.dataset.lower():
            datapath = os.path.join(args.dataset_path, 'Visual_Genome', 'VG_All')
            genecis_dataset = datasets.VAWValSubset(image_dir=datapath, val_split_path=prop_file, transform=preprocess)
        target_datasets.append(genecis_dataset)
        query_datasets.append(genecis_dataset)
        # compute_results_function = compute_results.genecis
        pairings.append('default')


    ### Evaluate performances.
    for query_dataset, target_dataset, pairing in zip(query_datasets, target_datasets, pairings):
        termcolor.cprint(f'\n------ Evaluating Retrieval Setup: {pairing}', color='yellow', attrs=['bold'])

        ### General Input Arguments.
        input_kwargs = {
            'args': args, 'query_dataset': query_dataset, 'target_dataset': target_dataset, 'clip_model': clip_model,
            'preprocess': preprocess, 'device': device, 'split': args.split, 'preload_dict': preload_dict,
            'blip_model': blip_model, 'blip_processor': blip_processor
        }

        print(f'Generating conditional query predictions (CLIP: {args.clip}, BLIP: {args.blip}).')
        out_dict = utils.generate_predictions(**input_kwargs)
        input_kwargs.update(out_dict)



if __name__ == '__main__':
    main()