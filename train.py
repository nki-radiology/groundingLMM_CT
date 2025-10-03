"""
train.py - GLaMM Model Training on Mixed Datasets

Trains the GLaMM model using Caption, Region, and Segmentation datasets with a random sampling approach. This method
is crucial for developing a versatile model capable of handling diverse applications effectively.
"""
import os
import sys
import time
import tqdm
import random
import gc
import torch
import re
import argparse
import json
import signal
import deepspeed
import numpy as np
import transformers
from functools import partial
from torch.utils.data import ConcatDataset
from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter
from model.surface_distance.surface_distance import metrics
from scipy.ndimage import binary_erosion, distance_transform_edt
from model.GLaMM import GLaMMForCausalLM
from model.llava import conversation as conversation_lib

from dataset.dataset import custom_collate_fn, HybridSegDataset, HybridRegDataset, HybridCapDataset
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, AverageMeter, ProgressMeter, dict_to_cuda,
                         Summary, intersectionAndUnionGPU, computeDiceCoefficient, computeNSD)

from dataset.segm_datasets.RefCOCO_Segm_ds import ReferSegmDataset
from dataset.region_datasets.RefCOCO_VG_Region_ds import RefCocoGRegDataset, VisualGenomeRegDataset
from dataset.caption_datasets.COCO_Caption_ds import CocoCapDataset
from dataset.segm_datasets.Semantic_Segm_ds import SemanticSegmDataset
from dataset.gcg_datasets.GranDf_gcg_ds import GranDfDataset, OpenPsgGCGDataset, Flickr30kGCGDataset, RefCOCOgGCGDataset
from dataset.gcg_datasets.GranDf_gcg_ds import OpenPsgGCGDataset, Flickr30kGCGDataset, RefCOCOgGCGDataset

def signal_handler(sig, frame):
    print('Received signal:', sig)
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def parse_args(args):
    parser = argparse.ArgumentParser(description="GLaMM Model Training")

    # Model-specific settings
    parser.add_argument("--version", default="MBZUAI/GLaMM-GranD-Pretrained")
    parser.add_argument("--vision_pretrained", default="./checkpoints/sam_vit_h_4b8939.pth", type=str)
    parser.add_argument("--vision-tower", default="openai/clip-vit-large-patch14-336", type=str)
    parser.add_argument("--conv_type", default="llava_v1", type=str, choices=["llava_v1", "llava_llama_2"])
    parser.add_argument("--tune_mm_mlp_adapter", action="store_true")
    parser.add_argument("--freeze_mm_mlp_adapter", action="store_true")
    parser.add_argument("--mm_use_im_start_end", action="store_true", default=True)
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--image_size", default=1024, type=int, help="Image size for grounding image encoder")
    parser.add_argument("--model_max_length", default=1536, type=int)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj", type=str)
    parser.add_argument("--with_region", action="store_true", default=True)
    parser.add_argument("--mm_vision_select_layer", default=-2, type=int)
    parser.add_argument("--pretrain_mm_mlp_adapter", default="", type=str)
    parser.add_argument("--precision", default='bf16', type=str)

    # Dataset settings
    parser.add_argument("--use_cap_data", action="store_true", help="Use caption data")
    parser.add_argument("--use_reg_data", action="store_true", help="Use region data")
    parser.add_argument("--use_segm_data", action="store_true", help="Use segmentation data")
    parser.add_argument("--weight_cap", default=0.15, type=float, help="Sampling weight for caption data")
    parser.add_argument("--weight_reg", default=0.40, type=float, help="Sampling weight for region data")
    parser.add_argument("--weight_segm", default=0.45, type=float, help="Sampling weight for segmentation data")
    parser.add_argument("--dataset_dir", default="./data", type=str)
    parser.add_argument("--seg_dataset", default="Semantic_Segm||Refer_Segm||RefCoco_GCG||PSG_GCG||Flickr_GCG||GranDf_GCG",
                        type=str, help="Choose from: Semantic_Segm, Refer_Segm, RefCoco_GCG, GranDf_GCG, PSG_GCG, Flickr_GCG")
    parser.add_argument("--segm_sample_rates", default="5,4,3,3,3,1", type=str)
    parser.add_argument("--reg_dataset", default="RefCoco_Reg||RefCocoG_Reg||RefCocoP_Reg||VisGen_Reg",
                        type=str, help="Choose from: RefCoco_Reg, RefCocoG_Reg, RefCocoP_Reg, VisGen_Reg, Flickr_Reg")
    parser.add_argument("--reg_sample_rates", default="1,1,1,1", type=str)
    parser.add_argument("--cap_dataset", default="CocoCap||LLaVaInstruct", type=str, help="Choose from: CocoCap, LLaVaInstruct")
    parser.add_argument("--cap_sample_rates", default="1,1", type=str)
    parser.add_argument("--semantic_segm_data", default="pancancer||pancancer_lesions", type=str)
    parser.add_argument("--refer_segm_data", default="refcoco||refcoco+||refcocog||refclef", type=str)
    parser.add_argument("--vqa_data", default="llava_instruct_150k", type=str)
    parser.add_argument("--num_classes_per_sample", default=3, type=int)

    # Training settings
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--auto_resume", action="store_true")
    parser.add_argument("--weight", default="", type=str)
    parser.add_argument("--lr", default=0.0003, type=float)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--steps_per_epoch", default=500, type=int)
    parser.add_argument("--batch_size", default=2, type=int, help="batch size per device per step")
    parser.add_argument("--grad_accumulation_steps", default=10, type=int)
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--workers", default=2, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--dice_loss_weight", default=0.5, type=float)
    parser.add_argument("--bce_loss_weight", default=2.0, type=float)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.95, type=float)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--train_mask_decoder", action="store_true", default=True)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")

    # Evaluation settings
    parser.add_argument("--val_dataset", default="CocoCapVal|RefCOCOgRegVal|RefCOCOgSegmVal", type=str,
                        help="Choose from: CocoCapVal, RefCOCOgRegVal, VisGenomeRegVal, RefCOCOgSegmVal, PsgGCGVal, "
                             "RefCocoGCGVal, FlickrGCGVal")
    parser.add_argument("--mask_validation", action="store_true")
    parser.add_argument("--class_validation", action="store_true")
    parser.add_argument("--no_eval", action="store_true")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--custom_resume", action="store_true")
    parser.add_argument("--resume_name", default="", type=str)

    # Experiment settings
    parser.add_argument("--log_base_dir", default="./output", type=str)
    parser.add_argument("--exp_name", default="GlamFinetuneOS", type=str)

    return parser.parse_args(args)


def initialize_environment(args):
    """ Set up logging and model directories. """
    args.log_dir = os.path.join(args.log_base_dir, args.exp_name)
    if args.local_rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        return SummaryWriter(args.log_dir)
    return None


def setup_tokenizer_and_special_tokens(args):
    """ Load tokenizer and add special tokens. """
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version, model_max_length=args.model_max_length, padding_side="right", use_fast=False
    )
    print('\033[92m' + "---- Initialized tokenizer from: {} ----".format(args.version) + '\033[0m')
    tokenizer.pad_token = tokenizer.unk_token

    if not args.pretrained:
        if args.use_mm_start_end:
            tokenizer.add_tokens(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
            )
        # modifications specific for regions
        reg_tokens = ['<bbox>', '<point>']
        # Adding special tokens for pixel grounding
        segmentation_tokens = ['[SEG]']
        # Adding tokens for GCG
        phrase_tokens = ['<p>', '</p>']
        special_tokens = reg_tokens + segmentation_tokens + phrase_tokens
        tokenizer.add_tokens(special_tokens, special_tokens=True)

    args.bbox_token_idx = tokenizer("<bbox>", add_special_tokens=False).input_ids[0]
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    args.bop_token_idx = tokenizer("<p>", add_special_tokens=False).input_ids[0]
    args.eop_token_idx = tokenizer("</p>", add_special_tokens=False).input_ids[0]

    return tokenizer


def initialize_model(args, tokenizer):
    """ Initialize the GLaMM model. """
    model_args = {k: getattr(args, k) for k in
                  ["train_mask_decoder", "out_dim", "ce_loss_weight", "dice_loss_weight", "bce_loss_weight",
                   "seg_token_idx", "vision_pretrained", "vision_tower", "use_mm_start_end", "mm_vision_select_layer",
                   "pretrain_mm_mlp_adapter", "tune_mm_mlp_adapter", "freeze_mm_mlp_adapter", "mm_use_im_start_end",
                   "with_region", "bbox_token_idx", "eop_token_idx", "bop_token_idx"]}
    model_args["num_level_reg_features"] = 4

    model = GLaMMForCausalLM.from_pretrained(
        args.version, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, **model_args
    )
    print('\033[92m' + "---- Initialized model from: {} ----".format(args.version) + '\033[0m')

    # Configure model tokens
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    return model


def prepare_model_for_training(model, tokenizer, args):
    # Enable input gradients
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    # Initialize vision tower
    print(
        '\033[92m' + "---- Initialized Global Image Encoder (vision tower) from: {} ----".format(
            args.vision_tower
        ) + '\033[0m'
    )
    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch.bfloat16, device=args.local_rank)

    # Initialize GLaMM model and adjust requires_grad
    if not args.pretrained:
        model.get_model().initialize_glamm_model(model.get_model().config)
    else:
        for param in model.get_model().grounding_encoder.parameters():
            param.requires_grad = False
        if model.get_model().config.train_mask_decoder:
            model.get_model().grounding_encoder.mask_decoder.train()
            for param in model.get_model().grounding_encoder.mask_decoder.parameters():
                param.requires_grad = True

        # Projection layer
        model.get_model().text_hidden_fcs.train()
        for param in model.get_model().text_hidden_fcs.parameters():
            param.requires_grad = True

    # Set requires_grad for vision tower and mm projector
    for p in vision_tower.parameters():
        p.requires_grad = False
    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = True

    # Set requires_grad based on LoRA training
    lora_r = args.lora_r
    if lora_r == 0:
        for p in model.get_model().layers.parameters():
            p.requires_grad = True
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = True

    # Configure conversation library
    conversation_lib.default_conversation = conversation_lib.conv_templates[args.conv_type]

    # Configure LoRA if applicable
    if lora_r > 0:
        lora_config = setup_lora_config(model, args)
        model = get_peft_model(model, lora_config)

    # Resize token embeddings
    model.resize_token_embeddings(len(tokenizer))

    # Make certain modules trainable
    set_trainable_modules(model)


def setup_lora_config(model, args):
    """ Configure LoRA settings for the model. """

    def find_proj_layers(model, target_modules):
        """ Identify projection layers in the model for LoRA adaptation. """
        linear_cls = torch.nn.Linear
        lora_module_names = set()
        for name, module in model.named_modules():
            if (isinstance(module, linear_cls) and all(
                    x not in name for x in ["grounding_encoder", "vision_tower", "mm_projector", "text_hidden_fcs"]
            ) and any(x in name for x in target_modules)):
                lora_module_names.add(name)
        return sorted(list(lora_module_names))

    # Extracting LoRA target modules
    lora_target_modules = args.lora_target_modules.split(",")
    lora_module_names = find_proj_layers(model, lora_target_modules)

    # Configuring LoRA
    lora_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, target_modules=lora_module_names, lora_dropout=args.lora_dropout,
        bias="none", task_type="CAUSAL_LM"
    )
    return lora_config


def set_trainable_modules(model):
    """ Make specified modules in the model trainable. """
    trainable_modules = ["lm_head", "embed_tokens", "mm_projector", "mask_decoder", "text_hidden_fcs", "region_encoder"]
    for name, param in model.named_parameters():
        if any(module in name for module in trainable_modules):
            print(f"Making trainable: {name}, Shape: {param.shape}")
            param.requires_grad = True

    def count_parameters(model):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print('\033[92m' + "---- Total parameters: ----{}".format(total_params) + '\033[0m')
        print('\033[92m' + "---- Trainable parameters: ----{}".format(trainable_params) + '\033[0m')

    count_parameters(model)


def initialize_datasets_and_loaders(args, tokenizer):
    world_size = torch.cuda.device_count()
    args.distributed = world_size > 1

    # Common dataset arguments
    common_ds_args = {"dataset_dir": args.dataset_dir, "tokenizer": tokenizer,
                      "global_image_encoder": args.vision_tower,
                      "epoch_samples": args.batch_size * args.grad_accumulation_steps * args.steps_per_epoch * world_size,
                      "precision": args.precision, "image_size": args.image_size,
                      "num_classes_per_sample": args.num_classes_per_sample}

    # Training datasets
    cap_train_dataset = HybridCapDataset(
        **common_ds_args, dataset=args.cap_dataset, sample_rate=[float(x) for x in args.cap_sample_rates.split(",")],
        batch_size=args.batch_size, ) if args.use_cap_data else None
    reg_train_dataset = HybridRegDataset(
        **common_ds_args, dataset=args.reg_dataset, sample_rate=[float(x) for x in args.reg_sample_rates.split(",")],
        batch_size=args.batch_size, ) if args.use_reg_data else None
    seg_train_dataset = HybridSegDataset(
        **common_ds_args, dataset=args.seg_dataset, sample_rate=[float(x) for x in args.segm_sample_rates.split(",")],
        semantic_segm_data=args.semantic_segm_data, refer_segm_data=args.refer_segm_data,
        batch_size=args.batch_size, ) if args.use_segm_data else None

    # Validation datasets
    val_datasets = []
    if not args.no_eval:
        val_dataset_classes = {'CocoCapVal': CocoCapDataset,
                               'RefCOCOgRegVal': RefCocoGRegDataset,
                               'VisGenomeRegVal': VisualGenomeRegDataset,
                               'RefCOCOgSegmVal': ReferSegmDataset,
                               'PsgGCGVal': OpenPsgGCGDataset,
                               'RefCocoGCGVal': RefCOCOgGCGDataset,
                               'FlickrGCGVal': Flickr30kGCGDataset,
                               "SemanticSegmVal": SemanticSegmDataset,
                               'GranDfGCGVal': GranDfDataset,
                               }
        for val_dataset_name in args.val_dataset.split('|'):
            val_dataset_class = val_dataset_classes.get(val_dataset_name)
            if val_dataset_class:
                if val_dataset_class == ReferSegmDataset:
                    # Modify this if other datasets in refer_segm_data need to be included in val
                    refer_segm_data = 'refcocog'
                    all_datasets = refer_segm_data.split("||")
                    for d in all_datasets:
                        val_dataset_class = val_dataset_class(
                            **common_ds_args, validation=True, refer_segm_data=d, split='val'
                        )
                        val_dataset_class._set_len(len(val_dataset_class.refer_segm_data[d]['images']))
                        val_datasets.append(val_dataset_class)
                elif val_dataset_class == SemanticSegmDataset:
                    semantic_segm_data = 'pancancer_val'
                    all_datasets = semantic_segm_data.split("||")
                    for d in all_datasets:
                        val_dataset_class = val_dataset_class(
                            **common_ds_args, validation=True, semantic_segm_data=d)
                        val_dataset_class._set_len(len(val_dataset_class.data2list[d]))
                        val_datasets.append(val_dataset_class)
                else:
                    val_datasets.append(val_dataset_class(**common_ds_args, validation=True))

    return cap_train_dataset, reg_train_dataset, seg_train_dataset, val_datasets


def setup_data_loaders(args, cap_train_dataset, reg_train_dataset, seg_train_dataset, val_datasets, tokenizer):
    sampler_args = {"shuffle": False, "drop_last": False}
    train_loader_args = {"batch_size": args.batch_size, "shuffle": False, "num_workers": args.workers,
                         "pin_memory": False}
    val_loader_args = {"batch_size": args.val_batch_size, "shuffle": False, "num_workers": args.workers,
                       "pin_memory": False}
    collate_fn_args_train = partial(
        custom_collate_fn, tokenizer=tokenizer, use_mm_start_end=args.use_mm_start_end, local_rank=args.local_rank,
        inference=False
    )
    inference_mode = args.mask_validation
    collate_fn_args_val = partial(
        custom_collate_fn, tokenizer=tokenizer, use_mm_start_end=args.use_mm_start_end, local_rank=args.local_rank,
        inference=inference_mode
    )

    # Training loaders
    cap_train_loader = torch.utils.data.DataLoader(
        cap_train_dataset, sampler=torch.utils.data.distributed.DistributedSampler(
            cap_train_dataset, **sampler_args
        ), collate_fn=collate_fn_args_train, **train_loader_args
    ) if cap_train_dataset is not None else None
    reg_train_loader = torch.utils.data.DataLoader(
        reg_train_dataset, sampler=torch.utils.data.distributed.DistributedSampler(
            reg_train_dataset, **sampler_args
        ), collate_fn=collate_fn_args_train, **train_loader_args
    ) if reg_train_dataset is not None else None
    seg_train_loader = torch.utils.data.DataLoader(
        seg_train_dataset, sampler=torch.utils.data.distributed.DistributedSampler(
            seg_train_dataset, **sampler_args
        ), collate_fn=collate_fn_args_train, **train_loader_args
    ) if seg_train_dataset is not None else None

    # Validation loader
    val_loader = None
    if val_datasets:
        combined_val_datasets = ConcatDataset(val_datasets)
        val_loader = torch.utils.data.DataLoader(
            combined_val_datasets, **val_loader_args, collate_fn=collate_fn_args_val,
            sampler=torch.utils.data.distributed.DistributedSampler(combined_val_datasets, **sampler_args), )

    return cap_train_loader, reg_train_loader, seg_train_loader, val_loader


def initialize_deepspeed(model, tokenizer, args):
    ds_config = {"train_micro_batch_size_per_gpu": args.batch_size,
                 "gradient_accumulation_steps": args.grad_accumulation_steps,
                 "optimizer": {"type": "AdamW", "params": {"lr": args.lr, "weight_decay": 0.01,
                                                           "betas": (args.beta1, args.beta2)}},
                 "scheduler": {"type": "WarmupDecayLR",
                               "params": {"total_num_steps": args.epochs * args.steps_per_epoch, "warmup_min_lr": 0,
                                          "warmup_max_lr": args.lr, "warmup_num_steps": 625, "warmup_type": "linear"}},
                 "fp16": {"enabled": args.precision == "fp16"}, "bf16": {"enabled": args.precision == "bf16"},
                 "gradient_clipping": 1.0,
                 "zero_optimization": {"stage": 2, "contiguous_gradients": True, "overlap_comm": True,
                                       "reduce_scatter": True, "reduce_bucket_size": 5e8,
                                       "allgather_bucket_size": 5e8}, }

    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model, model_parameters=model.parameters(), collate_fn=partial(
            custom_collate_fn, tokenizer=tokenizer, use_mm_start_end=args.use_mm_start_end, local_rank=args.local_rank
        ), config=ds_config
    )

    return model_engine, optimizer, scheduler


def resume_training_from_checkpoint(model_engine, args):
    if args.auto_resume and not args.resume:
        # resume = os.path.join("/projects/ct_vision_language/output/GroundingLMM_normal_GranD", "ckpt_model_best")
        resume = os.path.join("/data/groups/beets-tan/r.vt.hull/output/GroundingLMM_final_weights_final/ckpt_model_best")
        if os.path.exists(resume):
            args.resume = resume

    if args.resume:
        load_path, client_state = model_engine.load_checkpoint(args.resume)
        with open(os.path.join(args.resume, "latest"), "r") as f:
            ckpt_dir = f.readlines()[0].strip()
        # if ckpt_dir == "global_step1250" or ckpt_dir == "global_step2500":
        #     args.resume = "/data/groups/beets-tan/r.vt.hull/output/GroundingLMM_normal_GranD"
        args.start_epoch = int(ckpt_dir.replace("global_step", "")) // args.steps_per_epoch
        print(f"Resume training from {args.resume}, start from epoch {args.start_epoch}")

# def resume_training_from_checkpoint(model_engine, args):
#     if args.custom_resume and args.resume_name:
#         # Construct the path to the specific checkpoint within ckpt_model_best
#         checkpoint_dir = os.path.join(args.resume)
#         resume_checkpoint_path = os.path.join(checkpoint_dir, args.resume_name)
        
#         print(f"Attempting to resume from custom checkpoint: {resume_checkpoint_path}")
        
#         if os.path.exists(resume_checkpoint_path):
#             load_path, client_state = model_engine.load_checkpoint(resume_checkpoint_path)
#             print(f"Resuming training from custom checkpoint: {resume_checkpoint_path}")
#             # Extract the epoch from resume_name if it follows a specific naming convention
#             try:
#                 args.start_epoch = int(args.resume_name.split('_')[-1]) // args.steps_per_epoch
#                 print(f"Start epoch set to: {args.start_epoch}")
#             except ValueError:
#                 print("Unable to parse the start epoch from resume_name.")
#         else:
#             print(f"Specified checkpoint {resume_checkpoint_path} does not exist.")
#     else:
#         if args.auto_resume and not args.resume:
#             resume = os.path.join("/projects/ct_vision_language/output/GroundingLMM_normal_GranD", "ckpt_model_best")
#             if os.path.exists(resume):
#                 args.resume = resume

#         if args.resume:
#             load_path, client_state = model_engine.load_checkpoint(args.resume)
#             with open(os.path.join(args.resume, "latest"), "r") as f:
#                 ckpt_dir = f.readlines()[0].strip()
#                 print(f"Resuming training from {args.resume}, checkpoint: {ckpt_dir}")
#             args.start_epoch = int(ckpt_dir.replace("global_step", "")) // args.steps_per_epoch
#             print(f"Resume training from {args.resume}, start from epoch {args.start_epoch}")



def main(args):
    tokenizer = setup_tokenizer_and_special_tokens(args)
    model = initialize_model(args, tokenizer)
    prepare_model_for_training(model, tokenizer, args)

    model_engine, optimizer, scheduler = initialize_deepspeed(model, tokenizer, args)
    resume_training_from_checkpoint(model_engine, args)

    cap_train_dataset, reg_train_dataset, seg_train_dataset, val_datasets = (
        initialize_datasets_and_loaders(args, tokenizer))
    cap_train_loader, reg_train_loader, seg_train_loader, val_loader = (
        setup_data_loaders(args, cap_train_dataset, reg_train_dataset, seg_train_dataset, val_datasets, tokenizer))

    # Determine active datasets and their weights
    active_dataloaders = []
    weights = []

    if args.use_cap_data:
        active_dataloaders.append(('cap', cap_train_loader))
        weights.append(args.weight_cap)
    if args.use_reg_data:
        active_dataloaders.append(('reg', reg_train_loader))
        weights.append(args.weight_reg)
    if args.use_segm_data:
        active_dataloaders.append(('seg', seg_train_loader))
        weights.append(args.weight_segm)

    # Assert that at least one dataset is active
    assert active_dataloaders, "Error: At least one dataset (segm, reg, or cap) must be active."

    dataset_iters = {'cap': iter(cap_train_loader) if args.use_cap_data else None,
                     'reg': iter(reg_train_loader) if args.use_reg_data else None,
                     'seg': iter(seg_train_loader) if args.use_segm_data else None, }

    writer = initialize_environment(args)

    if args.eval_only:
        print("Starting evaluation...", flush=True)
        dice = validate_model_performance(val_loader, model_engine, args.start_epoch, writer, args)
        print(f"Validation completed with dice score: {dice}", flush=True)
        if args.local_rank == 0:
            print("All validation steps completed.", flush=True)
        
        torch.distributed.barrier()
        sys.exit(0)

    if not args.eval_only:
        epoch_seeds = [random.randint(0, 100000) for _ in range(args.epochs)]
        dataset_choices = [idx for idx, _ in enumerate(active_dataloaders)]
        
        best_dice = 0.0

        for epoch in range(args.start_epoch, args.epochs):
            random.seed(epoch_seeds[epoch])

            step_choices = random.choices(dataset_choices, weights=weights, k=args.steps_per_epoch)

            dataset_iters = train(
                active_dataloaders, model_engine, epoch, scheduler, writer, dataset_iters, args, step_choices
            )

            if not args.no_eval:
                if args.mask_validation and not args.class_validation:
                    dice = validate_model_performance(val_loader, model_engine, epoch, writer, args)
                    print(dice, best_dice, flush=True)
                    sys.exit("Stopping the script here.")
                    is_best = dice > best_dice
                    best_dice = max(dice, best_dice)
                    if args.local_rank == 0:
                        print(f"Epoch: {epoch}, dice: {dice}")
                    save_checkpoint(model_engine, args, epoch, 'dice', f"{dice:.4f}", is_best)
                
                elif not args.mask_validation and not args.class_validation:
                    giou, ciou = validate_model_performance(val_loader, model_engine, epoch, writer, args)
                    if args.local_rank == 0:  # Log the progress
                        print(f"Epoch: {epoch}, giou: {giou}, ciou: {ciou}")
                    save_checkpoint(model_engine, args, epoch, 'giou-ciou', f"{giou:.4f}-{ciou:.4f}", is_best)

                elif not args.mask_validation and args.class_validation:
                    dice = validate_model_performance(val_loader, model_engine, epoch, writer, args)
                    is_best = dice > best_dice
                    best_dice = max(dice, best_dice)
                    if args.local_rank == 0:  # Log the progress
                        print(f"Epoch: {epoch}, dice: {dice}")
                    save_checkpoint(model_engine, args, epoch, 'dice', f"{dice:.4f}", is_best)

                else:
                    cur_val_loss = validate_model_performance(val_loader, model_engine, epoch, writer, args)
                    is_best = cur_val_loss < best_val_loss
                    best_val_loss = min(cur_val_loss, best_val_loss)
                    if args.local_rank == 0:  # Log the progress
                        print(f"Epoch: {epoch}, Current dice score: {cur_val_loss:.4f}, Best dice score: {best_val_loss:}")
                    save_checkpoint(model_engine, args, epoch, 'dice', f"{cur_val_loss:.4f}", is_best)
            else:
                print("No eval.")
                save_checkpoint(model_engine, args, epoch, 'epoch', epoch, True)


def save_checkpoint(model_engine, args, epoch, metric_name, metric_value, is_best):
    """ Saves the model checkpoint. """
    # If the checkpoint is the best, save it in ckpt_model_best, else in ckpt_model_last_epoch
    save_dir_name = "ckpt_model_best" if is_best else "ckpt_model_last_epoch"
    save_dir = os.path.join(args.log_dir, save_dir_name)
    # Ensure the directory exists
    if args.local_rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        ckpt_filename = f"epoch_{epoch}_val_{metric_name}_{metric_value}.pth"
        torch.save({"epoch": epoch, f"val_{metric_name}": metric_value}, os.path.join(save_dir, ckpt_filename))
    torch.distributed.barrier()
    model_engine.save_checkpoint(save_dir)


def train(active_datasets, model, epoch, scheduler, writer, dataset_iters, args, step_choices):
    """Main training loop."""

    def get_next_input(iterator, data_loader):
        """Retrieve next input from the iterator, or reinitialize if necessary."""
        try:
            return next(iterator), iterator
        except StopIteration:
            new_iterator = iter(data_loader)
            return next(new_iterator), new_iterator

    def log_progress():
        """Log training progress."""
        if global_step % args.print_freq == 0:
            if args.distributed:
                for tracker in trackers.values():
                    tracker.all_reduce()

            if args.local_rank == 0:
                progress.display(global_step + 1)
                for key, tracker in trackers.items():
                    writer.add_scalar(f"train/{key}", tracker.avg, global_step)
                writer.add_scalar("metrics/total_secs_per_batch", batch_time.avg, global_step)
                writer.add_scalar("metrics/data_secs_per_batch", data_time.avg, global_step)

            for tracker in trackers.values():
                tracker.reset()

    batch_time = AverageMeter("Time", ":.4f")
    data_time = AverageMeter("Data", ":.4f")
    trackers = {"loss": AverageMeter("Loss", ":.4f"),
                "ce_loss": AverageMeter("CeLoss", ":.4f"),
                "mask_bce_loss": AverageMeter("MaskBCELoss", ":.4f"),
                "mask_dice_loss": AverageMeter("MaskDICELoss", ":.4f"),
                "mask_loss": AverageMeter("MaskLoss", ":.4f")}
    progress = ProgressMeter(args.steps_per_epoch, list(trackers.values()), prefix=f"Epoch: [{epoch}]")

    model.train()
    end = time.time()
    for global_step in range(args.steps_per_epoch):
        for _ in range(args.grad_accumulation_steps):
            # Select data loader based on step choice
            dataset_type, data_loader = active_datasets[step_choices[global_step]]
            data_batch, new_iter = get_next_input(dataset_iters[dataset_type], data_loader)
            dataset_iters[dataset_type] = new_iter

            data_time.update(time.time() - end)
            # Prepare data and convert relevant tensors to bfloat16
            data_batch = dict_to_cuda(data_batch)
            for key in ["global_enc_images", "grounding_enc_images"]:
                data_batch[key] = data_batch[key].bfloat16()

            output_dict = model(**data_batch)

            # Update training metrics
            for key, tracker in trackers.items():
                if key in output_dict:
                    if key == "mask_bce_loss":
                        tracker.update(output_dict[key].item()/args.bce_loss_weight, data_batch["grounding_enc_images"].size(0))
                    if key == "mask_dice_loss":
                        tracker.update(output_dict[key].item()/args.dice_loss_weight, data_batch["grounding_enc_images"].size(0))
                    else:
                        tracker.update(output_dict[key].item(), data_batch["grounding_enc_images"].size(0))

            model.backward(output_dict["loss"])
            model.step()

        batch_time.update(time.time() - end)
        end = time.time()
        log_progress()

        if global_step != 0:
            curr_lr = scheduler.get_last_lr()
            if args.local_rank == 0:
                writer.add_scalar("train/lr", curr_lr[0], global_step)

    return dataset_iters

def validate_model_performance(validation_loader, training_model, current_epoch, tensorboard_writer, args):
    if not args.mask_validation and not args.class_validation:
        # For use with only segmentation/GCG type datasets
        trackers = {"intersection": AverageMeter("Intersec", ":.4f", Summary.SUM),
                    "union": AverageMeter("Union", ":.4f", Summary.SUM),
                    "gIoU": AverageMeter("gIoU", ":.4f", Summary.SUM)}

        training_model.eval()
        for data_batch in tqdm.tqdm(validation_loader):
            # Prepare data and convert relevant tensors to bfloat16
            data_batch = dict_to_cuda(data_batch)
            for key in ["global_enc_images", "grounding_enc_images"]:
                data_batch[key] = data_batch[key].bfloat16()
            torch.cuda.empty_cache()
            # Model inference without gradient tracking
            with torch.no_grad():
                results = training_model(**data_batch)

            predictions = results["pred_masks"]
            gt_masks = results["gt_masks"][0].int()
            # Note: An error at this line may suggest that the dataset used for validation does not support
            # segmentation tasks. Ensure that the dataset is appropriate for segmentation analysis.
            predicted_masks = (predictions[0] > 0).int()
            assert len(predictions) == 1

            intersection, union, accuracy_iou = 0.0, 0.0, 0.0
            for target, prediction in zip(gt_masks, predicted_masks):
                intersect, union_, _ = intersectionAndUnionGPU(
                    prediction.contiguous().clone(), target.contiguous(), 2, ignore_index=255
                )
                intersection += intersect
                union += union_
                accuracy_iou += intersect / (union_ + 1e-5)
                # handles no-object targets
                accuracy_iou[union_ == 0] += 1.0

            intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
            accuracy_iou = accuracy_iou.cpu().numpy() / gt_masks.shape[0]
            trackers["intersection"].update(intersection)
            trackers["union"].update(union)
            trackers["gIoU"].update(accuracy_iou, n=gt_masks.shape[0])

        for meter in trackers.values():
            meter.all_reduce()

        iou_per_class = trackers["intersection"].sum / (trackers["union"].sum + 1e-10)
        class_iou = iou_per_class[1]
        global_iou = trackers["gIoU"].avg[1]

        if args.local_rank == 0:
            tensorboard_writer.add_scalar("val/giou", global_iou, current_epoch)
            tensorboard_writer.add_scalar("val/ciou", class_iou, current_epoch)
            print("giou: {:.4f}, ciou: {:.4f}".format(global_iou, class_iou))

        return global_iou, class_iou
    

    elif not args.mask_validation and not args.class_validation:

        # Your existing setup
        tumor_classes = ["tumor", "suspicious lymph node", "lung metastasis", 
            "liver metastasis", "abdominal metastasis", "bone metastasis",
            "brain metastasis", "adrenal metastasis"]

        organ_classes = ["lung", "kidney", "rib", "vertebrae", "spleen", "gallbladder", 
            "liver", "stomach", "pancreas", "adrenal gland", "esophagus", 
            "trachea", "thyroid gland", "small bowel", "duodenum", "colon", 
            "urinary bladder", "prostate", "kidney cyst", "sacrum", "heart", 
            "aorta", "pulmonary vein", "brachiocephalic trunk", "subclavian artery", 
            "common carotid artery", "brachiocephalic vein", "atrial appendage", 
            "superior vena cava", "inferior vena cava", "portal vein and splenic vein", 
            "iliac artery", "iliac vein", "humerus", "scapula", "clavicula", 
            "femur", "hip", "spinal cord", "gluteus maximus", "gluteus medius", 
            "gluteus minimus", "autochthon", "iliopsoas", "brain", "skull", 
            "sternum", "costal cartilages"]

        trackers_dingen = {
            "GCG": {"dice": AverageMeter("Dice", ":.4f", Summary.SUM),
                    "nsd_1": AverageMeter("NSD1", ":.4f", Summary.SUM),
                    "nsd_3": AverageMeter("NSD3", ":.4f", Summary.SUM)
                    },
            "Organs": {"dice": AverageMeter("Dice", ":.4f", Summary.SUM),
                    "nsd_1": AverageMeter("NSD1", ":.4f", Summary.SUM),
                    "nsd_3": AverageMeter("NSD3", ":.4f", Summary.SUM)
                    },
            "Tumors": {"dice": AverageMeter("Dice", ":.4f", Summary.SUM),
                    "nsd_1": AverageMeter("NSD1", ":.4f", Summary.SUM),
                    "nsd_3": AverageMeter("NSD3", ":.4f", Summary.SUM)
                    },
            "Total": {"dice": AverageMeter("Dice", ":.4f", Summary.SUM),
                    "nsd_1": AverageMeter("NSD1", ":.4f", Summary.SUM),
                    "nsd_3": AverageMeter("NSD3", ":.4f", Summary.SUM)
                    }
        }

        per_scan_tracker = {}
        class_trackers = {}

        training_model.eval()

        for data_batch in tqdm.tqdm(validation_loader):
            data_batch = dict_to_cuda(data_batch)
            if data_batch["sampled_classes_list"][0][0] in organ_classes:
                name = "Organs"
            elif data_batch["sampled_classes_list"][0][0] in tumor_classes:
                name = "Tumors"
            else:
                name = "GCG"

            for key in ["global_enc_images", "grounding_enc_images"]:
                data_batch[key] = data_batch[key].bfloat16()
            torch.cuda.empty_cache()

            with torch.no_grad():
                results = training_model(**data_batch)

            predictions = results["pred_masks"]
            gt_masks = results["gt_masks"][0].int()
            gt_labels = data_batch["sampled_classes_list"][0]
            predicted_masks = (predictions[0] > 0).int()

            filename = os.path.basename(data_batch["image_paths"][0])
            patient_id = extract_part(filename)

            for target, prediction, label in zip(gt_masks, predicted_masks, gt_labels):

                dice_ = computeDiceCoefficient(
                    prediction.contiguous().clone(), target.contiguous(), 2, ignore_index=255
                )
                nsd_1_, nsd_3_ = computeNSD(patient_id, prediction.contiguous().clone().cpu().numpy(), target.contiguous().cpu().numpy())

                if name == "Organs" or name == "Tumors":
                    if label not in class_trackers:
                        class_trackers[label] = {
                            "dice": AverageMeter("Dice", ":.4f", Summary.SUM),
                            "nsd_1": AverageMeter("NSD1", ":.4f", Summary.SUM),
                            "nsd_3": AverageMeter("NSD3", ":.4f", Summary.SUM)
                        }
                    class_trackers[label]["dice"].update(dice_, n=1)
                    class_trackers[label]["nsd_1"].update(nsd_1_, n=1)
                    class_trackers[label]["nsd_3"].update(nsd_3_, n=1)

                if patient_id not in per_scan_tracker:
                    per_scan_tracker[patient_id] = {
                        "GCG": {"dice": AverageMeter("Dice", ":.4f", Summary.SUM),
                                "nsd_1": AverageMeter("NSD1", ":.4f", Summary.SUM),
                                "nsd_3": AverageMeter("NSD3", ":.4f", Summary.SUM)
                                },
                        "Organs": {"dice": AverageMeter("Dice", ":.4f", Summary.SUM),
                                "nsd_1": AverageMeter("NSD1", ":.4f", Summary.SUM),
                                "nsd_3": AverageMeter("NSD3", ":.4f", Summary.SUM)
                                },
                        "Tumors": {"dice": AverageMeter("Dice", ":.4f", Summary.SUM),
                                "nsd_1": AverageMeter("NSD1", ":.4f", Summary.SUM),
                                "nsd_3": AverageMeter("NSD3", ":.4f", Summary.SUM)
                                },
                        "Total": {"dice": AverageMeter("Dice", ":.4f", Summary.SUM),
                                "nsd_1": AverageMeter("NSD1", ":.4f", Summary.SUM),
                                "nsd_3": AverageMeter("NSD3", ":.4f", Summary.SUM)
                                }
                    }
                
                trackers_dingen[name]["dice"].update(dice_, n=1)
                trackers_dingen[name]["nsd_1"].update(nsd_1_, n=1)
                trackers_dingen[name]["nsd_3"].update(nsd_3_, n=1)
                trackers_dingen["Total"]["dice"].update(dice_, n=1)
                trackers_dingen["Total"]["nsd_1"].update(nsd_1_, n=1)
                trackers_dingen["Total"]["nsd_3"].update(nsd_3_, n=1)

                per_scan_tracker[patient_id][name]["dice"].update(dice_, n=1)
                per_scan_tracker[patient_id][name]["nsd_1"].update(nsd_1_, n=1)
                per_scan_tracker[patient_id][name]["nsd_3"].update(nsd_3_, n=1)
                per_scan_tracker[patient_id]["Total"]["dice"].update(dice_, n=1)
                per_scan_tracker[patient_id]["Total"]["nsd_1"].update(nsd_1_, n=1)
                per_scan_tracker[patient_id]["Total"]["nsd_3"].update(nsd_3_, n=1)

        torch.cuda.synchronize()
        torch.distributed.barrier()  # Synchronize after validation

        # Reduce across all processes (all_reduce)
        for meter in trackers_dingen.values():
            torch.cuda.synchronize()
            meter["dice"].all_reduce()
            meter["nsd_1"].all_reduce()
            meter["nsd_3"].all_reduce()
            torch.cuda.synchronize()

        for tracker in per_scan_tracker.values():
            for meter in tracker.values():
                torch.cuda.synchronize()
                meter["dice"].all_reduce()
                meter["nsd_1"].all_reduce()
                meter["nsd_3"].all_reduce()
                torch.cuda.synchronize()

        for meter in class_trackers.values():
            torch.cuda.synchronize()
            meter["dice"].all_reduce()
            meter["nsd_1"].all_reduce()
            meter["nsd_3"].all_reduce()
            torch.cuda.synchronize()


        accumulators = {
            "Total": {"dice": torch.tensor(0.0, device='cuda'), "nsd_1": torch.tensor(0.0, device='cuda'), "nsd_3": torch.tensor(0.0, device='cuda'), "count": torch.tensor(0, device='cuda', dtype=torch.long)},
            "GCG": {"dice": torch.tensor(0.0, device='cuda'), "nsd_1": torch.tensor(0.0, device='cuda'), "nsd_3": torch.tensor(0.0, device='cuda'), "count": torch.tensor(0, device='cuda', dtype=torch.long)},
            "Tumors": {"dice": torch.tensor(0.0, device='cuda'), "nsd_1": torch.tensor(0.0, device='cuda'), "nsd_3": torch.tensor(0.0, device='cuda'), "count": torch.tensor(0, device='cuda', dtype=torch.long)},
            "Organs": {"dice": torch.tensor(0.0, device='cuda'), "nsd_1": torch.tensor(0.0, device='cuda'), "nsd_3": torch.tensor(0.0, device='cuda'), "count": torch.tensor(0, device='cuda', dtype=torch.long)},
        }

        # Step 1: Compute per-patient averages
        for patient_tracker in per_scan_tracker.values():
            for key in ["Total", "GCG", "Tumors", "Organs"]:
                if patient_tracker[key]["dice"].count > 0:
                # Calculate per-patient average
                    patient_dice_avg = patient_tracker[key]["dice"].sum / patient_tracker[key]["dice"].count
                    patient_nsd_1_avg = patient_tracker[key]["nsd_1"].sum / patient_tracker[key]["nsd_1"].count
                    patient_nsd_3_avg = patient_tracker[key]["nsd_3"].sum / patient_tracker[key]["nsd_3"].count

                    # Accumulate these averages
                    accumulators[key]["dice"] += patient_dice_avg
                    accumulators[key]["nsd_1"] += patient_nsd_1_avg
                    accumulators[key]["nsd_3"] += patient_nsd_3_avg
                    accumulators[key]["count"] += 1  # Increment by 1 for each patient


        # Reduce across all processes
        for key in accumulators.keys():
            torch.distributed.all_reduce(accumulators[key]["dice"], op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(accumulators[key]["nsd_1"], op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(accumulators[key]["nsd_3"], op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(accumulators[key]["count"], op=torch.distributed.ReduceOp.SUM)
        
        torch.distributed.barrier()
        
        # Compute averages
        averages = {
            key: {
                "dice": (accumulators[key]["dice"] / accumulators[key]["count"]).item(),
                "nsd_1": (accumulators[key]["nsd_1"] / accumulators[key]["count"]).item(),
                "nsd_3": (accumulators[key]["nsd_3"] / accumulators[key]["count"]).item(),
            }
            for key in accumulators
        }

        # Print averaged results (only on rank 0)
        if torch.distributed.get_rank() == 0:
            for name, tracker in trackers_dingen.items():
                print(f"type {name} - dice: {tracker['dice'].avg:.4f}, nsd 1 : {tracker['nsd_1'].avg:.4f}, nsd 3 : {tracker['nsd_3'].avg:.4f}", flush=True)
            print("\n")
            for name, tracker in class_trackers.items():
                print(f"class {name} - dice: {tracker['dice'].avg:.4f}, nsd 1 : {tracker['nsd_1'].avg:.4f}, nsd 3 : {tracker['nsd_3'].avg:.4f}", flush=True)
            print("\n")
            for category, metrics in averages.items():
                print(f"type {category} per scan - dice {metrics['dice']:.4f}, nsd 1 : {metrics['nsd_1']:.4f}, nsd 3 : {metrics['nsd_3']:.4f}", flush=True)

        torch.distributed.barrier()

        return trackers_dingen["Total"]["dice"].avg

    elif args.mask_validation and not args.class_validation:
        tumor_classes = ["tumor", "suspicious lymph node" ,"lung metastasis", 
            "liver metastasis", "abdominal metastasis","bone metastasis",
            "brain metastasis", "adrenal metastasis"]
        organ_classes = ["lung", "kidney", "rib", "vertebrae", "spleen", "gallbladder", 
            "liver", "stomach", "pancreas", "adrenal gland", "esophagus", 
            "trachea", "thyroid gland", "small bowel", "duodenum", "colon", 
            "urinary bladder", "prostate", "kidney cyst", "sacrum", "heart", 
            "aorta", "pulmonary vein", "brachiocephalic trunk", "subclavian artery", 
            "common carotid artery", "brachiocephalic vein", "atrial appendage", 
            "superior vena cava", "inferior vena cava", "portal vein and splenic vein", 
            "iliac artery", "iliac vein", "humerus", "scapula", "clavicula", 
            "femur", "hip", "spinal cord", "gluteus maximus", "gluteus medius", 
            "gluteus minimus", "autochthon", "iliopsoas", "brain", "skull", 
            "sternum", "costal cartilages"]

        trackers_dingen = {
            "GCG": {"dice": AverageMeter("Dice", ":.4f", Summary.SUM),
                    "nsd_1": AverageMeter("NSD1", ":.4f", Summary.SUM),
                    "nsd_3": AverageMeter("NSD3", ":.4f", Summary.SUM)
                    },
            "Organs": {"dice": AverageMeter("Dice", ":.4f", Summary.SUM),
                    "nsd_1": AverageMeter("NSD1", ":.4f", Summary.SUM),
                    "nsd_3": AverageMeter("NSD3", ":.4f", Summary.SUM)
                    },
            "Tumors": {"dice": AverageMeter("Dice", ":.4f", Summary.SUM),
                    "nsd_1": AverageMeter("NSD1", ":.4f", Summary.SUM),
                    "nsd_3": AverageMeter("NSD3", ":.4f", Summary.SUM)
                    },
            "Total": {"dice": AverageMeter("Dice", ":.4f", Summary.SUM),
                    "nsd_1": AverageMeter("NSD1", ":.4f", Summary.SUM),
                    "nsd_3": AverageMeter("NSD3", ":.4f", Summary.SUM)
                    }
            }
        # per_scan_tracker = {}
        # print(f"[Rank {args.local_rank}] Starting validation at epoch {current_epoch}")

        torch.cuda.synchronize()
        # print(f"[Rank {args.local_rank}] Before barrier at the start of validation")
        torch.distributed.barrier()  # Synchronize before starting validation
        # print(f"[Rank {args.local_rank}] After barrier at the start of validation")
        # scan_trackers = {}
        class_trackers = {}
        # scan_GCG_trackers = {}
        training_model.eval()


        for data_batch in tqdm.tqdm(validation_loader):
            # Prepare data and convert relevant tensors to bfloat16
            data_batch = dict_to_cuda(data_batch)
            if data_batch["sampled_classes_list"][0][0] in organ_classes:
                name = "Organs"
            elif data_batch["sampled_classes_list"][0][0] in tumor_classes:
                name = "Tumors"
            else:
                name = "GCG"

            for key in ["global_enc_images", "grounding_enc_images"]:
                data_batch[key] = data_batch[key].bfloat16()
            torch.cuda.empty_cache()
            # Model inference without gradient tracking
            with torch.no_grad():
                results = training_model(**data_batch)

            predictions = results["pred_masks"]
            gt_masks = results["gt_masks"][0].int()
            gt_labels = data_batch["sampled_classes_list"][0]
            predicted_masks = (predictions[0] > 0).int()
            # predicted_masks = (predictions[0].sigmoid() > 0.5).int()
            assert len(predictions) == 1

            filename = os.path.basename(data_batch["image_paths"][0])
            patient_id = extract_part(filename)

            for target, prediction, label in zip(gt_masks, predicted_masks, gt_labels):

                dice_ = computeDiceCoefficient(
                    prediction.contiguous().clone(), target.contiguous(), 2, ignore_index=255
                )
                nsd_1_, nsd_3_ = computeNSD(patient_id, prediction.contiguous().clone().cpu().numpy(), target.contiguous().cpu().numpy())
                # Update the trackers with individual dice score, not the cumulative one
                if name == "Organs" or name == "Tumors":
                    if label not in class_trackers:
                        class_trackers[label] = {
                            "dice": AverageMeter("Dice", ":.4f", Summary.SUM),
                            "nsd_1": AverageMeter("NSD1", ":.4f", Summary.SUM),
                            "nsd_3": AverageMeter("NSD3", ":.4f", Summary.SUM)
                        }
                    class_trackers[label]["dice"].update(dice_, n=1)
                    class_trackers[label]["nsd_1"].update(nsd_1_, n=1)
                    class_trackers[label]["nsd_3"].update(nsd_3_, n=1)

                if label in tumor_classes and label != "brain metastasis":
                    # if patient_id not in scan_trackers:
                    #     scan_trackers[patient_id] = {}
                    # if label not in scan_trackers[patient_id]:
                    #     scan_trackers[patient_id][label] = {
                    #             "dice": AverageMeter("Dice", ":.4f", Summary.SUM),
                    #             "nsd_1": AverageMeter("NSD1", ":.4f", Summary.SUM),
                    #             "nsd_3": AverageMeter("NSD3", ":.4f", Summary.SUM)
                    #         }
                    # scan_trackers[patient_id][label]["dice"].update(dice_, n=1)
                    # scan_trackers[patient_id][label]["nsd_1"].update(nsd_1_, n=1)
                    # scan_trackers[patient_id][label]["nsd_3"].update(nsd_3_, n=1)
                    trackers_dingen["Tumors"]["dice"].update(dice_, n=1)
                    trackers_dingen["Tumors"]["nsd_1"].update(nsd_1_, n=1)
                    trackers_dingen["Tumors"]["nsd_3"].update(nsd_3_, n=1)
                
                elif label in organ_classes:
                    trackers_dingen["Organs"]["dice"].update(dice_, n=1)
                    trackers_dingen["Organs"]["nsd_1"].update(nsd_1_, n=1)
                    trackers_dingen["Organs"]["nsd_3"].update(nsd_3_, n=1)     
                else:
                    trackers_dingen["GCG"]["dice"].update(dice_, n=1)
                    trackers_dingen["GCG"]["nsd_1"].update(nsd_1_, n=1)
                    trackers_dingen["GCG"]["nsd_3"].update(nsd_3_, n=1) 
                
                # if name == "GCG":
                #     if patient_id not in scan_GCG_trackers:
                #         scan_GCG_trackers[patient_id] = {
                #             "dice": AverageMeter("Dice", ":.4f", Summary.SUM),
                #             "nsd_1": AverageMeter("NSD1", ":.4f", Summary.SUM),
                #             "nsd_3": AverageMeter("NSD3", ":.4f", Summary.SUM)
                #         }
                #     scan_GCG_trackers[patient_id]["dice"].update(dice_, n=1)
                #     scan_GCG_trackers[patient_id]["nsd_1"].update(nsd_1_, n=1)
                #     scan_GCG_trackers[patient_id]["nsd_3"].update(nsd_3_, n=1)

                # Update the overall dice and nsd for the current batch
                trackers_dingen["Total"]["dice"].update(dice_, n=1)
                trackers_dingen["Total"]["nsd_1"].update(nsd_1_, n=1)
                trackers_dingen["Total"]["nsd_3"].update(nsd_3_, n=1)

        torch.cuda.synchronize()

        torch.distributed.barrier()  # Synchronize after validation
       
        for meter in trackers_dingen.values():
            torch.cuda.synchronize()
            meter["dice"].all_reduce()
            meter["nsd_1"].all_reduce()
            meter["nsd_3"].all_reduce()
            torch.cuda.synchronize()

        # for tracker in scan_trackers.values():
        #     for meter in tracker.values():
        #         torch.cuda.synchronize()
        #         meter["dice"].all_reduce()
        #         meter["nsd_1"].all_reduce()
        #         meter["nsd_3"].all_reduce()
        #         torch.cuda.synchronize()

        # for meter in scan_GCG_trackers.values():
        #     torch.cuda.synchronize()
        #     meter["dice"].all_reduce()
        #     meter["nsd_1"].all_reduce()
        #     meter["nsd_3"].all_reduce()
        #     torch.cuda.synchronize()

        for meter in class_trackers.values():
            torch.cuda.synchronize()
            meter["dice"].all_reduce()
            meter["nsd_1"].all_reduce()
            meter["nsd_3"].all_reduce()
            torch.cuda.synchronize()

        torch.distributed.barrier()

        if args.local_rank == 0:
            for name, tracker in trackers_dingen.items():
                print(f"type {name} - dice: {tracker['dice'].avg:.4f}, nsd 1 : {tracker['nsd_1'].avg:.4f}, nsd 3 : {tracker['nsd_3'].avg:.4f}", flush=True)
            print("\n")
            for name, tracker in class_trackers.items():
                print(f"class {name} - dice: {tracker['dice'].avg:.4f}, nsd 1 : {tracker['nsd_1'].avg:.4f}, nsd 3 : {tracker['nsd_3'].avg:.4f}", flush=True)
            print("\n")
            # for name, tracker in scan_GCG_trackers.items():
            #     print(f"patient {name} GCG - dice: {tracker['dice'].avg:.4f}, nsd 1 : {tracker['nsd_1'].avg:.4f}, nsd 3 : {tracker['nsd_3'].avg:.4f}", flush=True)
            # for name, tracker in scan_trackers.items():
            #     print("\npatient ", name, flush=True)
            #     for n, meter in tracker.items():
            #         print(f"class {n} - dice: {meter['dice'].avg:.4f}, nsd 1 : {meter['nsd_1'].avg:.4f}, nsd 3 : {meter['nsd_3'].avg:.4f}", flush=True)
        
        torch.distributed.barrier()

        return trackers_dingen["Total"]["dice"].avg

    # elif not args.mask_validation and args.class_validation:
    #     signal.signal(signal.SIGINT, signal_handler)
    #     signal.signal(signal.SIGTERM, signal_handler)

    #     # Defining classes
    #     tumor_classes = ["tumor", "suspicious lymph node" ,"lung metastasis", 
    #         "liver metastasis", "abdominal metastasis","bone metastasis",
    #         "brain metastasis", "adrenal metastasis"]
    #     organ_classes = ["lung", "kidney", "rib", "vertebrae", "spleen", "gallbladder", 
    #         "liver", "stomach", "pancreas", "adrenal gland", "esophagus", 
    #         "trachea", "thyroid gland", "small bowel", "duodenum", "colon", 
    #         "urinary bladder", "prostate", "kidney cyst", "sacrum", "heart", 
    #         "aorta", "pulmonary vein", "brachiocephalic trunk", "subclavian artery", 
    #         "common carotid artery", "brachiocephalic vein", "atrial appendage", 
    #         "superior vena cava", "inferior vena cava", "portal vein and splenic vein", 
    #         "iliac artery", "iliac vein", "humerus", "scapula", "clavicula", 
    #         "femur", "hip", "spinal cord", "gluteus maximus", "gluteus medius", 
    #         "gluteus minimus", "autochthon", "iliopsoas", "brain", "skull", 
    #         "sternum", "costal cartilages"]

    #     # Initializing performance trackers
    #     trackers_dingen = {
    #         "GCG": {"loss": AverageMeter("Loss", ":.4f", Summary.SUM), "ce_loss": AverageMeter("CeLoss", ":.4f", Summary.SUM),
    #                 "mask_bce_loss": AverageMeter("MaskBCELoss", ":.4f", Summary.SUM),
    #                 "mask_dice_loss": AverageMeter("MaskDICELoss", ":.4f", Summary.SUM),
    #                 "dice_coef": AverageMeter("DICECoeff", ":.4f", Summary.SUM),
    #                 "mask_loss": AverageMeter("MaskLoss", ":.4f", Summary.SUM),
    #                 "nsd": AverageMeter("NSD", ":.4f", Summary.SUM)
    #                 },
    #         "Organs": {"loss": AverageMeter("Loss", ":.4f", Summary.SUM), "ce_loss": AverageMeter("CeLoss", ":.4f", Summary.SUM),
    #                 "mask_bce_loss": AverageMeter("MaskBCELoss", ":.4f", Summary.SUM),
    #                 "mask_dice_loss": AverageMeter("MaskDICELoss", ":.4f", Summary.SUM),
    #                 "dice_coef": AverageMeter("DICECoeff", ":.4f", Summary.SUM),
    #                 "mask_loss": AverageMeter("MaskLoss", ":.4f", Summary.SUM),
    #                 "nsd": AverageMeter("NSD", ":.4f", Summary.SUM)},
    #         "Tumors": {"loss": AverageMeter("Loss", ":.4f", Summary.SUM), "ce_loss": AverageMeter("CeLoss", ":.4f", Summary.SUM),
    #                 "mask_bce_loss": AverageMeter("MaskBCELoss", ":.4f", Summary.SUM),
    #                 "mask_dice_loss": AverageMeter("MaskDICELoss", ":.4f", Summary.SUM),
    #                 "dice_coef": AverageMeter("DICECoeff", ":.4f", Summary.SUM),
    #                 "mask_loss": AverageMeter("MaskLoss", ":.4f", Summary.SUM),
    #                 "nsd": AverageMeter("NSD", ":.4f", Summary.SUM)},
    #         "Total": {"loss": AverageMeter("Loss", ":.4f", Summary.SUM), "ce_loss": AverageMeter("CeLoss", ":.4f", Summary.SUM),
    #                 "mask_bce_loss": AverageMeter("MaskBCELoss", ":.4f", Summary.SUM),
    #                 "mask_dice_loss": AverageMeter("MaskDICELoss", ":.4f", Summary.SUM),
    #                 "dice_coef": AverageMeter("DICECoeff", ":.4f", Summary.SUM),
    #                 "mask_loss": AverageMeter("MaskLoss", ":.4f", Summary.SUM),
    #                 "nsd": AverageMeter("NSD", ":.4f", Summary.SUM)}
    #     }

    #     class_trackers = {}

    #     # Prepare model for validation phase
    #     training_model.train()

    #     for data_batch in tqdm.tqdm(validation_loader):
    #         if data_batch["sampled_classes_list"][0][0] in organ_classes:
    #             name = "Organs"
    #         elif data_batch["sampled_classes_list"][0][0] in tumor_classes:
    #             name = "Tumors"
    #         else:
    #             name = "GCG"
    #         # Prepare data and convert relevant tensors to bfloat16
    #         data_batch = dict_to_cuda(data_batch)
    #         for key in ["global_enc_images", "grounding_enc_images"]:
    #             if data_batch[key] is not None:
    #                 data_batch[key] = data_batch[key].bfloat16()
    #         torch.cuda.empty_cache()
    #         # Model inference without gradient tracking
    #         with torch.no_grad():
    #             predictions = training_model(**data_batch)
    #         label = data_batch["sampled_classes_list"][0][0]
    #         # Update performance metrics
    #         for key, tracker in trackers_dingen[name].items():
    #             if isinstance(predictions[key], torch.Tensor):
    #                 tracker.update(predictions[key].cpu().numpy(), data_batch["grounding_enc_images"].size(0))
    #             else:
    #                 tracker.update(predictions[key], data_batch["grounding_enc_images"].size(0))
    #         for key, tracker in trackers_dingen["Total"].items():
    #             if isinstance(predictions[key], torch.Tensor):
    #                 tracker.update(predictions[key].cpu().numpy(), data_batch["grounding_enc_images"].size(0))
    #             else:
    #                 tracker.update(predictions[key], data_batch["grounding_enc_images"].size(0))

    #         if name == "Organs" or name == "Tumors":
    #             if label not in class_trackers:
    #                 class_trackers[label] = {"loss": AverageMeter("Loss", ":.4f", Summary.SUM), "ce_loss": AverageMeter("CeLoss", ":.4f", Summary.SUM),
    #                     "mask_bce_loss": AverageMeter("MaskBCELoss", ":.4f", Summary.SUM),
    #                     "mask_dice_loss": AverageMeter("MaskDICELoss", ":.4f", Summary.SUM),
    #                     "dice_coef": AverageMeter("DICECoeff", ":.4f", Summary.SUM),
    #                     "mask_loss": AverageMeter("MaskLoss", ":.4f", Summary.SUM),
    #                     "nsd": AverageMeter("NSD", ":.4f", Summary.SUM)}
    #             for key, tracker in class_trackers[label].items():
    #                 if isinstance(predictions[key], torch.Tensor):
    #                     tracker.update(predictions[key].cpu().numpy(), data_batch["grounding_enc_images"].size(0))
    #                 else:
    #                     tracker.update(predictions[key], data_batch["grounding_enc_images"].size(0))

    #     # Ensure all processes are synchronized before reducing metrics
    #     torch.distributed.barrier()
    #     # Synchronize metrics across processes
    #     for tracker in trackers_dingen.values():
    #         # for tracker in trackers.values():
    #         # tracker["loss"].all_reduce()
    #         tracker["ce_loss"].all_reduce()
    #         tracker["mask_bce_loss"].all_reduce()
    #         tracker["mask_dice_loss"].all_reduce()
    #         tracker["dice_coef"].all_reduce()
    #         # tracker["mask_loss"].all_reduce()
    #         tracker["nsd"].all_reduce()

    #     for trackers in class_trackers.values():
    #         # for tracker in trackers.values():x
    #         # tracker["loss"].all_reduce()
    #         tracker["ce_loss"].all_reduce()
    #         tracker["mask_bce_loss"].all_reduce()
    #         tracker["mask_dice_loss"].all_reduce()
    #         tracker["dice_coef"].all_reduce()
    #         # tracker["mask_loss"].all_reduce()
    #         tracker["nsd"].all_reduce()

    #     # Ensure all processes are synchronized before printing metrics
    #     torch.distributed.barrier()
    #     # Calculate average validation loss
    #     if args.local_rank == 0:
    #         for name, tracker in trackers_dingen.items():
    #             print(f"type {name} - dice loss: {tracker['mask_dice_loss'].avg:.4f}, dice_coef: {tracker['dice_coef'].avg:.4f}, nsd : {tracker['nsd'].avg:.4f}, ce_loss : {tracker['ce_loss'].avg:.4f}, bce_loss : {tracker['mask_bce_loss'].avg:.4f}")
    #         for name, tracker in class_trackers.items():
    #             print(f"class {name} - dice loss: {tracker['mask_dice_loss'].avg:.4f}, dice_coef: {tracker['dice_coef'].avg:.4f}, nsd : {tracker['nsd'].avg:.4f}, ce_loss : {tracker['ce_loss'].avg:.4f}, bce_loss : {tracker['mask_bce_loss'].avg:.4f}")
            
    #     # exit()

    #     # avg_val_losses = [trackers["ce_loss"].avg for trackers in trackers_dingen]
    #     # avg_val_losses = [trackers["ce_loss"].avg for trackers in trackers_dingen]
    #     # # Tensorboard logging for primary process
    #     # if args.local_rank == 0:
    #     #     tensorboard_writer.add_scalar("val/loss", trackers_dingen["Total"]["mask_dice_loss"], current_epoch)

    #     return trackers_dingen["Total"]["dice_coef"].avg


def extract_part(filename):
    # Pattern to capture parts with a follow-up exam
    pattern_followup = r'PANCANCER_(\d+_\d+)_\d+\.'
    # Pattern to capture parts without a follow-up exam
    pattern_single = r'PANCANCER_(\d+)_\d+\.'
    
    match_followup = re.search(pattern_followup, filename)
    match_single = re.search(pattern_single, filename)
    
    if match_followup:
        return match_followup.group(1)
    elif match_single:
        return match_single.group(1)
    return None

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)


def return_total(patient_id):
    dictie = {
    "0554": 631,
    "0260": 721,
    "0253": 650,
    "0535": 590,
    "1555": 696,
    "1440": 425,
    "1261": 381,
    "0232": 511,
    "1714": 666,
    "0800": 661,
    "1632": 626,
    "0947": 706,
    "1719": 526,
    "1660": 696,
    "0012": 400,
    "1188": 803,
    "1766": 766,
    "0954": 736,
    "0620": 286,
    "0872": 521,
    "1052": 697,
    "0175": 719,
    "0935": 686,
    "1115": 496,
    "1200": 356,
    "0053": 686,
    "1612": 741,
    "0346": 813,
    "0400": 595,
    "1272": 681,
    "0547": 336,
    "1527": 656,
    "1335": 696,
    "0582": 641,
    "0752": 376,
    "1162": 747,
    "0710": 901,
    "1722": 836,
    "0237": 610,
    "1445": 636,
    "0416": 736,
    "1550": 762,
    "0311": 546,
    "0530": 641,
    "1424": 391,
    "1323": 726,
    "0245": 619,
    "1437": 389,
    "1522": 312,
    "1724": 352,
    "1277": 746,
    "1279": 882,
    "0302": 171,
    "1570": 670,
    "0065": 734,
    "0099": 526,
    "1617": 536,
    "1110": 666,
    "0762": 674,
    "0170": 938,
    "0122": 564,
    "1750": 650,
    "0362": 351,
    "1510": 351,
    "1302": 666,
    "1217": 646,
    "0465": 706,
    "1436": 807,
    "1481": 741,
    "0303": 621,
    "0225": 351,
    "1457": 892,
    "1498": 361,
    "0404": 336,
    "0750": 506,
    "1730": 696,
    "1037": 686,
    "0676": 631,
    "0645": 631,
    "0098": 341,
    "0876": 406,
    "0005": 300,
    "1677": 325,
    "0722": 758,
    "0865": 666,
    "1698": 733,
    "0637": 396,
    "0079": 1117,
    "1131": 755,
    "0804": 690,
    "0785": 682,
    "1605": 677,
    "1017": 676,
    "0837": 681,
    "0310": 644,
    "1562": 721,
    "1370": 642,
    "0417": 698,
    "1444": 750,
    "0371": 621,
    "1425": 730,
    "0476": 650,
    "0707": 671,
    "0640": 361,
    "0812": 600,
    "1032": 688,
    "1735": 716,
    "0755": 610,
    "0934": 701,
    "1452": 726,
    "1273": 482,
    "1574": 687,
    "0527": 645,
    "1588": 811,
    "1355": 362,
    "0460": 439,
    "1212": 676,
    "1515": 676,
    "1307": 311,
    "0367": 657,
    "0388": 596,
    "1506": 708,
    "0555": 761,
    "1006": 666,
    "0688": 646,
    "1260": 726,
    "0803": 316,
    "0534": 646,
    "1567": 391,
    "1374": 396,
    "0315": 671,
    "0140": 644,
    "1107": 676,
    "0632": 646,
    "1040": 688,
    "1788": 721,
    "1747": 741,
    "1095": 724,
    "0946": 676,
    "0281": 471,
    "0586": 711,
    "0571": 746,
    "1548": 388,
    "1395": 704,
    "0908": 796,
    "0148": 607,
    "1149": 326,
    "1768": 689,
    "0109": 646,
    "1195": 796,
    "0728": 701,
    "0986": 761,
    "0389": 301,
    "0441": 842,
    "0595": 706,
    "1594": 657,
    "0990": 721,
    "0055": 378,
    "0618": 696,
    "1678": 701,
    "1180": 414,
    "0985": 631,
    "1390": 680,
    "0438": 391,
    "1285": 726,
    "0826": 706,
    "0297": 356,
    "0258": 726,
    "1023": 637,
    "1238": 689,
    "1778": 591,
    "1383": 376,
    "1484": 811,
    "0838": 677,
    "1018": 691,
    "1658": 636,
    "1079": 696,
    "1386": 893,
    "1258": 726,
    "0085": 730,
    "0190": 672,
    "1078": 471,
    "0955": 116,
    "0982": 701,
    "1097": 686,
    "1225": 354,
    "1790": 675,
    "0082": 709,
    "1741": 425,
    "1391": 674,
    "0169": 411,
    "0777": 751,
    "0692": 610,
    "1348": 501,
    "1480": 681,
    "1528": 576,
    "1701": 709,
    "1473": 672,
    "0768": 675,
    "1708": 330,
    "0647": 706,
    "1792": 791,
    "1326": 451,
    "0420": 641,
    "0327": 636,
    "1388": 756,
    "0181": 177,
    "0567": 684,
    "1801": 721,
    "1072": 646,
    "0600": 691,
    "1789": 646,
    "0988": 730,
    "1775": 676,
    "0833": 736,
    "1135": 686,
    "0820": 666,
    "0350": 662,
    "1000": 1027,
    "1707": 646,
    "0967": 525,
    "1147": 701,
    "0735": 451,
    "1220": 763,
    "0588": 356,
    "1306": 893,
    "0355": 661,
    "1216": 876,
    "0334": 322,
    "1367": 344,
    "0515": 625,
    "1639": 1038,
    "1159": 351,
    "0771": 646,
    "1637": 656,
    "0045": 711,
    "0657": 663,
    "0698": 381,
    "1770": 669,
    "0102": 501,
    "0605": 526,
    "1077": 321,
    "1602": 381,
    "0857": 621,
    "1531": 476,
    "1205": 557,
    "1417": 676,
    "1236": 675,
    "0322": 451,
    "0425": 476,
    "1257": 641,
    "0204": 391,
    "1362": 740,
    "1244": 801,
    "0506": 349,
    "0674": 691,
    "0276": 740,
    "1645": 487,
    "1098": 650,
    "1702": 689,
    "1036": 662,
    "1005": 741,
    "0825": 650,
    "1369": 676,
    "0216": 666,
    "1542": 344,
    "0511": 650,
    "0330": 671,
    "0277": 376,
    "0570": 626,
    "1523": 696,
    "1762": 650,
    "0110": 671,
    "0963": 673,
    "1625": 381,
    "0902": 676,
    "0329": 346,
    "0162": 716,
    "1657": 623,
    "0472": 538,
    "1150": 631,
    "0970": 661,
    "0264": 426,
    "1477": 624,
    "0205": 596,
    "0502": 636,
    "1551": 721,
    "0907": 746,
    "1127": 668,
    "1120": 365,
    "1613": 691,
    "0821": 661,
    "0052": 650,
    "0925": 696,
    "1620": 588,
    "0840": 786,
    "1813": 689,
    "1060": 407,
    "0612": 376,
    "0575": 716,
    "1240": 610,
    "1461": 390,
    "0432": 746,
    "1375": 524,
    "0200": 678,
    "0421": 381,
    "1472": 856,
    "1314": 726,
    "0886": 372,
    "1232": 726,
    "1420": 709,
    "1083": 631,
    "0440": 601,
    "0975": 684,
    "1661": 732,
    "1800": 744,
    "0020": 646,
    "1652": 338,
    "0154": 511,
    "0072": 563,
    "1421": 721,
    "1587": 716,
    "1354": 907,
    "0368": 640,
    "0686": 681,
    "0493": 806,
    "0880": 666,
    "1222": 376,
    "0092": 365,
    "0457": 396,
    "1748": 703,
    "0893": 524,
    "1582": 613,
    "0387": 646,
    "0480": 509,
    "1511": 693,
    "1663": 661,
    "0818": 641,
    "1038": 860,
    "0790": 346,
    "0900": 771,
    "0885": 686,
    "1105": 1079,
    "1791": 361,
    "0628": 650,
    "0418": 506,
    "1296": 394,
    "0485": 696,
    "0201": 310,
    "0173": 396,
    "0930": 331,
    "0638": 606,
    "0659": 646,
    "1458": 729,
    "1297": 763,
    "1303": 356,
    "1418": 676,
    "0639": 331,
    "0897": 694,
    "0782": 629,
    "1689": 746,
    "1182": 666,
    "0878": 333,
    "1058": 711,
    "1679": 701,
    "0497": 365,
    "0944": 790,
    "1438": 680,
    "1278": 656,
    "0080": 561,
    "1627": 730,
    "1728": 773,
    "0787": 341,
    "0979": 89,
    "1749": 351,
    "1168": 426,
    "0948": 730,
    "0386": 730,
    "1508": 736,
    "1387": 700,
    "0651": 776,
    "0308": 701,
    "0278": 461,
    "0395": 676,
    "1680": 876,
    "0709": 1104,
    "1187": 776,
    "0128": 361,
    "1717": 332,
    "1239": 461,
    "0484": 305,
    "1071": 754,
    "0105": 335,
    "0007": 180,
    "0183": 436,
    "0748": 594,
    "0403": 405,
    "0413": 370,
    "0761": 373,
    "1784": 886,
    "0199": 641,
    "1008": 361,
    "0695": 751,
    "1093": 671,
    "1293": 721,
    "1059": 619,
    "0393": 341,
    "0993": 390,
    "0391": 423,
    "0496": 714,
    "0896": 515,
    "0351": 716,
    "1404": 676,
    "1776": 640,
    "1405": 331,
    "1363": 851,
    "1464": 711,
    "0828": 371,
    "1111": 812,
    "1456": 451,
    "1163": 1077,
    "0856": 790,
    "0700": 519,
    "1710": 632,
    "0656": 601,
    "0323": 641,
    "0342": 663,
    "1322": 315,
    "1416": 500,
    "0966": 631,
    "0314": 646,
    "0213": 396,
    "0066": 334,
    "0514": 739,
    "0566": 701,
    "1253": 555,
    "0914": 431,
    "1507": 650,
    "1233": 562,
    "0851": 321,
    "0155": 271,
    "0073": 729,
    "0974": 354,
    "1746": 730,
    "0779": 359,
    "0189": 376,
    "0906": 666,
    "1734": 685,
    "1546": 730,
    "0498": 394,
    "1298": 876,
    "1476": 646,
    "0548": 301,
    "0444": 812,
    "0436": 506,
    "0331": 604,
    "0844": 488,
    "0591": 363,
    "0858": 436,
    "0952": 756,
    "0892": 616,
    "1631": 386,
    "0247": 1146,
    "0492": 716,
    "1578": 625,
    "1769": 742,
    "0603": 407,
    "1492": 336,
    "0928": 686,
    "0749": 684,
    "1108": 424,
    "1675": 491,
    "0466": 653,
    "0679": 669,
    "0879": 650,
    "0284": 426,
    "0583": 671,
    "0916": 169,
    "0479": 446,
    "1697": 330,
    "0983": 749,
    "1276": 594,
    "1571": 591,
    "1204": 888,
    "0236": 696,
    "0531": 721,
    "0873": 624,
    "1641": 678,
    "1334": 696,
    "0306": 491,
    "0335": 691,
    "1136": 645,
    "0401": 641,
    "1346": 601,
    "1327": 641,
    "0347": 430,
    "0989": 691,
    "0569": 599,
    "1715": 305,
    "0789": 662,
    "0775": 626,
    "1633": 696,
    "0104": 676,
    "0861": 649,
    "0197": 691,
    "1330": 339,
    "1172": 395,
    "1453": 386,
    "0910": 646,
    "1130": 563,
    "0923": 531,
    "1687": 430,
    "1699": 372,
    "1502": 667,
    "0562": 381,
    "0256": 676,
    "1563": 696,
    "0523": 386,
    "1433": 387,
    "0751": 359,
    "1004": 701,
    "0824": 681,
    "0123": 688,
    "1143": 356,
    "1644": 678,
    "1099": 441,
    "0036": 629,
    "0456": 311,
    "1224": 381,
    "1368": 381,
    "0424": 729,
    "1256": 616,
    "1299": 616,
    "1198": 632,
    "0716": 346,
    "1311": 476,
    "0151": 681,
    "0911": 416,
    "1636": 806,
    "0044": 671,
    "0933": 869,
    "0354": 406,
    "0033": 671,
    "1706": 671,
    "1134": 465,
    "0746": 701,
    "1774": 721,
    "1389": 365,
    "0847": 369,
    "0774": 668,
    "0166": 463,
    "0040": 857,
    "1013": 664,
    "1042": 513,
    "0770": 146,
    "0021": 684,
    "1653": 716,
    "1041": 661,
    "0034": 501,
    "0273": 1004,
    "1401": 791,
    "0706": 681,
    "1629": 747,
    "0032": 693,
    "0613": 696,
    "1621": 702,
    "0503": 641,
    "1566": 316,
    "0343": 321,
    "1614": 718,
    "1711": 361,
    "0163": 616,
    "1624": 846,
    "1123": 699,
    "0903": 665,
    "1064": 391,
    "0510": 386,
    "0662": 361,
    "0703": 376,
    "1696": 415,
    "1067": 701,
    "0862": 696,
    "1191": 531,
    "0383": 661,
    "1459": 721,
    "1583": 406,
    "1814": 711,
    "1543": 384,
    "1218": 272,
    "1329": 882,
    "0481": 695,
    "1693": 393,
    "1109": 560,
    "0929": 676,
    "0794": 619,
    "0881": 601,
    "1328": 412,
    "0529": 671,
    "0769": 426,
    "1223": 86,
    "1745": 505,
    "0690": 85,
    "1681": 1584,
    "1186": 374,
    "0549": 761,
    "1235": 1308,
    "0309": 546,
    "0733": 631,
    "1169": 726,
    "1113": 852,
    "1439": 687,
    "1725": 676,
    "1619": 329,
    "0165": 471,
    "1591": 516,
    "1056": 641,
    "0630": 335,
    "1751": 646,
    "1497": 515,
    "0931": 816,
    "0499": 456,
    "1331": 766,
    "1646": 375,
    "1503": 356,
    "0918": 836,
    "0793": 701,
    "1024": 544,
    "1723": 402,
    "0943": 694,
    "0711": 711,
    "0241": 711,
    "1114": 907,
    "1252": 341,
    "0061": 601,
    "1754": 826,
    "1010": 396,
    "1726": 488,
    "1600": 694,
    "0653": 396,
    "1021": 513,
    "1441": 665,
    "0715": 326,
    "0649": 392,
    "1601": 701,
    "1106": 443,
    "1054": 711,
    "1514": 441,
    "0366": 695,
    "0961": 692,
    "1589": 778,
    "0146": 212,
    "1673": 723,
    "0551": 513,
    "1025": 756,
    "1604": 724,
    "1044": 646,
    "0864": 696,
    "0024": 681,
    "0898": 358,
    "0004": 696,
    "1069": 613,
    "0816": 631,
    "0644": 624,
    "0464": 686,
    "0137": 691,
    "1783": 696,
    "1084": 721,
    "0839": 371,
    "1479": 674,
    "0874": 382,
    "1141": 699,
    "0094": 331,
    "1496": 641,
    "1740": 730,
    "0059": 421,
    "0348": 341,
    "0349": 430,
    "0594": 641,
    "0568": 728,
    "1081": 716,
    "1786": 709,
    "0849": 411,
    "0994": 686,
    "1129": 177,
    "0909": 730,
    "1246": 411,
    "1394": 744,
    "0969": 396,
    "0615": 394,
    "1753": 686,
    "1138": 623,
    "0358": 650,
    "0718": 361,
    "0219": 656,
    "1347": 156,
    "0830": 676,
    "1782": 639,
    "0191": 671,
    "1419": 560,
    "1080": 526,
    "0081": 467,
    "1196": 778,
    "1096": 751,
    "0292": 252,
    "0161": 235,
    "1802": 586,
    "1569": 766,
    "0112": 446,
    "1061": 538,
    "0022": 732,
    "1691": 321,
    "1170": 721,
    "0103": 616,
    "1102": 700,
    "0374": 446,
    "0158": 421,
    "0068": 316,
    "1499": 416,
    "0084": 666,
    "1315": 602,
    "0363": 676,
    "0011": 597,
    "0402": 341,
    "0114": 346,
    "1371": 640,
    "0742": 626,
    "1743": 394,
    "1412": 726,
    "1760": 619,
    "0459": 676,
    "1166": 406,
    "0167": 661,
    "0047": 632,
    "1597": 635,
    "1635": 661,
    "0740": 666,
    "1713": 711,
    "0026": 636,
    "0100": 203,
    "0721": 346,
    "0267": 711,
    "0671": 1161,
    "1415": 371,
    "1500": 401,
    "1504": 701,
    "1488": 746,
    "0320": 701,
    "1552": 730,
    "1467": 430,
    "1360": 661,
    "1572": 706,
    "0512": 533,
    "0288": 650,
    "1520": 736,
    "0035": 670,
    "1647": 649,
    "1140": 709,
    "0572": 391,
    "0960": 441,
    "0120": 751,
    "0732": 668,
    "1761": 666,
    "0054": 646,
    "0229": 491,
    "0172": 706,
    "0316": 671,
    "0422": 746,
    "1250": 386,
    "0599": 696,
    "1317": 721,
    "1505": 778,
    "1410": 629,
    "1423": 336,
    "0262": 661,
    "1156": 746,
    "1651": 425,
    "0602": 701,
    "1070": 287,
    "0745": 781,
    "0917": 681,
    "1020": 371,
    "1137": 707,
    "0663": 741,
    "1630": 635,
    "1124": 471,
    "0670": 712,
    "1610": 761,
    "0999": 670,
    "0125": 673,
    "0737": 676,
    "0965": 709,
    "1145": 687,
    "0611": 501,
    "1642": 396,
    "1810": 781,
    "0030": 686,
    "0364": 676,
    "0242": 633,
    "1183": 694,
    "0517": 613,
    "1365": 768,
    "1451": 745,
    "0176": 661,
    "1125": 681,
    "1643": 744,
    "0610": 596,
    "0842": 646,
    "1062": 730,
    "1765": 710,
    "0998": 694,
    "1144": 471,
    "0117": 526,
    "0705": 646,
    "0577": 671,
    "1226": 168,
    "0430": 696,
    "1242": 857,
    "1545": 416,
    "1154": 376,
    "0337": 633,
    "0505": 650,
    "1377": 406,
    "1230": 371,
    "1325": 844,
    "0598": 911,
    "1157": 671,
    "0250": 726,
    "0332": 656,
    "1352": 488,
    "1247": 347,
    "0435": 361,
    "0536": 601,
    "0541": 706,
    "1407": 618,
    "1035": 709,
    "0772": 698,
    "1712": 326,
    "1015": 658,
    "0835": 746,
    "0972": 756,
    "0941": 776,
    "1773": 641,
    "1807": 860,
    "1074": 650,
    "1655": 371,
    "0867": 845,
    "0340": 643,
    "1320": 656,
    "1532": 801,
    "0533": 641,
    "1475": 376,
    "0792": 718,
    "1290": 131,
    "0095": 645,
    "0687": 650,
    "0648": 688,
    "0196": 426,
    "1628": 657,
    "1512": 1092,
    "0345": 231,
    "0008": 673,
    "0468": 660,
    "1203": 741,
    "1558": 730,
    "0409": 316,
    "0228": 460,
    "1449": 661,
    "1268": 208,
    "0487": 666,
    "1192": 381,
    "0275": 439,
    "1695": 683,
    "0868": 379,
    "1048": 709,
    "0086": 496,
    "0809": 662,
    "1358": 641,
    "1585": 678,
    "1579": 638,
    "1518": 670,
    "1718": 359,
    "1683": 423,
    "1178": 318,
    "1184": 689,
    "0490": 671,
    "0797": 920,
    "1738": 676,
    "0778": 631,
    "0949": 509,
    "1690": 406,
    "0138": 636,
    "1779": 700,
    "0385": 688,
    "0597": 726,
    "0606": 513,
    "1804": 807,
    "0378": 646,
    "1521": 681,
    "1482": 656,
    "0397": 943,
    "1519": 726,
    "0738": 645,
    "1682": 666,
    "0557": 696,
    "0625": 364,
    "1295": 696,
    "1487": 701,
    "1208": 341,
    "0294": 730,
    "1088": 95,
    "0895": 346,
    "0300": 461,
    "1028": 709,
    "0780": 681,
    "0469": 695,
    "0392": 305,
    "0408": 641,
    "0133": 471,
    "1667": 736,
    "0634": 626,
    "0015": 721,
    "1720": 412,
    "0655": 646,
    "0834": 381,
    "0088": 757,
    "1027": 607,
    "0807": 631,
    "1373": 685,
    "1581": 776,
    "1447": 523,
    "0372": 709,
    "0475": 666,
    "0454": 376,
    "0446": 681,
    "1227": 631,
    "0407": 676,
    "1289": 701,
    "1275": 351,
    "1379": 646,
    "1615": 680,
    "0646": 601,
    "0067": 659,
    "1173": 856,
    "0627": 750,
    "0846": 716,
    "1674": 681,
    "1422": 361,
    "1055": 691,
    "0875": 841,
    "1202": 709,
    "0377": 881,
    "1345": 451,
    "0537": 421,
    "0802": 701,
    "1022": 356,
    "0650": 581,
    "0945": 882,
    "1165": 588,
    "1744": 610,
    "0717": 346,
    "0622": 601,
    "0704": 367,
    "1757": 385,
    "0765": 683,
    "0788": 359,
    "0305": 299,
    "1577": 351,
    "1337": 705,
    "0398": 335,
    "1051": 633,
    "1670": 759,
    "0798": 771,
    "0764": 371,
    "0063": 628,
    "0050": 621,
    "0525": 696,
    "0222": 316,
    "1450": 663,
    "1524": 826,
    "1210": 736,
    "0263": 736,
    "1262": 336,
    "0410": 709,
    "1300": 676,
    "1455": 661,
    "0520": 365,
    "1787": 513,
    "1732": 756,
    "0652": 348,
    "0121": 331,
    "0132": 706,
    "1161": 444,
    "1607": 430,
    "0039": 163,
    "1560": 416,
    "0312": 696,
    "0255": 431,
    "0552": 783,
    "0848": 659,
    "1068": 453,
    "1215": 561,
    "0180": 403,
    "1286": 346,
    "1392": 726,
    "1229": 771,
    "0353": 675,
    "0448": 666,
    "0592": 369,
    "0694": 707,
    "0668": 661,
    "0193": 371,
    "1057": 406,
    "1780": 776,
    "0359": 746,
    "0338": 650,
    "1490": 671,
    "0677": 386,
    "0185": 711,
    "0682": 365,
    "1796": 706,
    "1785": 721,
    "1082": 798,
    "0471": 331,
    "0058": 351,
    "0585": 676,
    "1283": 688,
    "0518": 661,
    "0289": 485,
    "0178": 691,
    "0209": 361,
    "0428": 414,
    "0980": 846,
    "0608": 407,
    "1648": 668,
    "1249": 401,
    "0287": 346,
    "0579": 626,
    "1338": 671,
    "0184": 257,
    "1809": 966,
    "0808": 309,
    "1251": 341,
    "0829": 691,
    "1094": 637,
    "0429": 466,
    "1087": 341,
    "0491": 709,
    "0691": 616,
    "0891": 643,
    "0724": 841,
    "1559": 665,
    "0642": 636,
    "1116": 671,
    "0623": 686,
    "0442": 365,
    "0913": 648,
    "0713": 630,
    "0561": 590,
    "1556": 344,
    "1254": 701,
    "0489": 346,
    "1207": 549,
    "0564": 661,
    "1471": 664,
    "0344": 721,
    "0136": 489,
    "0559": 526,
    "1104": 721,
    "0177": 397,
    "0643": 681,
    "1176": 691,
    "1671": 856,
    "1525": 391,
    "1403": 626,
    "0336": 714,
    "0139": 391,
    "0091": 661,
    "1090": 396,
    "1269": 647,
    "1181": 661,
    "1009": 681,
    "0249": 641,
    "1333": 326,
    "1209": 706,
    "1486": 371,
    "0578": 547,
    "1727": 136,
    "0978": 701,
    "0784": 696,
    "0083": 626,
    "0423": 656,
    "0379": 616,
    "1737": 453,
    "1704": 723,
    "1003": 374,
    "0744": 686,
    "0888": 631,
    "0101": 470,
    "1634": 743,
    "1153": 337,
    "0866": 431,
    "0319": 208,
    "0427": 726,
    "1321": 387,
    "1434": 287,
    "0540": 671,
    "1353": 812,
    "0932": 531,
    "1112": 730,
    "1626": 694,
    "0814": 706,
    "0113": 401,
    "0850": 142,
    "0023": 346,
    "1764": 816,
    "0756": 421,
    "0324": 666,
    "1399": 535,
    "0544": 626,
    "0936": 681,
    "1443": 378,
    "1599": 466,
    "1344": 588,
    "0214": 711,
    "0246": 516,
    "0406": 451,
    "0474": 650,
    "1341": 794,
    "1553": 709,
    "1446": 706,
    "0654": 371,
    "0495": 351,
    "1026": 696,
    "0806": 188,
    "0553": 421,
    "1266": 386,
    "0313": 751,
    "0773": 636,
    "0823": 730,
    "1101": 391,
    "0921": 488,
    "0730": 361,
    "0889": 441,
    "0953": 771,
    "1034": 498,
    "0521": 631,
    "0693": 301,
    "1513": 744,
    "0361": 701,
    "0863": 650,
    "1603": 621,
    "1376": 656,
    "1564": 773,
    "1263": 650,
    "0411": 610,
    "0251": 344,
    "0443": 728,
    "1304": 501,
    "1516": 650,
    "1211": 701,
    "1430": 478,
    "0463": 421,
    "0431": 686,
    "1736": 422,
    "0811": 406,
    "1031": 376,
    "0003": 663,
    "0843": 666,
    "0968": 621,
    "1396": 736,
    "1763": 431,
    "1491": 406,
    "1781": 661,
    "1393": 250,
    "1469": 730,
    "1649": 686,
    "0609": 721,
    "0049": 711,
    "0194": 404,
    "0269": 415,
    "1091": 471,
    "0758": 686,
    "0179": 361,
    "0519": 726,
    "1397": 938,
    "0513": 361,
    "0396": 656,
    "1319": 726,
    "1565": 455,
    "1139": 471,
    "0919": 826,
    "0984": 673,
    "0719": 346,
    "1271": 540,
    "0356": 586,
    "0451": 401,
    "0270": 706,
    "0097": 420,
    "1799": 456,
    "1142": 341,
    "0043": 341,
    "1411": 476,
    "0134": 326,
    "1466": 626,
    "1313": 341,
    "0854": 636,
    "1133": 636,
    "0741": 626,
    "0501": 426,
    "1474": 664,
    "1234": 159,
    "1806": 681,
    "1089": 485,
    "1014": 801,
    "0666": 511,
    "0689": 688,
    "0685": 101,
    "0614": 696,
    "0573": 681,
    "0274": 626,
    "1406": 456,
    "0455": 666,
    "1454": 801,
    "1716": 197,
    "0164": 358,
    "0291": 444,
    "1536": 378,
    "0504": 617,
    "0576": 646,
    "0116": 708,
    "0799": 211,
    "0904": 506,
    "0384": 724,
    "0796": 668,
    "1118": 414,
    "1739": 746,
    "0883": 693,
    "0168": 341,
    "0633": 646,
    "1584": 438,
    "1609": 726,
    "1694": 333,
    "1167": 253,
    "0069": 653,
    "0182": 445,
    "0111": 335,
    "0380": 709,
    "0538": 637,
    "1593": 489,
    "1318": 391,
    "0596": 361,
    "1291": 211,
    "0882": 707,
    "1731": 426,
    "0482": 701,
    "1483": 730,
    "0747": 162,
    "0341": 741,
    "1255": 688,
    "1803": 317,
    "0325": 706,
    "0357": 341,
    "1356": 678,
    "1798": 656,
    "1573": 292,
    "1495": 697,
    "0494": 650,
    "1793": 461,
    "1759": 697,
    "0739": 646,
    "0048": 439,
    "1340": 665,
    "1676": 312,
    "1332": 334,
    "0333": 359,
    "0223": 636,
    "0786": 476,
    "0915": 466,
    "0516": 671,
    "0951": 99,
    "0964": 481,
    "1470": 381,
    "0016": 326,
    "1288": 775,
    "0522": 426,
    "0845": 431,
    "1065": 626,
    "0500": 636,
    "0950": 716,
    "1122": 354,
    "0699": 416,
    "0922": 730,
    "1576": 336,
    "1530": 339,
    "0550": 816,
    "0115": 619,
    "1400": 394,
    "0453": 401,
    "1547": 419,
    "0326": 348,
    "1413": 337,
    "1709": 407,
    "0254": 666,
    "0832": 581,
    "0660": 356,
    "1640": 661,
    "1755": 345,
    "1460": 711,
    "0971": 486,
    "0017": 346,
    "0370": 616,
    "0265": 365,
    "1359": 121,
    "1590": 471,
    "0478": 436,
    "1489": 396,
    "0658": 336,
    "1618": 334,
    "0390": 419,
    "0195": 301,
    "0078": 451,
    "1595": 370,
    "0723": 461,
    "0680": 605,
    "1148": 332,
    "0528": 386,
    "1280": 160,
    "1529": 660,
    "0708": 616,
    "0795": 441,
    "1692": 431,
    "0486": 341,
    "0131": 610,
    "1190": 416,
    "1398": 272,
    "1350": 616,
    "0702": 660,
    "0077": 519,
    "1343": 376,
    "0257": 620,
    "1672": 599,
    "1189": 401,
    "0147": 346,
    "0220": 645,
    "0546": 346,
    "0852": 284,
    "0127": 381,
    "0060": 414,
    "1575": 529,
    "0304": 483,
    "0240": 486,
    "1213": 339,
    "0150": 359,
    "1656": 365,
    "0477": 767,
    "0542": 183,
    "0282": 291,
    "1357": 396,
    "1267": 333,
    "0218": 451,
    "1128": 406,
    "0369": 671,
    "1308": 416,
    "0976": 521,
    "0995": 138,
    "0211": 431,
    "0296": 526,
    "1638": 420,
    "0819": 636,
    "0991": 317,
    "0321": 341,
    "1493": 646,
    "1686": 374,
    "0805": 371,
    "0813": 351,
    "1812": 331,
    "0461": 656,
    "1380": 203,
    "1264": 606,
    "0244": 349,
    "1076": 582,
    "0298": 365,
    "0754": 616,
    "1265": 91,
    "0188": 336,
    "0507": 326,
    "1155": 406,
    "0019": 376,
    "0697": 359,
    "0884": 421,
    "1039": 371,
    "0108": 388,
    "1586": 397,
    "0149": 506,
    "1534": 371,
    "1126": 426,
    "0574": 356,
    "0636": 396,
    "0836": 345,
    "0593": 371,
    "0129": 352,
    "0130": 329,
    "1221": 556,
    "0261": 186,
    "0230": 646,
    "0526": 407,
    "0641": 681,
    "0664": 361,
    "0064": 472,
    "1616": 672,
    "1703": 719,
    "1742": 345,
    "0766": 326,
    "0621": 361,
    "1053": 112,
    "0714": 633,
    "0041": 416,
    "0266": 381,
    "1794": 375,
    "0696": 656,
    "1659": 239,
    "1033": 324,
    "0684": 259,
    "0619": 376,
    "0729": 331,
    "0186": 399,
    "1549": 346,
    "0433": 396,
    "0462": 342,
    "1414": 657,
    "0171": 608,
    "1771": 445,
    "0604": 601,
    "1366": 371,
    "0589": 359,
    "1146": 336,
    "0734": 321,
    "1001": 701,
    "0673": 617,
    "1073": 749,
    "0394": 381,
    "1349": 475,
    "1219": 669,
    "1592": 118,
    "0447": 451,
    "0093": 396,
    "1431": 336,
    "0419": 455,
    "0859": 166,
    "0899": 222,
    "0025": 412,
    "1103": 347,
    "0328": 361,
    "1305": 569,
    "1117": 424,
    "0449": 591,
    "1665": 140,
    "1756": 359,
    "0145": 490,
    "1100": 152,
    "0905": 517,
    "1030": 361,
    "0810": 341,
    "1193": 665,
    "0683": 351,
    "0720": 222,
    "0920": 344,
    "0938": 666,
    "0488": 649,
    "0712": 701,
    "0940": 616,
    "1160": 550,
    "1179": 481,
    "1185": 766,
    "0352": 626,
    "1435": 538,
    "0760": 626,
    "1815": 396,
    "1688": 410,
    "0871": 671,
    "1598": 379,
    "0157": 381,
    "0776": 621,
    "1662": 477,
    "0759": 351,
    "0062": 691,
    "1705": 689,
    "0937": 311,
    "1544": 385,
    "0268": 421,
    "0192": 710,
    "1049": 381,
    "0887": 620,
    "0992": 439,
    "0208": 496,
    "0028": 461,
    "1808": 642,
    "1622": 666,
    "1650": 733,
    "0046": 346,
    "0667": 476,
    "0912": 162,
    "1132": 326,
    "1075": 709,
    "0607": 661,
    "1312": 386,
    "1007": 686,
    "0675": 48,
    "1700": 381,
    "1557": 630,
    "0124": 466,
    "1310": 127,
    "0198": 476,
    "0042": 361,
    "0957": 136,
    "1623": 351,
    "1002": 556,
    "0822": 373,
    "0087": 636,
    "1050": 411,
    "0870": 208,
    "0450": 476,
    "0890": 321,
    "1668": 503,
    "1177": 416,
    "1287": 451,
    "1608": 356,
    "1197": 419,
    "0558": 356,
    "0584": 610,
    "1339": 331,
    "1121": 159,
    "0010": 335,
    "1199": 672,
    "0051": 317,
    "0014": 376,
    "0031": 355,
    "0089": 365,
    "1408": 445,
    "1086": 630,
    "1478": 85,
    "1409": 211,
    "0532": 346,
    "0701": 416,
    "1811": 555,
    "1463": 385,
    "1721": 410,
    "0206": 410,
    "1533": 356,
    "1654": 333,
    "0831": 666,
    "0924": 676,
    "1231": 691,
    "0556": 444,
    "0996": 391,
    "1029": 636,
    "0781": 401,
    "0894": 686,
    "1294": 676,
    "0869": 386,
    "1426": 351,
    "1561": 476,
    "1324": 693,
    "0956": 509,
    "1274": 286,
    "1501": 345,
    "1666": 154,
    "0981": 681,
    "1381": 598,
    "0669": 664,
    "1119": 406,
    "0118": 356,
    "1596": 331,
    "1758": 346,
    "1795": 111,
    "0215": 374,
    "0259": 397,
    "1364": 376,
    "1805": 361,
    "0437": 291,
    "0587": 233,
    "0801": 145,
    "1248": 22,
    "1259": 307,
    "0987": 326,
    "0252": 320,
    "1237": 219,
    "0626": 141,
    "0581": 322,
    "0283": 396,
    "0226": 361,
    "1336": 144,
    "1541": 361,
    "0203": 292,
    "1011": 89,
    "0473": 85,
    "1158": 97,
    "0509": 116,
    "0817": 92,
    "1194": 130,
    "0565": 140,
    "0470": 197,
    "0678": 68,
    "0106": 89,
    "0727": 129,
    "1243": 101,
    "1282": 81,
    "1281": 87,
    "0144": 95,
    "0545": 87,
    "0927": 61,
    "0731": 117,
    "0210_1": 670,
    "1342_1": 681,
    "1448_1": 655,
    "0624": 601,
    "1228": 656,
    "0757_1": 642,
    "0563_1": 64,
    "0107": 439,
    "0942": 210,
    "0901_1": 551,
    "0763": 606,
    "0373": 671,
    "0234_1": 268,
    "0233_1": 202,
    "0543": 541,
    "0539_1": 402,
    "1442_1": 649,
    "0757": 721,
    "1539_1": 432,
    "0524": 832,
    "0901": 401,
    "1152_1": 571,
    "1309_1": 721,
    "0791_1": 724,
    "0107_1": 351,
    "0539": 431,
    "1228_1": 581,
    "0726_1": 586,
    "0543_1": 366,
    "0926": 631,
    "0661": 531,
    "0119_1": 434,
    "0412_1": 235,
    "0412": 740,
    "0743_1": 193,
    "0373_1": 355,
    "0672_1": 641,
    "0926_1": 160,
    "0508_1": 888,
    "0141_1": 211,
    "0635_1": 350,
    "1066_1": 351,
    "1309": 631,
    "1066": 194,
    "0156_1": 331,
    "0524_1": 751,
    "0661_1": 301,
    "0239_1": 232,
    "0027_1": 376,
    "0090_1": 444,
    "0942_1": 359,
    "1448": 259,
    "1402_1": 181,
    "0018_1": 379,
    "0508": 93,
    "0624_1": 355,
    "0763_1": 291,
    "1442": 179,
    "0563": 51
    }
    return dictie[patient_id]
