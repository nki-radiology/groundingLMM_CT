import torch
import torch.nn as nn
from abc import ABC, abstractmethod

from utils.utils import IGNORE_INDEX, IMAGE_TOKEN_INDEX
from model.layers import MLVLROIQueryModule
from model.llava.model.multimodal_encoder.builder import build_vision_tower
import re
import os

# class PerceiverModel(nn.Module):
#     def __init__(self, input_dim=1024, latent_dim=4096, num_latents=576, num_cross_att_heads=8, num_self_att_heads=8, num_self_att_layers=6):
#         super(PerceiverModel, self).__init__()
        
#         self.latents = nn.Parameter(torch.randn(1, num_latents, latent_dim))
        
#         self.cross_attention = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=num_cross_att_heads)
#         self.cross_attention_proj = nn.Linear(input_dim, latent_dim)
#         self.layer_norm1 = nn.LayerNorm(latent_dim)
#         self.layer_norm2 = nn.LayerNorm(latent_dim)
        
#         self.self_attention_layers = nn.ModuleList([
#             nn.TransformerEncoderLayer(d_model=latent_dim, nhead=num_self_att_heads)
#             for _ in range(num_self_att_layers)
#         ])
        
#         # Initialize weights
#         self._initialize_weights()

#     def _initialize_weights(self):
#         nn.init.kaiming_normal_(self.cross_attention_proj.weight, nonlinearity='relu')
#         for layer in self.self_attention_layers:
#             nn.init.xavier_uniform_(layer.self_attn.in_proj_weight)
#             nn.init.xavier_uniform_(layer.self_attn.out_proj.weight)
#             nn.init.constant_(layer.self_attn.in_proj_bias, 0)
#             nn.init.constant_(layer.self_attn.out_proj.bias, 0)
#             nn.init.xavier_uniform_(layer.linear1.weight)
#             nn.init.xavier_uniform_(layer.linear2.weight)
#             nn.init.constant_(layer.linear1.bias, 0)
#             nn.init.constant_(layer.linear2.bias, 0)

#     def forward(self, x):
#         # Cross-attention
#         x_proj = self.cross_attention_proj(x)
#         if torch.isnan(x_proj).any():
#             raise ValueError("NaNs found after 1st cross-attention")
#         latents = self.latents.expand(x.size(0), -1, -1)
#         latents = latents.transpose(0, 1)  # (num_latents, batch_size, latent_dim)
#         x_proj = x_proj.transpose(0, 1)    # (seq_len, batch_size, latent_dim)

#         latents, _ = self.cross_attention(latents, x_proj, x_proj)
#         latents = self.layer_norm1(latents)
        
#         # Check for NaNs after cross-attention
#         if torch.isnan(latents).any():
#             raise ValueError("NaNs found after cross-attention")
        
#         # Self-attention
#         for layer in self.self_attention_layers:
#             latents = layer(latents)
#             latents = self.layer_norm2(latents)
#             # Check for NaNs after each layer
#             if torch.isnan(latents).any():
#                 raise ValueError("NaNs found after self-attention layer")

#         latents = latents.transpose(0, 1)  # (batch_size, num_latents, latent_dim)
#         return latents

class LlavaMetaModel:
    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            modules = [nn.Linear(config.mm_hidden_size, config.hidden_size),
                       nn.GELU(),
                       nn.Linear(config.hidden_size, config.hidden_size)]
            self.mm_projector = nn.Sequential(*modules)
            # self.mm_projector = PerceiverModel()
            # for name, param in self.mm_projector.named_parameters():
            #     if param.grad is not None:
            #         if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
            #             print(f"NaN/Inf in gradients of {name}")
        self.region_encoder = MLVLROIQueryModule(embed_dims=1024, out_dims=4096, num_levels=4)

    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.mm_vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        self.config.mm_vision_tower = vision_tower

        vision_tower = build_vision_tower(model_args)

        if fsdp is not None and len(fsdp) > 0:
            self.vision_tower = [vision_tower]
        else:
            self.vision_tower = vision_tower

        self.config.use_mm_proj = True
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature

        if not hasattr(self, "mm_projector"):
            self.mm_projector = nn.Linear(
                self.config.mm_hidden_size, self.config.hidden_size
            )

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(
                pretrain_mm_mlp_adapter, map_location="cpu"
            )

            def get_w(weights, keyword):
                return {
                    k.split(keyword + ".")[1]: v
                    for k, v in weights.items()
                    if keyword in k
                }

            self.mm_projector.load_state_dict(
                get_w(mm_projector_weights, "mm_projector")
            )


class LlavaMetaForCausalLM(ABC):
    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_clip(self, images):
        image_features_cls, image_forward_outs = self.get_model().get_vision_tower()(images)
        return image_features_cls

    def encode_images(self, images, names):
        # (8, 576, 1024)
        image_features, image_forward_outs = self.get_model().get_vision_tower()(images)
        # for idx, name in enumerate(names):
        #     pos_enc = self.add_pos(os.path.basename(name), device='cuda')
        #     pos_enc = pos_enc.unsqueeze(0).expand(576, 1024)  # Shape (576, 1024)
        #     image_features[idx] += pos_enc

        image_features = self.get_model().mm_projector(image_features)
        return image_features, image_forward_outs

    def extract_part(self, filename):
        # Pattern to capture parts with a follow-up exam
        pattern_followup = r'PANCANCER_(\d+_\d+)_\d+\.npy'
        # Pattern to capture parts without a follow-up exam
        pattern_single = r'PANCANCER_(\d+)_\d+\.npy'
        
        match_followup = re.search(pattern_followup, filename)
        match_single = re.search(pattern_single, filename)
        
        if match_followup:
            return match_followup.group(1)
        elif match_single:
            return match_single.group(1)
        return None

    def normalize_position(self, position, total):
        # Normalize the position relative to the total number of slices
        return position / total

    def add_pos(self, name, device='cpu'):
        match = re.search(r'_(\d+)\.npy$', name)
        patient_id = self.extract_part(name)
        total = self.return_total(patient_id)

        slice = float(match.group(1))
        slice = self.normalize_position(slice, total)
    
        positional_encoding = torch.zeros(1024, device=device)  # Create a tensor directly on the specified device

        slice_tensor = torch.tensor(slice, device=device)  # Convert slice to a tensor

        for j in range(0, 1024, 2):
            positional_encoding[j] = torch.sin(slice_tensor / (10000 ** ((2 * j) / 1024)))
            positional_encoding[j + 1] = torch.cos(slice_tensor / (10000 ** ((2 * j) / 1024)))
            
        return positional_encoding

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, attention_mask, past_key_values, labels, images, bboxes, names
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if (
                past_key_values is not None
                and vision_tower is not None
                and images is not None
                and input_ids.shape[1] == 1
            ):
                attention_mask = torch.ones(
                    (attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
            return input_ids, attention_mask, past_key_values, None, labels
        
        mlvl_reg_features = [None for _ in range(len(input_ids))]

        if type(images) is list or images.ndim == 5:
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            image_features = [x.flatten(0, 1) for x in image_features]
        else:
            # Process for region
            # (8, 576, 4096)
            image_features, image_forward_outs = self.encode_images(images, names)
            if self.config.with_region:
                select_hidden_state_layer = self.config.mm_vision_select_layer
                num_level_reg_features = self.config.num_level_reg_features
                mlvl_reg_features = image_forward_outs.hidden_states[select_hidden_state_layer::-3]
                mlvl_reg_features = mlvl_reg_features[::-1]
                mlvl_reg_features = mlvl_reg_features[-num_level_reg_features:]
                mlvl_reg_features = [item[:, 1:].to(images.dtype) for item in mlvl_reg_features]

                if bboxes is not None and (len(bboxes) > 0):
                    mlvl_reg_features = self.model.region_encoder(mlvl_reg_features, bboxes)
                else:
                    mlvl_reg_features = [None for _ in range(len(input_ids))]

        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_image_idx = 0
        for batch_idx, (cur_input_ids, reg_feat) in enumerate(zip(input_ids, mlvl_reg_features)): # Adjusted the loop to include reg_feat
            curr_full_input_ids = []
            if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:
                # multimodal LLM, but the current sample is not multimodal
                cur_input_embeds = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = (
                    cur_input_embeds
                    + (
                        0.0 * self.get_model().mm_projector(vision_tower.dummy_feature)
                    ).sum()
                )
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue
            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            cur_new_input_embeds = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape
            while image_token_indices.numel() > 0:
                cur_image_features = image_features[cur_image_idx]
                image_token_start = image_token_indices[0]
                if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(
                    self.config, "mm_use_im_start_end", False
                ):
                    # preparing input embedding
                    cur_new_input_embeds.append(
                        self.get_model()
                        .embed_tokens(cur_input_ids[: image_token_start - 1])
                        .detach()
                    )
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(
                            cur_input_ids[image_token_start - 1 : image_token_start]
                        )
                    )
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(
                            cur_input_ids[image_token_start + 1 : image_token_start + 2]
                        )
                    )
                    # preparing input_ids
                    curr_full_input_ids.append(cur_input_ids[: image_token_start - 1])
                    curr_full_input_ids.append(cur_input_ids[image_token_start - 1: image_token_start])
                    curr_full_image_token = torch.full((cur_image_features.shape[0],), image_token_start, dtype=torch.int64)
                    curr_full_input_ids.append(curr_full_image_token)
                    curr_full_input_ids.append(cur_input_ids[image_token_start + 1: image_token_start + 2])
                    # preparing labels
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(
                            torch.full(
                                (cur_image_features.shape[0],),
                                IGNORE_INDEX,
                                device=labels.device,
                                dtype=labels.dtype,
                            )
                        )
                        cur_new_labels.append(
                            cur_labels[image_token_start : image_token_start + 1]
                        )
                        cur_labels = cur_labels[image_token_start + 2 :]
                elif getattr(self.config, "mm_use_im_start_end", False):
                    # preparing input embedding
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_ids[:image_token_start])
                    )
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(
                            cur_input_ids[image_token_start + 1 : image_token_start + 2]
                        )
                    )
                    # preparing input_ids
                    curr_full_input_ids.append(cur_input_ids[: image_token_start])
                    curr_full_image_token = torch.full((cur_image_features.shape[0],), image_token_start,
                                                       dtype=torch.int64)
                    curr_full_input_ids.append(curr_full_image_token)
                    curr_full_input_ids.append(cur_input_ids[image_token_start + 1: image_token_start + 2])
                    # preparing labels
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(
                            torch.full(
                                (cur_image_features.shape[0],),
                                IGNORE_INDEX,
                                device=labels.device,
                                dtype=labels.dtype,
                            )
                        )
                        cur_new_labels.append(
                            cur_labels[image_token_start + 1 : image_token_start + 2]
                        )
                        cur_labels = cur_labels[image_token_start + 2 :]
                else:
                    # preparing input embedding
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_ids[:image_token_start])
                    )
                    cur_new_input_embeds.append(cur_image_features)
                    # preparing input_ids
                    curr_full_input_ids.append(cur_input_ids[: image_token_start])
                    curr_full_image_token = torch.full((cur_image_features.shape[0],), image_token_start,
                                                       dtype=torch.int64)
                    curr_full_input_ids.append(curr_full_image_token)
                    # preparing labels
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(
                            torch.full(
                                (cur_image_features.shape[0],),
                                IGNORE_INDEX,
                                device=labels.device,
                                dtype=labels.dtype,
                            )
                        )
                        cur_labels = cur_labels[image_token_start + 1 :]

                cur_image_idx += 1
                if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(
                    self.config, "mm_use_im_start_end", False
                ):
                    cur_input_ids = cur_input_ids[image_token_start + 2 :]
                elif getattr(self.config, "mm_use_im_start_end", False):
                    cur_input_ids = cur_input_ids[image_token_start + 2 :]
                else:
                    cur_input_ids = cur_input_ids[image_token_start + 1 :]
                image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            if cur_input_ids.numel() > 0:
                if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(
                    self.config, "mm_use_im_start_end", False
                ):
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_ids).detach()
                    )
                elif getattr(self.config, "mm_use_im_start_end", False):
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_ids)
                    )
                else:
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_ids)
                    )
                curr_full_input_ids.append(cur_input_ids)
                if labels is not None:
                    cur_new_labels.append(cur_labels)
            cur_new_input_embeds = [
                x.to(device=self.device) for x in cur_new_input_embeds
            ]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            curr_full_input_ids = [x.to(device=self.device) for x in curr_full_input_ids]
            curr_full_input_ids = torch.cat(curr_full_input_ids, dim=0)
            # current new_input_embeds computation complete (Lx4096)
            # Replace embeds of <bbox> with region feats (num_box x 4096)
            if reg_feat is not None:
                BBOX_TOKEN_ID = self.config.bbox_token_idx
                reg_embeds = torch.zeros_like(cur_new_input_embeds)  # (Lx4096)
                reg_mask = (curr_full_input_ids == BBOX_TOKEN_ID)

                # To Handle errors: Check if the shapes of reg_embeds[reg_mask] and reg_feat match
                if reg_embeds[reg_mask].shape[0] != reg_feat.shape[0]:
                    # If they don't match, slice reg_feat to make the shapes match
                    min_shape = reg_embeds[reg_mask].shape[0]
                    reg_feat = reg_feat[:min_shape]

                reg_embeds[reg_mask] = reg_feat.to(reg_embeds.dtype)
                cur_new_input_embeds = cur_new_input_embeds * (~reg_mask).to(
                    cur_new_input_embeds.dtype)[:, None] + reg_embeds

            new_input_embeds.append(cur_new_input_embeds)
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)

        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat(
                    (
                        cur_new_embed,
                        torch.zeros(
                            (max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]),
                            dtype=cur_new_embed.dtype,
                            device=cur_new_embed.device,
                        ),
                    ),
                    dim=0,
                )
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat(
                        (
                            cur_new_label,
                            torch.full(
                                (max_len - cur_new_label.shape[0],),
                                IGNORE_INDEX,
                                dtype=cur_new_label.dtype,
                                device=cur_new_label.device,
                            ),
                        ),
                        dim=0,
                    )
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)

            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(
                    attention_mask, _new_labels, new_labels
                ):
                    new_attn_mask_pad_left = torch.full(
                        (cur_new_labels.shape[0] - labels.shape[1],),
                        True,
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )
                    new_attn_mask_pad_right = torch.full(
                        (cur_new_labels_align.shape[0] - cur_new_labels.shape[0],),
                        False,
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )
                    cur_new_attention_mask = torch.cat(
                        (
                            new_attn_mask_pad_left,
                            cur_attention_mask,
                            new_attn_mask_pad_right,
                        ),
                        dim=0,
                    )
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels = torch.stack(new_labels, dim=0)

            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full(
                    (
                        attention_mask.shape[0],
                        new_input_embeds.shape[1] - input_ids.shape[1],
                    ),
                    True,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat(
                    (new_attn_mask_pad_left, attention_mask), dim=1
                )
                assert attention_mask.shape == new_input_embeds.shape[:2]
        # new_input_embeds is (8, 717, 4096), new_labels is (8, 717), attention_masks is (8, 717)
        return None, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, num_new_tokens):

        if model_args.mm_use_im_start_end:

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(
                    model_args.pretrain_mm_mlp_adapter, map_location="cpu"
                )
                embed_tokens_weight = mm_projector_weights["model.embed_tokens.weight"]
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[
                        -num_new_tokens:
                    ]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(
                        f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}."
                    )
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

    def return_total(self, patient_id):
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