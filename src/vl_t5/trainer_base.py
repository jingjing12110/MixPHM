import os

import torch
import torch.nn as nn
from pprint import pprint

# from transformers import T5Config, BartConfig
from transformers.optimization import AdamW, \
    get_linear_schedule_with_warmup

from vl_t5.modeling_t5 import T5LayerNorm
from vl_t5.tokenization import VLT5TokenizerFast
from vl_t5.utils import load_state_dict


class TrainerBase(object):
    def __init__(self, args):
        self.args = args
        
        # self.train_loader = train_loader
        # self.val_loader = val_loader
        # self.test_loader = test_loader
        
        self.verbose = True
        # if self.args.distributed:
        #     if self.args.gpu != 0:
        #         self.verbose = False
        
        if self.args.tokenizer is None:
            self.args.tokenizer = self.args.backbone
    
    def create_config(self):
        if 't5' in self.args.backbone:
            from .my_transformers.configuration_t5 import T5Config
            config_class = T5Config
        else:
            return None
        
        config = config_class.from_pretrained(self.args.backbone)
        args = self.args
        
        # *********** VL-T5 ***********
        config.feat_dim = args.feat_dim
        config.pos_dim = args.pos_dim
        config.n_images = 2
        
        config.use_vis_order_embedding = args.use_vis_order_embedding
        
        config.dropout_rate = args.dropout
        config.dropout = args.dropout
        config.attention_dropout = args.dropout
        config.activation_dropout = args.dropout
        
        config.use_vis_layer_norm = args.use_vis_layer_norm
        config.individual_vis_layer_norm = args.individual_vis_layer_norm
        config.losses = args.losses
        
        config.share_vis_lang_layer_norm = args.share_vis_lang_layer_norm
        config.classifier = args.classifier
        # *********** VL-T5 ***********
        
        # *********** adaMix ***********
        config.adapter_type = args.adapter_type
        config.apply_adapter = args.apply_adapter
        config.adapter_size = args.adapter_size

        config.apply_expert_soup = args.apply_expert_soup
        config.num_experts = args.num_experts
        config.inference_level = args.inference_level
        config.sharing_down = args.sharing_down
        config.sharing_up = args.sharing_up
        
        config.apply_lora = args.apply_lora
        config.lora_alpha = args.lora_alpha
        config.lora_r = args.lora_r
        
        config.use_consistency_loss = args.use_consistency_loss

        # *********** BitFit ***********
        if args.apply_bitfit:
            config.bitfit = True
        else:
            config.bitfit = False
        
        # *********** MoEBERT ***********
        config.apply_moebert = args.apply_moebert
        config.moebert_expert_num = args.moebert_expert_num
        config.moebert_route_method = args.moebert_route_method
        config.moebert_route_hash_list = args.moebert_route_hash_list
        
        # MoE
        config.add_gate = args.add_gate
        config.apply_moe = args.apply_moe
        config.apply_adapter_moe = args.apply_adapter_moe
        
        # =========================================
        config.use_adapter_r2loss = args.use_adapter_r2loss
        config.red_w = args.red_w
        config.using_mi_loss = args.using_mi_loss
        config.align_w = args.align_w
        config.mi_measure = args.mi_measure
        config.ada_alpha = args.ada_alpha

        # *********** compacter ***********
        config.apply_compacter = args.apply_compacter

        config.input_dim = config.d_model
        config.non_linearity = "gelu_new"

        # Hypercomplex adapters parameters
        # compacter++ default hyperparameters
        config.reduction_factor = args.reduction_factor
        config.hypercomplex_adapter = args.hypercomplex_adapter
        config.hypercomplex_division = args.hypercomplex_division
        # whether PHM
        config.learn_phm = args.learn_phm
        config.shared_phm_rule = args.shared_phm_rule
        # whether low-rank PHM
        config.factorized_phm = args.factorized_phm
        config.factorized_phm_rule = False
        
        config.hypercomplex_nonlinearity = args.hypercomplex_nonlinearity
        config.phm_c_init = "normal"
        
        config.shared_W_phm = False  # ?
        config.phm_rank = args.phm_rank
        config.phm_init_range = 0.01
        config.kronecker_prod = False

        # Low-rank adapter.
        config.low_rank_adapter = args.low_rank_adapter
        config.low_rank_w_init = "glorot-uniform"
        config.low_rank_rank = args.low_rank_rank
        
        config.use_adapter_cross_attn = args.use_adapter_cross_attn
        config.use_adapter_self_attn = args.use_adapter_self_attn
        
        #
        config.apply_phm_moe = args.apply_phm_moe
        
        return config
    
    def create_model(self, model_class, config=None, **kwargs):
        print(f'Building Model at GPU')
        model_name = self.args.backbone
        model = model_class.from_pretrained(
            model_name,
            config=config,
            **kwargs
        )
        return model
    
    def create_tokenizer(self, **kwargs):
        if 't5' in self.args.tokenizer:
            if self.args.use_vision:
                # tokenizer_class = VLT5Tokenizer
                tokenizer_class = VLT5TokenizerFast
            else:
                from transformers import T5TokenizerFast
                # tokenizer_class = T5Tokenizer
                tokenizer_class = T5TokenizerFast
        
        tokenizer_name = self.args.backbone
        
        tokenizer = tokenizer_class.from_pretrained(
            tokenizer_name,
            max_length=self.args.max_text_length,
            do_lower_case=self.args.do_lower_case,
            **kwargs
        )
        
        return tokenizer
    
    def create_optimizer_and_scheduler(self):
        if self.verbose:
            print('Building Optimizer')
        
        lr_scheduler = None
        if 'adamw' in self.args.optim:
            batch_per_epoch = len(self.train_loader)
            t_total = batch_per_epoch // self.args.gradient_accumulation_steps \
                      * self.args.epochs
            warmup_ratio = self.args.warmup_ratio
            warmup_iters = int(t_total * warmup_ratio)
            if self.verbose:
                print("Batch per epoch: %d" % batch_per_epoch)
                print("Total Iters: %d" % t_total)
                print('Warmup ratio:', warmup_ratio)
                print("Warm up Iters: %d" % warmup_iters)
            
            no_decay = ["bias", "LayerNorm.weight"]
            params = list(filter(lambda p: p[1].requires_grad,
                                 self.model.named_parameters()))
            param_1 = [p for n, p in params
                       if not any(nd in n for nd in no_decay)]

            param_2 = [p for n, p in params
                       if any(nd in n for nd in no_decay)]
            # grad_para_2 = filter(lambda p: p.requires_grad, param_2)

            optimizer_grouped_parameters = [
                {
                    "params": param_1,
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": param_2,
                    "weight_decay": 0.0,
                },
            ]

            # optimizer_grouped_parameters = [
            #     {
            #         "params": [p for n, p in self.model.named_parameters() if
            #                    not any(nd in n for nd in no_decay)],
            #         "weight_decay": self.args.weight_decay,
            #     },
            #     {
            #         "params": [p for n, p in self.model.named_parameters() if
            #                    any(nd in n for nd in no_decay)],
            #         "weight_decay": 0.0,
            #     },
            # ]
            
            optim = AdamW(optimizer_grouped_parameters,
                          lr=self.args.lr, eps=self.args.adam_eps)
            lr_scheduler = get_linear_schedule_with_warmup(
                optim, warmup_iters, t_total)
        
        else:
            optim = self.args.optimizer(
                list(self.model.parameters()), self.args.lr)
        
        return optim, lr_scheduler
    
    def load_checkpoint(self, ckpt_path):
        state_dict = load_state_dict(ckpt_path, 'cpu')
        original_keys = list(state_dict.keys())
        for key in original_keys:
            if key.startswith("vis_encoder."):
                new_key = 'encoder.' + key[len("vis_encoder."):]
                state_dict[new_key] = state_dict.pop(key)
            if key.startswith("model.vis_encoder."):
                new_key = 'model.encoder.' + key[len("model.vis_encoder."):]
                state_dict[new_key] = state_dict.pop(key)
        results = self.model.load_state_dict(state_dict, strict=False)
        if self.verbose:
            print('Model loaded from ', ckpt_path)
            # pprint(results)
    
    def init_weights(self):
        def init_bert_weights(module):
            """ Initialize the weights."""
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=1)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        
        self.model.apply(init_bert_weights)
        self.model.init_weights()
    
    def predict(self):
        pass
    
    def evaluate(self):
        pass
    
    def save(self, name):
        if not os.path.isdir(self.args.output):
            os.makedirs(self.args.output, exist_ok=True)
        torch.save(self.model.state_dict(),
                   os.path.join(self.args.output, "%s.pth" % name))
    
    def load(self, path, loc=None):
        if loc is None and hasattr(self.args, 'gpu'):
            loc = f'cuda:{self.args.gpu}'
        state_dict = torch.load("%s.pth" % path, map_location=loc)
        
        original_keys = list(state_dict.keys())
        for key in original_keys:
            if key.startswith("module.vis_encoder."):
                new_key = 'module.encoder.' + key[len("module.vis_encoder."):]
                state_dict[new_key] = state_dict.pop(key)
            
            if key.startswith("module.model.vis_encoder."):
                new_key = 'module.model.encoder.' + key[len(
                    "module.model.vis_encoder."):]
                state_dict[new_key] = state_dict.pop(key)
        
        results = self.model.load_state_dict(state_dict, strict=False)
        if self.verbose:
            print('Model loaded from ', path)
            # pprint(results)
    
    def print_trainable_params_percentage(self, model):
        orig_param_size = sum(p.numel() for p in model.parameters())
        
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        trainable_size = count_parameters(model)
        
        percentage = trainable_size / orig_param_size * 100
        
        print(f"Trainable param percentage: "
              f"{percentage:.2f}% "
              f"({trainable_size / 1e6:.2f}/{orig_param_size / 1e6:.2f}) M")
        
        return percentage
    
    def freeze_whole_model(self):
        for n, p in self.model.named_parameters():
            p.requires_grad = False
    
    def unfreeze_adapter(self):
        for name, param in self.model.named_parameters():
            if 'adapter' in name:
                param.requires_grad = True
    
    def unfreeze_vis_embedding(self):
        targets = ["visual_embedding"]
        # unfreeze the parameters in targets anyway
        for n, p in self.model.named_parameters():
            if any(t in n for t in targets):
                p.requires_grad = True
                print(f"{n} is trainable...")
    
    def unfreeze_layer_norm(self):
        for name, sub_module in self.model.named_modules():
            # if self.args.unfreeze_layer_norms:
            if isinstance(sub_module, (T5LayerNorm, nn.LayerNorm)):
                print(f"{name} is trainable...")
                # this will not consider layer norms inside adapters then.
                # if len(name.split(".")) < 7:
                for param_name, param in sub_module.named_parameters():
                    param.requires_grad = True
    
    def unfreeze_parameters(self):
        targets = ["visual_embedding"]
        # unfreeze the parameters in targets anyway
        for n, p in self.model.named_parameters():
            if any(t in n for t in targets):
                p.requires_grad = True
                print(f"{n} is trainable...")
        
        # if self.args.unfreeze_language_model:
        #     targets = ["lm_head", "shared"]
        #     for n, p in self.model.named_parameters():
        #         if any(t in n for t in targets):
        #             p.requires_grad = True
        #             print(f"{n} is trainable...")
        #     for name, sub_module in self.model.named_modules():
        #         if isinstance(sub_module, (
        #                 modeling_vl_t5.T5Stack, modeling_vl_t5.JointEncoder)):
        #             print(f"{name} is trainable...")
        #             for param_name, param in sub_module.named_parameters():
        #                 param.requires_grad = True
        #
        # if self.args.unfreeze_lm_head:
        #     # shared and lm_head share the same weight
        #     targets = ["lm_head", "shared"]
        #     for n, p in self.model.named_parameters():
        #         if any(t in n for t in targets):
        #             p.requires_grad = True
        #             print(f"{n} is trainable...")
        
        for name, sub_module in self.model.named_modules():
            if self.args.unfreeze_layer_norms:
                if isinstance(sub_module, (T5LayerNorm, nn.LayerNorm)):
                    print(f"{name} is trainable...")
                    # this will not consider layer norms inside adapters then.
                    # if len(name.split(".")) < 7:
                    for param_name, param in sub_module.named_parameters():
                        param.requires_grad = True
            
            if self.args.unfreeze_batch_norms:
                if isinstance(sub_module, nn.BatchNorm2d):
                    print(f"{name} is trainable...")
                    # if len(name.split(".")) < 7:
                    # this will not consider layer norms inside adapters then.
                    for param_name, param in sub_module.named_parameters():
                        param.requires_grad = True
            
            # if self.args.ffn_mode is not None or self.args.attn_mode is not None:
            #     if isinstance(sub_module, AdapterLayer):
            #         print(f"{name} is trainable...")
            #         for param_name, param in sub_module.named_parameters():
            #             param.requires_grad = True


