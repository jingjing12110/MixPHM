import ast
import yaml
import random
import pprint
import argparse
import numpy as np

import torch


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args(parse=True, **optional_kwargs):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--seed', type=int, default=9595,
        help='random seed [0, ]',
    )
    parser.add_argument(
        '--seeds', type=str, default="13, 21, 42, 87, 100",
        help='random seed [13, 21, 42, 87, 100]',
    )
    parser.add_argument(
        '--dataseed', type=int, default=9595, help='random seed'
    )
    parser.add_argument(
        "--es_patience", type=int, default=10,
        help="patience of early stopping, k<100: 20, k>=100: 10"
    )

    # Data Splits
    # parser.add_argument("--train", default='karpathy_train')
    # parser.add_argument("--valid", default='karpathy_val')
    # parser.add_argument("--test", default="karpathy_test")
    parser.add_argument("--train", default='train')
    parser.add_argument("--valid", default='train')
    parser.add_argument("--test", default="minival,nominival")

    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--submit', action='store_true')

    # Quick experiments
    parser.add_argument('--train_topk', type=int, default=-1)
    parser.add_argument('--valid_topk', type=int, default=-1)

    # Checkpoint
    parser.add_argument('--output', type=str, default='snap/test')
    parser.add_argument('--load', type=str,
                        default='snap/pretrain/VLT5-novqa/Epoch30_base',
                        help='Load the model (usually the fine-tuned model).')
    parser.add_argument('--from_scratch', action='store_true')

    # CPU/GPU
    parser.add_argument("--multiGPU", action='store_const', default=False,
                        const=True)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument("--distributed", action='store_true')
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument('--local_rank', type=int, default=-1)

    # Model Config
    parser.add_argument('--backbone', type=str, default='t5-base')
    parser.add_argument('--tokenizer', type=str, default=None)

    parser.add_argument('--feat_dim', type=float, default=2048)
    parser.add_argument('--pos_dim', type=float, default=4)

    parser.add_argument('--use_vision', default=True, type=str2bool)
    parser.add_argument('--use_vis_order_embedding', default=True, type=str2bool)
    parser.add_argument('--use_vis_layer_norm', default=True, type=str2bool)
    parser.add_argument('--individual_vis_layer_norm', default=True,
                        type=str2bool)
    parser.add_argument('--share_vis_lang_layer_norm', action='store_true')

    parser.add_argument('--n_boxes', type=int, default=36)
    parser.add_argument('--max_n_boxes', type=int, default=36)
    parser.add_argument('--max_text_length', type=int, default=20)

    # Training
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--valid_batch_size', type=int, default=100)
    parser.add_argument('--optim', default='adamw')
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--clip_grad_norm', type=float, default=5)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--adam_eps', type=float, default=1e-6)
    parser.add_argument('--adam_beta1', type=float, default=0.9)
    parser.add_argument('--adam_beta2', type=float, default=0.999)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--dropout', type=float, default=0.1)

    parser.add_argument("--losses",
                        default='lm,obj,attr,feat',
                        type=str)

    parser.add_argument('--log_train_accuracy', action='store_true')
    parser.add_argument('--n_ground', type=int, default=1)
    parser.add_argument("--wordMaskRate", dest='word_mask_rate', default=0.15,
                        type=float)
    parser.add_argument("--objMaskRate", dest='obj_mask_rate', default=0.15,
                        type=float)

    # Inference
    parser.add_argument('--num_beams', type=int, default=5)
    parser.add_argument('--gen_max_length', type=int, default=20)

    # Data
    parser.add_argument('--caption_only', action='store_true')
    parser.add_argument('--coco_only', action='store_true')
    parser.add_argument('--caption_cocoonly', default=True, type=str2bool)

    parser.add_argument('--do_lower_case', action='store_true')
    parser.add_argument('--oscar_tags', action='store_true')
    parser.add_argument('--prefix', type=str, default=None)

    # Pretraining
    parser.add_argument('--ground_upsample', type=int, default=1)
    parser.add_argument('--ground_weight', type=int, default=1)
    parser.add_argument('--itm_cocoonly', default=True, type=str2bool)
    parser.add_argument('--single_vqa_prefix', action='store_true')

    # COCO Caption
    parser.add_argument('--no_prefix', action='store_true')

    # VQA
    parser.add_argument("--raw_label", action='store_true')
    parser.add_argument("--answer_normalize", action='store_true')
    parser.add_argument("--classifier", action='store_true')
    parser.add_argument("--test_answerable", action='store_true')

    # RefCOCOg
    parser.add_argument('--RefCOCO_GT', action='store_true')
    parser.add_argument('--RefCOCO_BUTD', action='store_true')
    parser.add_argument("--shuffle_boxes", action='store_true')

    # Multitask
    parser.add_argument("--multitask_sampling", type=str, default='roundrobin')
    parser.add_argument("--tasks", type=str, default='')

    # Etc.
    parser.add_argument('--comment', type=str, default='')
    parser.add_argument("--dry", action='store_true')

    # TODO: added
    parser.add_argument(
        '--device', default='cuda')
    parser.add_argument(
        '--world_size', default=1, type=int,
        help='number of distributed processes')
    parser.add_argument(
        '--dist_url', default='env://',
        help='url used to set up distributed training')
    
    parser.add_argument(
        '--data_name', default='okvqa', type=str)
    parser.add_argument(
        '--feat_type', default='roi', type=str, help='roi or clip')
    parser.add_argument(
        '--k', default=16, type=int, help='k-shot learning')
    
    # MoEBERT
    parser.add_argument(
        "--apply_moebert", default=False, type=ast.literal_eval,
    )
    parser.add_argument(
        "--moebert_expert_num", default=4, type=int,
    )
    parser.add_argument(
        "--moebert_route_method", default="gate-token", type=str,
    )
    parser.add_argument(
        "--moebert_route_hash_list", default=None, type=str,
    )
    parser.add_argument(
        "--moebert_expert_dim", default=768, type=int,
    )
    parser.add_argument("--moebert_share_importance")
    
    # AdaMix
    parser.add_argument(
        "--apply_lora", default=False, type=ast.literal_eval,
        help="Whether to apply LoRA or not."
    )
    parser.add_argument(
        "--lora_alpha", default=1., type=int,
        help="LoRA alpha"
    )
    parser.add_argument(
        "--lora_r", default=8, type=int,
        help="LoRA r"
    )
    parser.add_argument(
        "--lora_path", default=None, type=str,
        help="The file path of LoRA parameters."
    )
    
    parser.add_argument(
        "--apply_adapter", default=False, type=ast.literal_eval,
        help="Whether to apply adapter or not."
    )
    parser.add_argument(
        "--adapter_path", default=None, type=str,
        help="The file path of adapter parameters."
    )
    parser.add_argument(
        "--adapter_type", default='houlsby', type=str,
        help="houlsby or pfeiffer"
    )
    parser.add_argument(
        "--adapter_size", default=64, type=int,
        help="8, 16, 32, 64"
    )
    
    parser.add_argument(
        "--apply_expert_soup", default=False, type=ast.literal_eval,
        help="Whether to apply expert soup or not."
    )
    parser.add_argument(
        "--expert_soup_path", default=None, type=str,
        help="The file path of adapter parameters."
    )
    parser.add_argument(
        "--num_experts", default=4, type=int,
        help="The file path of adapter parameters."
    )
    parser.add_argument(
        "--inference_level", default=3, type=int,
        help="1, 3, 4"
    )
    parser.add_argument(
        "--sharing_down", default=0, type=int,
        help="Whether to sharing down projection for expert or not."
    )
    parser.add_argument(
        "--sharing_up", default=0, type=int,
        help="Whether to sharing down projection for expert or not."
    )
    parser.add_argument(
        "--sparsity", default=0.75, type=float,
        help="Whether to sharing down projection for expert or not."
    )
    parser.add_argument(
        "--use_consistency_loss", default=False, type=ast.literal_eval,
    )

    parser.add_argument(
        "--apply_bitfit", default=False, type=ast.literal_eval,
        help="Whether to apply bitfit or not."
    )
    
    # MoE
    parser.add_argument(
        "--add_gate", default=False, type=ast.literal_eval,
    )
    parser.add_argument(
        "--apply_moe", default=False, type=ast.literal_eval,
    )
    # Adapter-based MoE
    parser.add_argument(
        "--apply_adapter_moe", default=False, type=ast.literal_eval,
    )
    
    # =================================================================
    parser.add_argument(
        "--use_fewvlm_prompt", default=False, type=ast.literal_eval,
    )
    parser.add_argument(
        "--prompt", default=3, type=int,
    )

    # TODO
    parser.add_argument(
        "--use_adapter_r2loss", default=False, type=ast.literal_eval
    )
    parser.add_argument(
        "--red_w", default=0.1, type=float
    )
    parser.add_argument(
        "--using_mi_loss", default=False, type=ast.literal_eval
    )
    parser.add_argument(
        "--align_w", default=0.1, type=float
    )
    parser.add_argument(
        "--mi_measure", default="JSD", type=str,
    )
    parser.add_argument(
        "--ada_alpha", default=1.0, type=float
    )
    
    parser.add_argument(
        "--use_adapter_cross_attn", default=False, type=ast.literal_eval
    )
    parser.add_argument(
        "--use_adapter_self_attn", default=False, type=ast.literal_eval
    )
    
    # compacter
    parser.add_argument(
        "--apply_compacter", default=False, type=ast.literal_eval
    )
    parser.add_argument(
        "--hypercomplex_adapter", default=False, type=ast.literal_eval
    )
    parser.add_argument(
        "--hypercomplex_division", default=4, type=int,
        help="n, the hypercomplex division number"
    )
    parser.add_argument(
        "--reduction_factor", default=12, type=int,
        help="bottleneck=d_model/reduction_factor [32, 16, 8]"
    )
    parser.add_argument(
        "--factorized_phm", default=False, type=ast.literal_eval,
        help='Whether to factorize W into low rank product.'
    )
    parser.add_argument(
        "--factorized_phm_rule", default=False, type=ast.literal_eval,
        help='If set, If set, it factorizes the shared weights for the W in'
             'hypercomplex adapters.'
    )
    parser.add_argument(
        "--learn_phm", default=False, type=ast.literal_eval,
        help="If set, learns the phm rules in Hypercomplex adapters."
    )
    parser.add_argument(
        "--shared_phm_rule", default=False, type=ast.literal_eval,
        help="Whether the phm rule is shared across layer."
    )
    parser.add_argument(
        "--phm_rank", default=4, type=int,
    )
    parser.add_argument(
        "--hypercomplex_nonlinearity", default="glorot-uniform", type=str
    )
    
    parser.add_argument(
        "--low_rank_adapter", default=False, type=ast.literal_eval
    )
    parser.add_argument(
        "--low_rank_rank", default=1, type=int,
    )
    
    parser.add_argument(
        "--apply_phm_moe", default=False, type=ast.literal_eval
    )

    parser.add_argument(
        "--val_interval", default=10, type=int,
    )

    # =================================================================
    
    # Parse the arguments.
    if parse:
        args = parser.parse_args()
    else:
        args = parser.parse_known_args()[0]

    # Namespace => Dictionary
    kwargs = vars(args)
    kwargs.update(optional_kwargs)

    args = Config(**kwargs)

    # # Bind optimizer class.
    # verbose = False
    # args.optimizer = get_optimizer(args.optim, verbose=verbose)

    # Set seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    return args


class Config(object):
    def __init__(self, **kwargs):
        """Configuration Class: set kwargs as class attributes with setattr"""
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    @property
    def config_str(self):
        return pprint.pformat(self.__dict__)
    
    def __repr__(self):
        """Pretty-print configurations in alphabetical order"""
        config_str = 'Configurations\n'
        config_str += self.config_str
        return config_str
    
    def save(self, path):
        with open(path, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)
    
    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            kwargs = yaml.load(f)
        
        return Config(**kwargs)


if __name__ == '__main__':
    args = parse_args(True)
