import torch
import torch.nn as nn

from vl_t5.modeling_t5 import VLT5


class VLT5VQA(VLT5):
    def __init__(self, config, num_answers=None, label2ans=None):
        super().__init__(config)
        
        self.num_answers = num_answers
        self.label2ans = label2ans
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def train_step(self, batch):
        device = next(self.parameters()).device
        vis_feats = batch['vis_feats'].to(device)  # [bs, 36, 2048]
        input_ids = batch['input_ids'].to(device)  # [bs, len_seq]
        vis_pos = batch['boxes'].to(device)  # [bs, 36, 4]

        lm_labels = batch["target_ids"].to(device)

        output = self(
            input_ids=input_ids,
            vis_inputs=(vis_feats, vis_pos),
            labels=lm_labels,
            use_cache=True,
            return_dict=True,
        )
        assert 'loss' in output

        lm_mask = (lm_labels != -100).float()
        B, L = lm_labels.size()

        loss = output['loss'][0].view(B, L) * lm_mask
        loss = loss.sum(dim=1) / lm_mask.sum(dim=1).clamp(min=1)  # B
        loss = loss * batch['scores'].to(device=device)
        loss = loss.mean()

        result = {
            'task': loss.item(),
        }

        if output['loss'][1] is not None:
            loss += output['loss'][1]
            result['consistency'] = output['loss'][1].item()

        if "ada_r2loss" in output.keys():
            loss += output['ada_r2loss']
            result['ada_r2loss'] = output['ada_r2loss'].item()

        if "ada_red_loss" in output.keys():
            loss += output['ada_red_loss']
            result['ada_red'] = output['ada_red_loss'].item()

        if "ada_align_loss" in output.keys():
            loss += output['ada_align_loss']
            result['ada_align'] = output['ada_align_loss'].item()

        if "balance_loss" in output.keys():
            loss += output['balance_loss']
            result['balance_loss'] = output['balance_loss'].item()

        result['loss'] = loss

        return result
    
    @torch.no_grad()
    def test_step(self, batch, **kwargs):
        self.eval()
        device = next(self.parameters()).device
        vis_feats = batch['vis_feats'].to(device)
        input_ids = batch['input_ids'].to(device)
        vis_pos = batch['boxes'].to(device)
        
        result = {}

        output = self.generate(
            input_ids=input_ids,
            vis_inputs=(vis_feats, vis_pos),
            use_cache=True,
            **kwargs
        )
        generated_sents = self.tokenizer.batch_decode(
            output, skip_special_tokens=True
        )
        result['token_ids'] = output
        result['pred_ans'] = generated_sents
        return result
