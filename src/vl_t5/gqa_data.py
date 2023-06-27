import os
import re
import json
import random
import h5py
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from transformers import T5TokenizerFast
from vl_t5.tokenization import VLT5TokenizerFast


def get_loader(args, split='train', mode='train',
               batch_size=32, workers=4,
               shuffle=False, drop_last=False, distributed=False):
    
    _dset = GQADataset(split)
    
    dataset = GQAFineTuneDataset(
        split=split,
        raw_dataset=_dset,
        args=args,
        mode=mode
    )
    if distributed:
        sampler = torch.utils.data.DistributedSampler(
            dataset,
            num_replicas=args.num_tasks,
            rank=args.global_rank,
            shuffle=shuffle
        )
    else:
        sampler = None

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        num_workers=workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=drop_last,
        collate_fn=dataset.collate_fn,
    )
    
    loader.evaluator = GQAEvaluator(_dset)
    loader.task = 'gqa'
    
    return loader


class GQAFineTuneDataset(Dataset):
    def __init__(self, split='train,valid', raw_dataset=None,
                 args=None, mode='train'):
        super().__init__()
        self.args = args
        self.mode = mode

        if 't5' in self.args.backbone:
            if self.args.use_vision:
                self.tokenizer = VLT5TokenizerFast.from_pretrained(
                    args.backbone,
                    # max_length=self.args.max_text_length,
                    do_lower_case=self.args.do_lower_case
                )
            else:
                self.tokenizer = T5TokenizerFast.from_pretrained(
                    args.backbone,
                    # max_length=self.args.max_text_length,
                    do_lower_case=self.args.do_lower_case
                )

        self.raw_dataset = raw_dataset
        self.data = self.raw_dataset.data

        self.sources = split.split(',')
        if 'train' in split:
            random.seed(args.dataseed)
            random.shuffle(self.data)
            if 'train' in split and mode == 'train':
                self.data = self.data[:args.k]
            elif 'train' in split and mode == 'val':
                self.data = self.data[args.k:2 * args.k]
        print('Data sources: ', len(self.data))

        if self.args.feat_type == 'roi':
            if "train" in self.sources:
                self.roi_h5_file = h5py.File(
                    "data/vg_imgfeat/vg_gqa_obj36.h5", 'r')
            else:
                self.roi_h5_file = h5py.File(
                    "data/vg_imgfeat/gqa_testdev_obj36.h5", 'r')
        
        self.n_boxes = args.n_boxes
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        
        out_dict = {'args': self.args}

        datum = self.data[idx]
        
        # ##### Image ######
        if self.args.use_vision:
            img_id = datum['img_id']
            out_dict['img_id'] = img_id

            if self.args.feat_type == 'clip':
                path = os.path.join(self.clip_h5_dir, f"{img_id}.h5")
                with h5py.File(path, 'r') as f:
                    feats = f[f"{img_id}/features"][...]
                    out_dict['vis_feats'] = feats  # (L, D)
        
                    boxes = torch.zeros(feats.shape[0], 4)  # (L, 4)
                    out_dict['boxes'] = boxes
            else:
                # if self.mode == "test":
                #     img_id = img_id.split("_")[-1]
                # img_id = int(img_id.split("_")[-1])
                feats = np.zeros(shape=(36, 2048), dtype=np.float32)
                try:
                    self.roi_h5_file[f'{img_id}/features'].read_direct(feats)
                except KeyError:
                    print('img_id', img_id)
                    print(datum)
                    exit()
                out_dict['vis_feats'] = torch.from_numpy(feats)
            
            # Normalize the boxes (to 0 ~ 1)
            img_h = self.roi_h5_file[f'{img_id}/img_h'][()]
            img_w = self.roi_h5_file[f'{img_id}/img_w'][()]
            boxes = self.roi_h5_file[f'{img_id}/boxes'][()]  # (x1, y1, x2, y2)
            boxes[:, (0, 2)] /= img_w
            boxes[:, (1, 3)] /= img_h
            np.testing.assert_array_less(boxes, 1 + 1e-5)
            np.testing.assert_array_less(-boxes, 0 + 1e-5)
            boxes = torch.from_numpy(boxes)
            boxes.clamp_(min=0.0, max=1.0)
            out_dict['boxes'] = boxes
        
        # ##### Text #####
        if 'sent' in datum:
            sent = datum['sent']
        elif 'question' in datum:
            sent = datum['question']

        if self.args.use_fewvlm_prompt:
            if self.args.prompt == 0:
                input_ids = self.tokenizer.encode(
                    sent, max_length=20,
                    truncation=True
                )
            elif self.args.prompt == 1:
                input_ids = self.tokenizer.encode(
                    f'{sent} <extra_id_0>',
                    max_length=20,
                    truncation=True
                )
            elif self.args.prompt == 2:
                input_ids = self.tokenizer.encode(
                    f'question: {sent} answer: ',
                    max_length=20,
                    truncation=True
                )
            elif self.args.prompt == 3:
                input_ids = self.tokenizer.encode(
                    f'question: {sent} answer: <extra_id_0>',
                    max_length=20,
                    truncation=True
                )
        else:
            # VL-T5 original
            input_ids = self.tokenizer.encode(
                f'vqa: {sent}', max_length=20, truncation=True
            )

        out_dict['question_id'] = datum['question_id']

        out_dict['sent'] = sent
        out_dict['input_ids'] = torch.LongTensor(input_ids)
        out_dict['input_length'] = len(input_ids)
        
        if 'label' in datum:
            label = datum['label']
            out_dict['label'] = label
            
            answers = []
            scores = []
            for a, s in label.items():
                answers.append(a)
                scores.append(s)
            
            score_sum = sum(scores)
            
            if score_sum == 0:
                answer = ''
                score = 0.
            else:
                prob = [score / score_sum for score in scores]
                choice = np.random.multinomial(1, prob).argmax()
                answer = answers[choice]
                score = scores[choice]
                assert len(answer) > 0, (sent, label, choice, answer)
            
            out_dict['answer'] = answer
            out_dict['score'] = score
            out_dict['all_answers'] = answers
            
            if sum(scores) > 0:
                best_answers = []
                best_score = max(scores)
                for a, s in label.items():
                    if s == best_score and s > 0:
                        best_answers.append(a)
                out_dict['best_answers_tokenized'] = [self.tokenizer.encode(a) for
                                                      a in best_answers]
            else:
                out_dict['best_answers_tokenized'] = [[]]

            if self.args.use_fewvlm_prompt:
                target_ids = self.tokenizer.encode(
                    f'<extra_id_0> {answer}'
                )
            else:
                # VL-T5 original
                target_ids = self.tokenizer.encode(
                    answer
                )

            out_dict['target_ids'] = torch.LongTensor(target_ids)
            out_dict['target_length'] = len(target_ids)
        
        return out_dict
    
    def collate_fn(self, batch):
        batch_entry = {}
        
        args = batch[0]['args']
        
        B = len(batch)
        
        S_W_L = max(entry['input_length'] for entry in batch)
        input_ids = torch.ones(B, S_W_L,
                               dtype=torch.long) * self.tokenizer.pad_token_id
        
        if args.use_vision:
            V_L = len(batch[0]['boxes'])
            feat_dim = batch[0]['vis_feats'].shape[-1]
            
            boxes = torch.zeros(B, V_L, 4, dtype=torch.float)
            vis_feats = torch.zeros(B, V_L, feat_dim, dtype=torch.float)
        
        if 'target' in batch[0]:
            # targets = []
            targets = torch.zeros(B, len(batch[0]['target']), dtype=torch.float)
        if 'target_ids' in batch[0]:
            T_W_L = max(entry['target_length'] for entry in batch)
            target_ids = torch.ones(B, T_W_L,
                                    dtype=torch.long) * self.tokenizer.pad_token_id
        
        sentences = []
        question_ids = []
        answers = []
        all_answers = []
        all_answers_tokenized = []
        best_answers_tokenized = []
        img_ids = []
        img_paths = []
        labels = []
        scores = []
        
        for i, entry in enumerate(batch):
            input_ids[i, :entry['input_length']] = entry['input_ids']
            
            if args.use_vision:
                boxes[i] += entry['boxes']
                vis_feats[i] += entry['vis_feats']
                # img_ids.append(entry['img_id'])
                # img_paths.append(entry['img_path'])
            
            if 'target_ids' in entry:
                target_ids[i, :entry['target_length']] = entry['target_ids']
            
            if 'target' in entry:
                targets[i] += entry['target']
                # targets.append(entry['target'])
            
            sentences.append(entry['sent'])
            question_ids.append(entry['question_id'])
            if 'answer' in entry:
                answers.append(entry['answer'])
            if 'all_answers' in entry:
                all_answers.append(entry['all_answers'])
            if 'all_answers_tokenized' in entry:
                all_answers_tokenized.append(entry['all_answers_tokenized'])
            if 'best_answers_tokenized' in entry:
                best_answers_tokenized.append(entry['best_answers_tokenized'])
            if 'score' in entry:
                scores.append(entry['score'])
            
            if 'label' in entry:
                labels.append(entry['label'])
        
        batch_entry['input_ids'] = input_ids
        if 'target_ids' in batch[0]:
            word_mask = target_ids != self.tokenizer.pad_token_id
            target_ids[~word_mask] = -100
            batch_entry['target_ids'] = target_ids
        if 'target' in batch[0]:
            # targets = torch.stack(targets, dim=0)
            batch_entry['targets'] = targets
        
        if args.use_vision:
            batch_entry['boxes'] = boxes
            batch_entry['vis_feats'] = vis_feats
            # batch_entry['img_id'] = img_ids
            # batch_entry['img_paths'] = img_paths
        
        batch_entry['sent'] = sentences
        batch_entry['question_ids'] = question_ids
        batch_entry['answers'] = answers
        batch_entry['all_answers'] = all_answers
        batch_entry['all_answers_tokenized'] = all_answers_tokenized
        batch_entry['best_answers_tokenized'] = best_answers_tokenized
        batch_entry['scores'] = torch.FloatTensor(scores)
        batch_entry['labels'] = labels
        
        batch_entry['task'] = 'gqa'
        
        return batch_entry


class GQADataset:
    """
    A GQA data example in json file:
    {
        "img_id": "2375429",
        "label": {
            "pipe": 1.0
        },
        "question_id": "07333408",
        "sent": "What is on the white wall?"
    }
    """
    
    def __init__(self, splits, mode='train'):
        self.splits = splits.split(',')

        # Loading datasets to data
        self.data = []
        for split in self.splits:
            self.data.extend(json.load(open(os.path.join(
                "data/annotation/gqa", f'{split}.json'), 'r'
            )))
        
        # List to dict (for evaluation and others)
        self.id2datum = {
            datum['question_id']: datum
            for datum in self.data
        }
        
        # Answers
        self.ans2label = json.load(
            open(f"data/annotation/gqa/trainval_ans2label.json"))
        self.label2ans = json.load(
            open(f"data/annotation/gqa/trainval_label2ans.json"))
        
        assert len(self.ans2label) == len(self.label2ans)
        for ans, label in self.ans2label.items():
            assert self.label2ans[label] == ans
    
    @property
    def num_answers(self):
        return len(self.ans2label)
    
    def __len__(self):
        return len(self.data)


class GQAEvaluator:
    def __init__(self, dataset: GQADataset):
        self.dataset = dataset
        self.punct = [';', r"/", '[', ']', '"', '{', '}',
                      '(', ')', '=', '+', '\\', '_', '-',
                      '>', '<', '@', '`', ',', '?', '!']
        self.periodStrip = re.compile("(?!<=\d)(\.)(?!\d)")
        self.commaStrip = re.compile("(\d)(\,)(\d)")
        self.articles = ['a', 'an', 'the']
    
    def processArticle(self, inText):
        outText = []
        tempText = inText.lower().split()
        for word in tempText:
            if word not in self.articles:
                outText.append(word)
        outText = " ".join(outText)
        return outText
    
    def processPunctuation(self, inText):
        outText = inText
        for p in self.punct:
            if (p + ' ' in inText or ' ' + p in inText) or (
                    re.search(self.commaStrip, inText) != None):
                outText = outText.replace(p, '')
            else:
                outText = outText.replace(p, ' ')
        outText = self.periodStrip.sub("",
                                       outText,
                                       re.UNICODE)
        return outText
    
    def evaluate(self, quesid2ans: dict):
        score = 0.
        for quesid, ans in quesid2ans.items():
            datum = self.dataset.id2datum[quesid]
            label = datum['label']
            ans = self.processPunctuation(ans)
            ans = self.processArticle(ans)
            if ans in label:
                score += label[ans]
        return score / len(quesid2ans)
    
    def dump_result(self, quesid2ans: dict, path):
        """
        Dump the result to a GQA-challenge submittable json file.
        GQA json file submission requirement:
            results = [result]
            result = {
                "questionId": str,      # Note: it's a actually an int number but the server requires an str.
                "prediction": str
            }
        :param quesid2ans: A dict mapping question id to its predicted answer.
        :param path: The file path to save the json file.
        :return:
        """
        with open(path, 'w') as f:
            result = []
            for ques_id, ans in quesid2ans.items():
                datum = self.dataset.id2datum[ques_id]
                label = datum['label']
                result.append({
                    'questionId': ques_id,
                    'prediction': ans,
                    'label': label
                })
            json.dump(result, f, indent=4, sort_keys=True)
