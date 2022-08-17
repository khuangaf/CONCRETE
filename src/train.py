from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
import argparse
import torch
import torch.nn as nn
import numpy as np
import json
from tqdm import tqdm
import random
import time
import os

from model import BERTModelForClassification
from data import PropaFakeDataset
from constants import XFACT_LABEL2IDX, XFACT_IDX2LABEL
from xfact_eval import calculate_fscore
from utils import read_data

seed = 42
# fix random seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.enabled = False


# configuration
parser = argparse.ArgumentParser()
parser.add_argument('--max_sequence_length', default=512, type=int)
parser.add_argument('--model_name', default='bert-base-multilingual-cased')
parser.add_argument('--data_dir', default='../data/x-fact')
parser.add_argument('--retrieval_dir', default='../CORA/mDPR/retrieved_docs/')
parser.add_argument('--warmup_epoch', default=5, type=int)
parser.add_argument('--max_epoch', default=12, type=int)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--eval_batch_size', default=4, type=int)
parser.add_argument('--accumulate_step', default=8, type=int)
parser.add_argument('--n_candidates', default=5, type=int)
parser.add_argument('--n_classes', default=7, type=int)
parser.add_argument('--dataset', default='x-fact', type=str)
parser.add_argument('--output_dir', required=True)
parser.add_argument('--disable_retrieval', action="store_true")
parser.add_argument('--use_mbert_retrieval', action="store_true")
parser.add_argument('--use_mdpr_retrieval', action="store_true")
parser.add_argument('--use_bm25_retrieval', action="store_true")
parser.add_argument('--use_google_search', action="store_true")
parser.add_argument('--do_monolingual_retrieval', action="store_true")
parser.add_argument('--do_monolingual_eval', action="store_true")
parser.add_argument('--learning_rate', default=5e-05, type=float)
parser.add_argument('--remove_lang', default=None, type=str)
parser.add_argument('--training_subset_langs', nargs='+', default=[])
parser.add_argument('--training_subset_portion', default=1.0, type=float)

timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
args = parser.parse_args()

if sum([args.use_mbert_retrieval, args.use_mdpr_retrieval, args.use_bm25_retrieval, args.use_google_search]) > 1:
    raise ValueError("only one of use_mbert_retrieval, use_mdpr_retrieval and use_bm25_retrieval can be true.")

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

if args.dataset == 'x-fact':
    label2idx = XFACT_LABEL2IDX
    idx2label = XFACT_IDX2LABEL
    train_path = os.path.join(args.data_dir,'train.all.tsv')
    dev_path = os.path.join(args.data_dir,'dev.all.tsv')
    test_path = os.path.join(args.data_dir,'test.all.tsv')
    if args.use_mbert_retrieval:
        train_retrieval_path = os.path.join(args.retrieval_dir,'train.all.mbert.json')
        dev_retrieval_path = os.path.join(args.retrieval_dir,'dev.all.mbert.json')
        test_retrieval_path = os.path.join(args.retrieval_dir,'test.all.mbert.json')
    elif args.use_mdpr_retrieval:
        train_retrieval_path = os.path.join(args.retrieval_dir,'train.all.json')
        dev_retrieval_path = os.path.join(args.retrieval_dir,'dev.all.json')
        test_retrieval_path = os.path.join(args.retrieval_dir,'test.all.json')
    elif args.use_bm25_retrieval:
        train_retrieval_path = os.path.join(args.retrieval_dir,'train.all.bm25.json')
        dev_retrieval_path = os.path.join(args.retrieval_dir,'dev.all.bm25.json')
        test_retrieval_path = os.path.join(args.retrieval_dir,'test.all.bm25.json')
    else:
        if args.remove_lang is None:
            train_retrieval_path = os.path.join(args.retrieval_dir,'train.all.xict.json')
            dev_retrieval_path = os.path.join(args.retrieval_dir,'dev.all.xict.json')
            test_retrieval_path = os.path.join(args.retrieval_dir,'test.all.xict.json')
        else:
            train_retrieval_path = os.path.join(args.retrieval_dir,f'train.all.xict-rm_{args.remove_lang}.json')
            dev_retrieval_path = os.path.join(args.retrieval_dir,f'dev.all.xict-rm_{args.remove_lang}.json')
            test_retrieval_path = os.path.join(args.retrieval_dir,f'test.all.xict-rm_{args.remove_lang}.json')
    dev_examples = read_data(dev_path, args, is_train_dev=True)#.iterrows()
    test_examples = read_data(test_path, args)#.iterrows()
else:
    raise NotImplementedError

output_dir = os.path.join(args.output_dir, timestamp)
os.makedirs(output_dir)

model = BERTModelForClassification(args.model_name, args).cuda()

# init loader

train_set = PropaFakeDataset(train_path, train_retrieval_path, args, label2idx, idx2label, is_train_dev=True)
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=train_set.collate_fn)
dev_set = PropaFakeDataset(dev_path, dev_retrieval_path, args, label2idx, idx2label, is_train_dev=True)
dev_loader = DataLoader(dev_set, batch_size=args.eval_batch_size, shuffle=False, collate_fn=dev_set.collate_fn)
test_set = PropaFakeDataset(test_path, test_retrieval_path, args, label2idx, idx2label)
test_loader = DataLoader(test_set, batch_size=args.eval_batch_size, shuffle=False, collate_fn=test_set.collate_fn)



# define loss
critera = nn.CrossEntropyLoss()


state = dict(model=model.state_dict())

# optimizer
param_groups = [
    {
        'params': [p for n, p in model.named_parameters() if n.startswith('bert')],
        'lr': args.learning_rate, 'weight_decay': 1e-05
    },
    {
        'params': [p for n, p in model.named_parameters() if not n.startswith('bert')],
        'lr': 1e-3, 'weight_decay': 0.001
    },
    
]

batch_num = len(train_set) // (args.batch_size * args.accumulate_step)
+ (len(train_set) % (args.batch_size * args.accumulate_step) != 0)

optimizer = AdamW(params=param_groups)
schedule = get_linear_schedule_with_warmup(optimizer,
                                           num_warmup_steps=batch_num*args.warmup_epoch,
                                           num_training_steps=batch_num*args.max_epoch)

best_dev_f1 = 0
model_path = os.path.join(output_dir,'best.pt')
for epoch in range(args.max_epoch):
    training_loss = 0
    model.train()
    for batch_idx, (input_ids, attn_mask, candidate_input_ids, candidate_attention_mask, labels) in enumerate(tqdm(train_loader)):        
        
        outputs = model(input_ids, attention_mask=attn_mask, 
                # candidate_embeddings = candidate_embeddings, 
                candidate_input_ids = candidate_input_ids,
                candidate_attention_mask = candidate_attention_mask,
                # indexer = indexer,
                epoch=epoch)
        outputs = outputs.view(-1, args.n_classes)

        # loss
        
        loss = critera(outputs, labels)

        
        loss.backward()
        training_loss += loss.item()
        if (batch_idx + 1) % args.accumulate_step == 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 5.0)
        
            optimizer.step()
            schedule.step()
            optimizer.zero_grad()

        
    # train the last batch
    if batch_num % args.accumulate_step != 0:
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), 5.0)
        optimizer.step()
        schedule.step()
        optimizer.zero_grad()

    # validation
    with torch.no_grad():
        model.eval()
        dev_outputs = []
        dev_labels = []
        for _, (input_ids, attn_mask, candidate_input_ids, candidate_attention_mask, labels) in enumerate(tqdm(dev_loader)):
            outputs = model(input_ids, attention_mask=attn_mask, 
                # candidate_embeddings = candidate_embeddings, 
                candidate_input_ids = candidate_input_ids,
                candidate_attention_mask = candidate_attention_mask,
                # indexer = indexer
                )
            outputs = outputs.view(-1, args.n_classes)
            dev_outputs.append(outputs) 
            dev_labels.append(labels)
        dev_outputs = torch.cat(dev_outputs, dim=0) # n_sample, n_class
        dev_labels = torch.cat(dev_labels, dim=0) # n_sample,
        
        dev_outputs = dev_outputs.argmax(dim=1) # n_sample
        
        assert len(dev_outputs) == len(dev_labels) == len(dev_examples), (len(dev_outputs), len(dev_labels), len(dev_examples))
        dev_f1 = calculate_fscore(dev_outputs.detach().cpu().numpy(), dev_labels.detach().cpu().numpy(), dev_examples.itertuples())
        
        if dev_f1 > best_dev_f1:
            
            print(f"Saving to {model_path}")
            best_dev_f1 = dev_f1
            torch.save(state, model_path)
        print(f"Epoch {epoch} dev F1: {dev_f1*100:.2f}. Best dev F1: {best_dev_f1*100:.2f}.")    


checkpoint = torch.load(model_path)
print(f"Loading from {model_path}")
model.load_state_dict(checkpoint['model'], strict=True)    
test_output_file = os.path.join(output_dir, 'test_pred.json')

with torch.no_grad():
    model.eval()
    test_outputs = []
    test_labels = []
    for _, (input_ids, attn_mask, candidate_input_ids, candidate_attention_mask, labels) in enumerate(test_loader):
        outputs = model(input_ids, attention_mask=attn_mask, 
            candidate_input_ids = candidate_input_ids,
            candidate_attention_mask = candidate_attention_mask,
            )
        outputs = outputs.view(-1, args.n_classes)
        test_outputs.append(outputs) 
        test_labels.append(labels)
    test_outputs = torch.cat(test_outputs, dim=0) # n_sample, n_class
    test_labels = torch.cat(test_labels, dim=0) # n_sample,
    
    test_outputs = test_outputs.argmax(dim=1) # n_sample
    
    assert len(test_outputs) == len(test_labels) == len(test_examples), (len(test_outputs), len(test_labels), len(test_examples))
    test_f1 = calculate_fscore(test_outputs.detach().cpu().numpy(), test_labels.detach().cpu().numpy(), test_examples.itertuples())
    print(f"Epoch {epoch} test F1: {test_f1:.4f}. ")
    
    test_outputs = [float(o) for o in test_outputs]
    with open(test_output_file,'w') as f:
        json.dump({'output':test_outputs}, f)