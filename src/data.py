import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import json
import pandas as pd
from constants import LANG_CODE2COUNTRY, MONOLINGUAL_LANGUAGES
from utils import read_data

class PropaFakeDataset(Dataset):
    def __init__(self, tsv_path, retrieved_path, args, label2idx, idx2label, is_train_dev=False):
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        
        self.data = []
        # do not 
        all_retrieved_docs = json.load(open(retrieved_path,'r'))

        for row, retrieved_docs in zip(read_data(tsv_path).itertuples(), all_retrieved_docs):
            
            label = row.label
            label = label2idx[label]
            claim_language = row.language
            # print(row.claim, retrieved_docs['question'])
            # inputs = self.tokenizer(row.claim, max_length=args.max_sequence_length, padding="max_length", truncation=True)
            assert row.claim == retrieved_docs['question'], (row.claim, retrieved_docs['question'])
            if args.do_monolingual_eval:
                
                
                if claim_language not in MONOLINGUAL_LANGUAGES: continue
                
                # in monolingual retrieval setup, we can only train/test on data that are in the passage collection
                if args.do_monolingual_retrieval:
                    retrieved_docs['ctxs'] = [ctx for ctx in retrieved_docs['ctxs'] if ctx['id'][:2] == claim_language]
                
            if len(args.training_subset_langs) > 0 and is_train_dev:
                
                if claim_language not in args.training_subset_langs: continue

            top_k_candidates = [passage['text'].split('<sep>')[1] for passage in retrieved_docs['ctxs'][:args.n_candidates]]
            top_k_candidates += [''] * (args.n_candidates - len(top_k_candidates))
            assert len(top_k_candidates) == args.n_candidates, len(top_k_candidates)
            top_k_candidates_string = self.tokenizer.sep_token.join(top_k_candidates)
            # inputs = self.tokenizer(row.claim , max_length=args.max_sequence_length, padding="max_length", truncation=True)
            inputs_string = f"Claim made by {row.claimant} on {row.claimDate.split('T')[0]}, reported in {LANG_CODE2COUNTRY[row.language]}: {row.claim}"

            if args.disable_retrieval:
                
                inputs = self.tokenizer(inputs_string , max_length=args.max_sequence_length, padding="max_length", truncation=True)
                top_k_candidates = [''] * args.n_candidates
                candidates = self.tokenizer(top_k_candidates, max_length=args.max_sequence_length, padding="max_length", truncation=True)
            elif args.use_google_search:
                top_k_candidates = [row.evidence_1, row.evidence_2, row.evidence_3, row.evidence_4, row.evidence_5]
                top_k_candidates = [str(candidate) for candidate in top_k_candidates]
                inputs = self.tokenizer(inputs_string, max_length=args.max_sequence_length, padding="max_length", truncation=True)
                candidates = self.tokenizer(top_k_candidates, max_length=args.max_sequence_length, padding="max_length", truncation=True)
            else:
                # inputs_string = self.tokenizer.sep_token.join([row.language, row.site, row.claimant, row.claim, top_k_candidates_string])
                # inputs_string = f"Reported in {LANG_CODE2COUNTRY[row.language]}: {row.claim}"
                
                inputs_string = inputs_string #+ self.tokenizer.sep_token #+ top_k_candidates_string
                inputs = self.tokenizer(inputs_string, max_length=args.max_sequence_length, padding="max_length", truncation=True)
                candidates = self.tokenizer(top_k_candidates, max_length=args.max_sequence_length, padding="max_length", truncation=True)
            
            # candidates = self.tokenizer(top_k_candidates_string, max_length=args.max_sequence_length, padding="max_length", truncation=True)
            self.data.append({
                'input_ids':inputs['input_ids'],
                'attention_mask': inputs['attention_mask'],
                'candidate_input_ids': candidates['input_ids'],
                'candidate_attention_mask': candidates['attention_mask'],
                'label': label
            })
            
        

    def __len__(self):
        # 200K datapoints
        return len(self.data)

    def __getitem__(self, idx):
        
        return self.data[idx]['input_ids'], self.data[idx]['attention_mask'], self.data[idx]['candidate_input_ids'], self.data[idx]['candidate_attention_mask'], self.data[idx]['label']
    
    def collate_fn(self, batch):
        # print(batch)
        input_ids = torch.cuda.LongTensor([inst[0] for inst in batch])
        attention_masks = torch.cuda.LongTensor([inst[1]for inst in batch])
        candidate_input_ids = torch.cuda.LongTensor([inst[2] for inst in batch])
        candidate_attention_masks = torch.cuda.LongTensor([inst[3]for inst in batch])
        labels = torch.cuda.LongTensor([inst[4]for inst in batch])

        return input_ids, attention_masks, candidate_input_ids, candidate_attention_masks, labels