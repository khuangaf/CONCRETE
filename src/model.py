import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


# define model
class BERTModelForClassification(nn.Module):

    def __init__(self, model_name, args):
        super().__init__()
        self.n_candidates = args.n_candidates
        self.bert = AutoModel.from_pretrained(model_name)
        # transformer_layer = nn.TransformerEncoderLayer(d_model=self.bert.config.hidden_size, nhead=8)
        # encoder_norm = nn.LayerNorm(self.bert.config.hidden_size)
        # self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=1, norm=encoder_norm)
        self.disable_retrieval = args.disable_retrieval
        if self.disable_retrieval:
            self.linear = nn.Linear(self.bert.config.hidden_size , args.n_classes)
        else:
            self.linear = nn.Linear(self.bert.config.hidden_size * (args.n_candidates + 1), args.n_classes)
        self.n_classes = args.n_classes
        # TODO: maybe need to warm start W_candidate and W_query
        # self.W_candidate = nn.Linear(self.bert.config.hidden_size, 256, bias=False)
        # self.W_query = nn.Linear(self.bert.config.hidden_size, 256, bias=False)

    def forward(self, 
                input_ids: torch.tensor, 
                attention_mask: torch.tensor, 
                candidate_input_ids: torch.tensor,
                candidate_attention_mask: torch.tensor,
                # indexer,
                epoch: int = None):

        batch_size = input_ids.size(0)
        seq_len = input_ids.size(1)
        hidden_states = self.bert(input_ids=input_ids,
                    attention_mask=attention_mask)[0] # batch, seq_len, emb_dim
        query_cls_embeddings = hidden_states[:,0,:]
        if self.disable_retrieval:
            final_repr = query_cls_embeddings
        else:
            candidate_input_ids = candidate_input_ids.view(-1, seq_len)
            candidate_attention_mask = candidate_attention_mask.view(-1, seq_len)
            
            retrieved_hidden_states = self.bert(input_ids=candidate_input_ids,
                        attention_mask=candidate_attention_mask)[0] # batch * num_retrieved, seq_len, emb_dim
            
            retrieved_cls_embeddings = retrieved_hidden_states[:,0,:]
            
            retrieved_cls_embeddings = retrieved_cls_embeddings.reshape(batch_size, -1)
            final_repr = torch.cat([query_cls_embeddings, retrieved_cls_embeddings], dim=1)
        
        logits = self.linear(final_repr)
        
        

        return logits#, candidate_outputs

    def retrieve(self, indexer, query_embeddings, candidate_input_ids, candidate_attention_mask):#query_input_ids, query_attention_mask):
        '''
        Return doc_input_ids, doc_attention_mask of the most relevant article
        '''
        # Maybe just pass the CLS embedding of the query input text as input parameters.
        batch_size = query_embeddings.size(0)
        # query_embeddings = self.W_query(query_embeddings) # (batch, emb_dim)
        # candidate_embeddings = self.W_candidate(candidate_embeddings) # (n_candidates, emb_dim)
        
        
        # retrival_scores = torch.matmul(query_embeddings, candidate_embeddings.T) # (batch, n_candidates)
        # retrival_log_probs = F.log_softmax(retrival_scores, dim=1)
        # retrieved_candidate_log_probs, retrieved_indices = retrival_log_probs.max(dim=1)
        
        retrieved_indices = indexer.search(query_embeddings.detach().cpu().numpy(), self.n_candidates).reshape(-1) # (batch * k, )
        # Do not move these to CUDA until now to save memory.
        retrieved_candidate_input_ids = candidate_input_ids[retrieved_indices].to(query_embeddings.device) # (batch * k, seq)
        retrieved_candidate_attention_mask = candidate_attention_mask[retrieved_indices].to(query_embeddings.device) # (batch * k, seq)
        
        # recompute candidate log probs using input ids and attention mask
        # retrieved_candidate_input_ids = retrieved_candidate_input_ids.view(batch_size, self.n_candidates, -1)
        # retrieved_candidate_attention_mask = retrieved_candidate_attention_mask.view(batch_size, self.n_candidates, -1)

        return retrieved_candidate_input_ids, retrieved_candidate_attention_mask


