import argparse
from ast import parse
from base64 import encode
from email.policy import default
import json
import os
import random
from glob import glob
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from spacy.lang.ru import Russian
from spacy.lang.fr import French
from spacy.lang.fa import Persian
from spacy.lang.pt import Portuguese
from spacy.lang.id import Indonesian
from spacy.lang.en import English
from spacy.lang.ar import Arabic
from spacy.tokenizer import Tokenizer




m2o_model_name = "facebook/mbart-large-50-many-to-one-mmt"
o2m_model_name = "facebook/mbart-large-50-one-to-many-mmt"

m2o_model = MBartForConditionalGeneration.from_pretrained(m2o_model_name)
m2o_tokenizer = MBart50TokenizerFast.from_pretrained(m2o_model_name)

o2m_model = MBartForConditionalGeneration.from_pretrained(o2m_model_name)
o2m_tokenizer = MBart50TokenizerFast.from_pretrained(o2m_model_name)

# print(tokenizer.lang_code_to_id)
o2m_model.cuda()
m2o_model.cuda()

target_languages = ['ru','fr','fa','pt','id','ar','en']

def get_language_code(lang, tokenizer):
    # language_code_to_id = {'ar_AR': 250001, 'cs_CZ': 250002, 'de_DE': 250003, 'en_XX': 250004, 'es_XX': 250005, 'et_EE': 250006, 'fi_FI': 250007, 'fr_XX': 250008, 'gu_IN': 250009, 'hi_IN': 250010, 'it_IT': 250011, 'ja_XX': 250012, 'kk_KZ': 250013, 'ko_KR': 250014, 'lt_LT': 250015, 'lv_LV': 250016, 'my_MM': 250017, 'ne_NP': 250018, 'nl_XX': 250019, 'ro_RO': 250020, 'ru_RU': 250021, 'si_LK': 250022, 'tr_TR': 250023, 'vi_VN': 250024, 'zh_CN': 250025, 'af_ZA': 250026, 'az_AZ': 250027, 'bn_IN': 250028, 'fa_IR': 250029, 'he_IL': 250030, 'hr_HR': 250031, 'id_ID': 250032, 'ka_GE': 250033, 'km_KH': 250034, 'mk_MK': 250035, 'ml_IN': 250036, 'mn_MN': 250037, 'mr_IN': 250038, 'pl_PL': 250039, 'ps_AF': 250040, 'pt_XX': 250041, 'sv_SE': 250042, 'sw_KE': 250043, 'ta_IN': 250044, 'te_IN': 250045, 'th_TH': 250046, 'tl_XX': 250047, 'uk_UA': 250048, 'ur_PK': 250049, 'xh_ZA': 250050, 'gl_ES': 250051, 'sl_SI': 250052}    
    language_codes = list(tokenizer.lang_code_to_id.keys())
    lang2language_code = {str(code).split('_')[0]: code for code in language_codes}
    
        

    return lang2language_code[lang]

def get_nlp(language):
    if language == 'en': return English()
    elif language == 'ru': return Russian()
    elif language == 'fr': return French()
    elif language == 'fa': return Persian()
    elif language == 'pt': return Portuguese()
    elif language == 'id': return Indonesian()
    elif language == 'ar': return Arabic()
    # print(language)
    raise NotImplementedError

def _format_mbart_lang(lang, tokenizer):
    # if is_target:
    #     formatted_lang = f'{lang}_XX'
    # else:
    formatted_lang = get_language_code(lang, tokenizer)

    return formatted_lang

def translate(sentence, src_lang, tgt_lang):
    '''
    src_lang, tgt_lang: one of ['ru','fr','fa','pt','id','ar','en']
    '''
    # first, do many to one translation
    m2o_tokenizer.src_lang = _format_mbart_lang(src_lang, m2o_tokenizer)
    tgt_lang = _format_mbart_lang(tgt_lang, o2m_tokenizer)
    encoded_sent = m2o_tokenizer(sentence, return_tensors="pt")

    for k, v in encoded_sent.items():
        encoded_sent[k] = v.cuda()

    generated_tokens = m2o_model.generate(
        **encoded_sent,
        # forced_bos_token_id=m2o_tokenizer.lang_code_to_id[tgt_lang],
        do_sample=False,
        num_beams=4
    )
    m2o_translated_sent = m2o_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

    # do need to do o2m if the tgt lang is english.
    if tgt_lang == _format_mbart_lang('en', o2m_tokenizer):
        translated_sent = m2o_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    else:
        # then, do one to many translation
        encoded_sent = o2m_tokenizer(m2o_translated_sent, return_tensors="pt")

        for k, v in encoded_sent.items():
            encoded_sent[k] = v.cuda()

        generated_tokens = o2m_model.generate(
            **encoded_sent,
            forced_bos_token_id=o2m_tokenizer.lang_code_to_id[tgt_lang],
            do_sample=False,
            num_beams=4
        )
        translated_sent = o2m_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    # print(src_lang, tgt_lang)
    # print(sentence, translated_sent)
    
    return translated_sent

def create_instance(title, relevant_passages, src_language, src_passage_idx, all_passages, num_negative_samples=5):
    instance = {}

    # nlp = get_nlp(src_language)
    # nlp.add_pipe("sentencizer")
    # doc = nlp(src_passage)
    # if len([sent for sent in  doc.sents]) < 3: return None
    src_sentence = title
    # do translation 1/7 of the time
    tgt_language = random.choice(target_languages)
    # always do translation
    # tgt_language = random.choice(list(set(target_languages) - set([src_language])))
    
    # assert tgt_language != src_language
    
    if tgt_language != src_language:
        tgt_sentence = translate(src_sentence, src_lang=src_language, tgt_lang=tgt_language)
    else:
        
        tgt_sentence = src_sentence
        

    instance['claim'] = tgt_sentence
    instance['src_language'] = src_language
    instance['tgt_language'] = tgt_language
    instance['positive_ctxs'] = relevant_passages
    # if is_train, then use in-batch negatives. Otherwise, randomly sample one from all_passages
    
    negative_passages = all_passages[:src_passage_idx] + all_passages[src_passage_idx+1:]
    # print(len(negative_passages))
    # sample negative
    instance['negative_ctxs'] = np.random.choice(negative_passages, replace=False, size=num_negative_samples).tolist()

    
    return instance

def _is_other_passage(current_passage, other_passage):
    current_passage_id = current_passage['id']
    other_passage_id = other_passage['id']
    same_doc_id = current_passage_id.split('-')[0] == other_passage_id.split('-')[0]
    different_passage_idx = current_passage_id.split('-')[1] != other_passage_id.split('-')[1]
    return same_doc_id  and different_passage_idx

def main(args):
    all_passages = []
    doc_id2passages = {} #defaultdict(list)
    for lang in target_languages:
        for passage_path in  glob(os.path.join(args.passage_dir,f'{lang}/*.txt')):
            # passage_id looks like: fffb0c17b17c30ec1de7be57c45c50db2d51c14c05736cbda7aac84b252e4b92-0
            passage_id = passage_path.split('/')[-1].split('.txt')[0]#.split('-')
            doc_id = passage_id.split('-')[0]
            title, passage = open(passage_path,'r').read().split('<sep>')
            # do not include passages that does not have body text
            if len(passage.strip()) == 0: continue
            current_psg = {
                'lang':lang,
                'passage': passage,
                'id': passage_id,
                'doc_id':doc_id,
                'title': title
                }
            all_passages.append(current_psg)
            if doc_id not in doc_id2passages:
                doc_id2passages[doc_id] = []

            doc_id2passages[doc_id].append(current_psg)
            # if len(all_passages) > 20:
            #     break
    
    all_doc_ids = list(doc_id2passages.keys())
    shard_size = int(len(all_doc_ids) / args.num_shards)
    start_idx = args.shard_id * shard_size
    end_idx = start_idx + shard_size
    

    doc_ids_to_be_process = all_doc_ids[start_idx:end_idx]
    
    if args.is_eval:
        doc_ids_to_be_process = doc_ids_to_be_process[:200]

    instances = []


    for idx, doc_id in enumerate(tqdm(doc_ids_to_be_process)):
        
        relevant_passages = doc_id2passages[doc_id]
        # exclude samples with no passage
        if len(relevant_passages) == 0: continue
        title = relevant_passages[0]['title']
        language = relevant_passages[0]['lang']
        if len(relevant_passages) == 0: 
            continue
        instance = create_instance(title=title, relevant_passages=relevant_passages, src_language=language, src_passage_idx=idx, all_passages=all_passages)
        if instance is not None:
            instances.append(instance)

    with open(args.out_file+f'_{args.shard_id}','w') as f:
        for inst in instances:
            f.write(json.dumps(inst)+'\n')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--passage_dir', type=str, default=None, help='Dir of passages.')
    
    parser.add_argument('--out_file', required=True, type=str, default=None,
                        help='output .tsv file path to write results to ')    
    parser.add_argument('--num_shards', type=int, default=4, help='Dir of passages.')
    parser.add_argument('--shard_id', type=int, default=0, help='The index of this shard.')
    parser.add_argument("--is_eval", action='store_true', help="if is eval, do retrieval for 500 docs only")
    args = parser.parse_args()

    
    main(args)