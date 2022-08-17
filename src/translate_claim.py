from asyncore import read
from dataclasses import dataclass
from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm
from utils import read_data
import argparse

language2model_name = {
    'tr':'Helsinki-NLP/opus-mt-tr-en', 
    'ka':'Helsinki-NLP/opus-mt-ka-en', 
    'pt':'Helsinki-NLP/opus-mt-pt-uk',
    'id':'Helsinki-NLP/opus-mt-id-en', 
    # 'sr', 
    'it':'Helsinki-NLP/opus-mt-it-en', 
    'de':'Helsinki-NLP/opus-mt-de-en', 
    'ro':'Helsinki-NLP/opus-mt-ro-fr',
    # 'ta':,
    'pl':'Helsinki-NLP/opus-mt-pl-en', 
    'hi':'Helsinki-NLP/opus-mt-hi-en',
    'ar':'Helsinki-NLP/opus-mt-ar-en', 
    'es':'Helsinki-NLP/opus-mt-es-en',
    'uk': 'Helsinki-NLP/opus-mt-uk-en', 

    'ru': 'Helsinki-NLP/opus-mt-ru-en',
    'mr': 'Helsinki-NLP/opus-mt-mr-en',
    'sq': 'Helsinki-NLP/opus-mt-sq-en',
    # 'gu': 'Helsinki-NLP/opus-mt-gu-en',
    'fr': 'Helsinki-NLP/opus-mt-fr-en',
    'no': 'Helsinki-NLP/opus-mt-no-es',
    # 'si': 'Helsinki-NLP/opus-mt-si-en',
    'nl': 'Helsinki-NLP/opus-mt-nl-en',
    'az': 'Helsinki-NLP/opus-mt-az-en',
    'bn': 'Helsinki-NLP/opus-mt-bn-en',
    # 'fa': 'Helsinki-NLP/opus-mt-fa-en',
    'pa': 'Helsinki-NLP/opus-mt-pa-en'
}

language2tokenizer = {
    lang: MarianTokenizer.from_pretrained(model_name) for lang, model_name in language2model_name.items()
}
language2model = {
    lang: MarianMTModel.from_pretrained(model_name) for lang, model_name in language2model_name.items()
}


parser = argparse.ArgumentParser()
parser.add_argument('--input_file_name', required=True, type=str)
parser.add_argument('--output_file_name', required=True, type=str)
args = parser.parse_args()

def translate(claim, language):
    tokenizer = language2tokenizer[language]
    model = language2model[language]
    translated = model.generate(**tokenizer(claim, max_length=512, return_tensors="pt", padding="longest", truncation=True))[0]
    return tokenizer.decode(translated, skip_special_tokens=True) 


data = read_data(args.input_file_name)
translated_claims = []
for row in tqdm(data.itertuples()):

    if row.language not in language2model_name:
        translated_claims.append(None)
        continue
    
    current_src_language = row.language
    current_claim = row.claim
    current_translated = translate(current_claim, current_src_language)
    current_tgt_language = language2model_name[current_src_language][-2:]
    while current_tgt_language != 'en':
        current_src_language = current_tgt_language
        current_claim = current_translated
        current_translated = translate(current_claim, current_src_language)
        current_tgt_language = language2model_name[current_src_language][-2:]
    
    translated_claims.append(current_translated)

data['translated_claims'] = translated_claims

data.to_csv(args.output_file_name, index=None, sep='\t')