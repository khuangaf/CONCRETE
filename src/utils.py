import pandas as pd
from constants import MONOLINGUAL_LANGUAGES


def read_data(file_name, args=None, is_train_dev=False):
    data = [l.strip().split('\t') for l in open(file_name,'r').readlines()]
    assert len( set([len(row) for row in data]))  == 1
    df = pd.DataFrame(data[1:], columns= data[0])

    if args is not None and (args.do_monolingual_eval or args.do_monolingual_retrieval):
        df = df.loc[df.language.isin(MONOLINGUAL_LANGUAGES)]
    
    if args is not None and len(args.training_subset_langs) > 0 and is_train_dev:
        df = df.loc[df.language.isin(args.training_subset_langs)]
    
    return df