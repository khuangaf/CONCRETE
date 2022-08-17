for l in en ar fa fr id pt ru;
do 
    #
    python generate_dense_embeddings.py --model_file ../models/mDPR_biencoder_best.cpt --batch_size 64 --ctx_file ../../data/bbc_passages/$l/ --out_file ./embeddings_multilingual/emb_$l.pkl
    # python generate_dense_embeddings.py --model_file ../models/mDPR_biencoder_best.cpt --batch_size 64 --ctx_file ../../data/bbc_passages/$l/ --out_file ./embeddings_multilingual/emb_$l.pkl

    # python generate_dense_embeddings.py --model_file xict_outputs/xICT_biencoder.pt.38.56069 --batch_size 64 --ctx_file ../../data/bbc_passages/$l/ --out_file ./embeddings_multilingual/emb_$l.xict.pkl
    
    # python generate_dense_embeddings.py --model_file xict_outputs/xICT_biencoder.pt.0.9188 --batch_size 64 --ctx_file ../../data/bbc_passages/$l/ --out_file ./embeddings_multilingual/emb_$l.mbert.pkl

    # X-ICT
    # python generate_dense_embeddings.py --model_file xict_outputs/xICT_biencoder.pt.37.9188 --batch_size 64 --ctx_file ../../data/bbc_passages/$l/ --out_file ./embeddings_multilingual/emb_$l.xict.pkl
    
done