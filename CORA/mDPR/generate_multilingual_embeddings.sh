for l in en ar fa fr id pt ru;
do 
    
    python generate_dense_embeddings.py --model_file ../models/mDPR_biencoder_best.cpt --batch_size 64 --ctx_file ../../data/bbc_passages/$l/ --out_file ./embeddings_multilingual/emb_$l.pkl
    
done