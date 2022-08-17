mkdir retrieved_docs/
for f in train.all dev.all test.all ood zeroshot;
do
    # X-ICT
    python dense_retriever.py --model_file xict_outputs/xICT_biencoder.pt.37.9188 --ctx_file ../../data/bbc_passages/ --claim_file ../../data/x-fact/$f.tsv --encoded_dir ./embeddings_multilingual/ --out_file retrieved_docs/$f.xict.json --batch_size 64 --n-docs 100
    
done