python ./MoE_RQ-VAE/main.py \
  --device cuda:1 \
  --data_path /home/user/yuhou/Item_tokenization/data_process/concat10.emb-llama-td.npy\
  --alpha 0.01 \
  --beta 0.0001 \
  --cf_emb ./RQ-VAE/ckpt/Instruments-32d-sasrec.pt\
  --ckpt_dir ../checkpoint/