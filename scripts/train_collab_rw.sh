cd "$(dirname $0)"

python ../main.py \
--data_name=ogbl-collab  \
--encoder TRANSFORMER \
--predictor=DOT \
--use_valedges_as_input=True \
--year=2010 \
--train_on_subgraph=True \
--epochs=800 \
--eval_last_best=True \
--dropout=0.3 \
--gnn_num_layers=2 \
--grad_clip_norm=1 \
--use_lr_decay=True \
--random_walk_augment=True \
--walk_length=10 \
--loss_func=WeightedHingeAUC \
--num_heads 4 \
--device 1 \
--emb_hidden_channels 128 \
--gnn_hidden_channels 128 \
--use_node_feats=True \
--pretrain_emb=True \
--seed 0 

# --encoder TRANSFORMER \
