cd "$(dirname $0)"

python main.py \
--data_name=ogbl-collab \
--predictor=DOT \
--use_valedges_as_input=True \
--year=2010 \
--train_on_subgraph=True \
--epochs=800 \
--eval_last_best=True \
--dropout=0.3