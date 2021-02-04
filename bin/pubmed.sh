# ERM
python src/main.py --method=erm --dataset=pubmed --lr=0.001 --weight_decay=0.001\
    --dropout=0.3

# IRM
python src/main.py --method=irm --dataset=pubmed --lr=0.001 --weight_decay=0.001\
    --dropout=0.3 --anneal_iters=1000 --l_regret=0.01

# RGM
python src/main.py --method=rgm --dataset=pubmed --lr=0.001 --weight_decay=0.001\
    --dropout=0.1 --anneal_iters=10 --l_regret=0.1

# DRO
python src/main.py --method=dro --dataset=pubmed --lr=0.00001 --weight_decay=0.001\
    --dropout=0.3

# PI (Ours)
python src/main.py --method=ours --dataset=pubmed --lr=0.0001 --weight_decay=0.001\
    --dropout=0.5
