# ERM
python src/main.py --method=erm --dataset=beer_0 --lr=0.001 --weight_decay=0.001\
    --dropout=0.1

# IRM
python src/main.py --method=erm --dataset=beer_0 --lr=0.001 --weight_decay=0.001\
    --dropout=0.3 --anneal_iters=10 --l_regret=0.1

# RGM
python src/main.py --method=rgm --dataset=beer_0 --lr=0.0001 --weight_decay=0.001\
    --dropout=0.3 --anneal_iters=10 --l_regret=0.1

# DRO
python src/main.py --method=dro --dataset=beer_0 --lr=0.001 --weight_decay=0.001\
    --dropout=0.3

# PI (Ours)
python src/main.py --method=ours --dataset=beer_0 --lr=0.0001 --weight_decay=0.001\
    --dropout=0.3

# Oracle
python src/main.py --method=oracle --dataset=beer_0 --lr=0.001 --weight_decay=0.001\
    --dropout=0.5
