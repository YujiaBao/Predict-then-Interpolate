# ERM
python src/main.py --method=erm --dataset=beer_2 --lr=0.0001 --weight_decay=0.001\
    --dropout=0.5

# IRM
python src/main.py --method=erm --dataset=beer_2 --lr=0.0001 --weight_decay=0.001\
    --dropout=0.3 --anneal_iters=100 --l_regret=0.01

# RGM
python src/main.py --method=rgm --dataset=beer_2 --lr=0.0001 --weight_decay=0.001\
    --dropout=0.3 --anneal_iters=10 --l_regret=0.1

# DRO
python src/main.py --method=dro --dataset=beer_2 --lr=0.001 --weight_decay=0.001\
    --dropout=0.1

# PI (Ours)
python src/main.py --method=ours --dataset=beer_2 --lr=0.0001 --weight_decay=0.001\
    --dropout=0.5

# Oracle
python src/main.py --method=oracle --dataset=beer_2 --lr=0.0001 --weight_decay=0.001\
    --dropout=0.5
