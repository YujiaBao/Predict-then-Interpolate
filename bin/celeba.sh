# ERM
python src/main.py --method=erm --dataset=celeba --lr=0.00001 --weight_decay=0.001\
    --dropout=0.0

# IRM
python src/main.py --method=erm --dataset=celeba --lr=0.0001 --weight_decay=0.1\
    --dropout=0.0 --anneal_iters=10 --l_regret=10

# RGM
python src/main.py --method=rgm --dataset=celeba --lr=0.0001 --weight_decay=0.01\
    --dropout=0.0 --anneal_iters=1000 --l_regret=10

# DRO
python src/main.py --method=dro --dataset=celeba --lr=0.0001 --weight_decay=0.01\
    --dropout=0.0

# PI (Ours)
python src/main.py --method=ours --dataset=celeba --lr=0.00001 --weight_decay=1.0\
    --dropout=0.0
