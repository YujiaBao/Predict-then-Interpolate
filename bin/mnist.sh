# ERM
python src/main.py --method=erm --dataset=MNIST --lr=0.001 --weight_decay=0.001\
    --dropout=0.1

# IRM
python src/main.py --method=irm --dataset=MNIST --lr=0.001 --weight_decay=0.01\
    --dropout=0.5 --anneal_iters=1000 --l_regret=0.01

# RGM
python src/main.py --method=rgm --dataset=MNIST --lr=0.001 --weight_decay=0.01\
    --dropout=0.3 --anneal_iters=100 --l_regret=0.1

# DRO
python src/main.py --method=dro --dataset=MNIST --lr=0.001 --weight_decay=0.01\
    --dropout=0.5

# PI (Ours)
python src/main.py --method=ours --dataset=MNIST --lr=0.001 --weight_decay=0.01\
    --dropout=0.5

# Oracle
python src/main.py --method=oracle --dataset=MNIST --lr=0.001 --weight_decay=0.01\
    --dropout=0.5
