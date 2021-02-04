import argparse
import datetime
import random
import json
import os

import torch
import numpy as np
from tensorboardX import SummaryWriter

from data_utils import get_dataset
from model_utils import get_model
from train_utils import train_val_test



def get_parser():
    parser = argparse.ArgumentParser(description=
                                     'Predict, then Interpolate: A Simple Algorithm to Learn Stable Classifiers')
    parser.add_argument('--cuda', type=int, default=0)

    # data sample
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--num_batches', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--num_query', type=int, default=20)
    parser.add_argument('--seed', type=int, default=0)

    # model
    parser.add_argument('--hidden_dim', type=int, default=390)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--dropout', type=float, default=0.0)

    #dataset
    parser.add_argument('--dataset', type=str, default='MNIST')
    parser.add_argument('--val', type=str, default='in_domain')

    parser.add_argument('--method', type=str, default='irm')

    #optimization
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--l_regret', type=float, default=1)
    parser.add_argument('--anneal_iters', type=float, default=1)
    parser.add_argument('--clip_grad', type=float, default=1)
    parser.add_argument('--patience', type=int, default=20)

    #result file
    parser.add_argument('--results_path', type=str, default='')

    return parser


def print_args(args):
    '''
        Print arguments (only show the relevant arguments)
    '''
    print("Parameters:")
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}={}".format(attr.upper(), value))
    print('''
    (Credit: Maija Haavisto)                        /
                                 _,.------....___,.' ',.-.
                              ,-'          _,.--"        |
                            ,'         _.-'              .
                           /   ,     ,'                   `
                          .   /     /                     ``.
                          |  |     .                       \.\\
                ____      |___._.  |       __               \ `.
              .'    `---""       ``"-.--"'`  \               .  \\
             .  ,            __               `              |   .
             `,'         ,-"'  .               \             |    L
            ,'          '    _.'                -._          /    |
           ,`-.    ,".   `--'                      >.      ,'     |
          . .'\\'   `-'       __    ,  ,-.         /  `.__.-      ,'
          ||:, .           ,'  ;  /  / \ `        `.    .      .'/
          j|:D  \          `--'  ' ,'_  . .         `.__, \   , /
         / L:_  |                 .  "' :_;                `.'.'
         .    ""'                  """""'                    V
          `.                                 .    `.   _,..  `
            `,_   .    .                _,-'/    .. `,'   __  `
             ) \`._        ___....----"'  ,'   .'  \ |   '  \  .
            /   `. "`-.--"'         _,' ,'     `---' |    `./  |
           .   _  `""'--.._____..--"   ,             '         |
           | ." `. `-.                /-.           /          ,
           | `._.'    `,_            ;  /         ,'          .
          .'          /| `-.        . ,'         ,           ,
          '-.__ __ _,','    '`-..___;-...__   ,.'\ ____.___.'
          `"^--'..'   '-`-^-'"--    `-^-'`.''"""""`.,^.`.--' mh
    ''')


def set_seed(seed):
    '''
        Setting random seeds
    '''
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    torch.cuda.set_device(args.cuda)

    print_args(args)

    set_seed(args.seed)

    # get dataset:
    train_data, test_data = get_dataset(args.dataset, args.val)

    # get data loaders for training and testing:
    # env 0 and 1 are used for training
    # env 2 is used for validation
    # env 3 is used for testing

    # initialize model and optimizer based on the dataset and the method
    if args.dataset[:4] == 'beer' or args.dataset == 'pubmed':
        model, opt = get_model(args, train_data.vocab)
    else:
        model, opt = get_model(args)

    # start training
    print("{}, Start training {} on train env for {}".format(
        datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S'),
        args.method, args.dataset), flush=True)

    # get tensorboard logger
    logdir = './log/{}.'.format(args.dataset)
    filename = 'lr_{}_method_{}_regret_{}_dropout_{}_wd_{}'.format(
        args.lr,
        args.method,
        args.l_regret,
        args.dropout,
        args.weight_decay)
    filename += '_seed_{}'.format(args.seed)
    writer = SummaryWriter(logdir=logdir + filename)

    train_res, val_res, test_res = train_val_test(train_data, test_data,
                                                  model, opt, writer, args)

    res = {'train_res': train_res, 'val_res': val_res, 'test_res': test_res}
    for attr, value in sorted(args.__dict__.items()):
        res[attr] = value

    if args.results_path != '':
        with open(args.results_path, 'w') as f:
            json.dump(res, f)

    writer.close()
