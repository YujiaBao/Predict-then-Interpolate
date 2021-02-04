import torch
import torch.nn as nn

from models.embedding.cnn import CNN
from models.embedding.textcnn import TextCNN
from models.classifier.mlp import MLP
from models.embedding.resnet import Resnet50

def get_model(args, vocab=None):
    model = {}
    if args.dataset[:5] == 'MNIST':
        model['ebd'] = CNN(include_fc=True, hidden_dim=args.hidden_dim).cuda()
        out_dim=args.hidden_dim
        num_classes = 10

    if args.dataset == 'celeba':
        model['ebd'] = Resnet50().cuda()
        out_dim=model['ebd'].out_dim
        num_classes = 2

    if args.dataset[:4] == 'beer' or args.dataset == 'pubmed':
        model['ebd'] = TextCNN(vocab, num_filters=args.hidden_dim,
                               dropout=args.dropout).cuda()
        out_dim = args.hidden_dim * 3  # 3 different filters
        num_classes = 2

    if args.method in ['irm', 'erm', 'dro', 'ours', 'oracle']:
        model['clf_all'] = MLP(out_dim, args.hidden_dim, num_classes,
                               args.dropout, depth=1).cuda()

        opt = torch.optim.Adam(list(model['ebd'].parameters()) +
                               list(model['clf_all'].parameters()), lr=args.lr,
                               weight_decay=args.weight_decay)

        return model, opt

    if args.method in ['rgm']:
        # RGM learns environment-specific classifiers together with the overall
        # classifier
        model['clf_all'] = MLP(out_dim, args.hidden_dim, num_classes,
                               args.dropout, depth=1).cuda()

        model['clf_0'] = MLP(out_dim, args.hidden_dim, num_classes,
                             args.dropout, depth=2).cuda()

        model['clf_1'] = MLP(out_dim, args.hidden_dim, num_classes,
                             args.dropout, depth=2).cuda()

        opt_all = torch.optim.Adam(list(model['ebd'].parameters()) +
                               list(model['clf_all'].parameters()), lr=args.lr,
                               weight_decay=args.weight_decay)

        opt_0 = torch.optim.Adam(model['clf_0'].parameters(),
                                 lr=args.lr, weight_decay=args.weight_decay)

        opt_1 = torch.optim.Adam(model['clf_1'].parameters(),
                                 lr=args.lr, weight_decay=args.weight_decay)

        return model, [opt_all, opt_0, opt_1]

