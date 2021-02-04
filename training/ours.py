import copy
import datetime
from termcolor import colored
from tqdm import tqdm
import torch
import numpy as np
import training.utils as utils
import torch.nn.functional as F
from data_utils import EnvSampler
from torch.utils.data import DataLoader
from model_utils import get_model


def train_dro_loop(train_loaders, model, opt, ep, args):
    stats = {}
    for k in ['worst_loss', 'avg_loss', 'worst_acc', 'avg_acc']:
        stats[k] = []

    step = 0
    for batches in zip(*train_loaders):
        # work on each batch
        model['ebd'].train()
        model['clf_all'].train()

        x, y = [], []

        for batch in batches:
            batch = utils.to_cuda(utils.squeeze_batch(batch))
            x.append(batch['X'])
            y.append(batch['Y'])

        if 'beer' in args.dataset or 'pubmed' in args.dataset:
            # text models have varying length between batches
            pred = []
            for cur_x in x:
                pred.append(model['clf_all'](model['ebd'](cur_x)))
            pred = torch.cat(pred, dim=0)
        else:
            pred = model['clf_all'](model['ebd'](torch.cat(x, dim=0)))

        cur_idx = 0

        avg_loss = 0
        avg_acc = 0
        worst_loss = 0
        worst_acc = 0

        for cur_true in y:
            cur_pred = pred[cur_idx:cur_idx+len(cur_true)]
            cur_idx += len(cur_true)

            loss = F.cross_entropy(cur_pred, cur_true)
            acc = torch.mean((torch.argmax(cur_pred, dim=1) == cur_true).float()).item()

            avg_loss += loss.item()
            avg_acc += acc

            if loss.item() > worst_loss:
                worst_loss = loss
                worst_acc = acc

        opt.zero_grad()
        worst_loss.backward()
        opt.step()

        avg_loss /= len(y)
        avg_acc /= len(y)

        stats['avg_acc'].append(avg_acc)
        stats['avg_loss'].append(avg_loss)
        stats['worst_acc'].append(worst_acc)
        stats['worst_loss'].append(worst_loss.item())

    for k, v in stats.items():
        stats[k] = float(np.mean(np.array(v)))

    return stats


def train_loop(train_loader, model, opt, ep, args):
    stats = {}
    for k in ['acc', 'loss']:
        stats[k] = []

    step = 0
    for batch in train_loader:
        # work on each batch
        model['ebd'].train()
        model['clf_all'].train()

        batch = utils.to_cuda(utils.squeeze_batch(batch))

        x = model['ebd'](batch['X'])
        y = batch['Y']

        acc, loss = model['clf_all'](x, y, return_pred=False,
                                     grad_penalty=False)

        opt.zero_grad()
        loss.backward()
        opt.step()

        stats['acc'].append(acc)
        stats['loss'].append(loss.item())

    for k, v in stats.items():
        stats[k] = float(np.mean(np.array(v)))

    return stats


def test_loop(test_loader, model, ep, args, return_idx=False, att_idx_dict=None):
    loss_list = []
    true, pred, cor = [], [], []
    if (att_idx_dict is not None) or return_idx:
        idx = []

    for batch in test_loader:
        # work on each batch
        model['ebd'].eval()
        model['clf_all'].eval()

        batch = utils.to_cuda(utils.squeeze_batch(batch))

        x = model['ebd'](batch['X'])
        y = batch['Y']
        c = batch['C']

        y_hat, loss = model['clf_all'](x, y, return_pred=True)

        true.append(y)
        pred.append(y_hat)
        cor.append(c)

        if (att_idx_dict is not None) or return_idx:
            idx.append(batch['idx'])

        loss_list.append(loss.item())

    true = torch.cat(true)
    pred = torch.cat(pred)

    acc = torch.mean((true == pred).float()).item()
    loss = np.mean(np.array(loss_list))

    if return_idx:
        cor = torch.cat(cor).tolist()
        true = true.tolist()
        pred = pred.tolist()
        idx = torch.cat(idx).tolist()

        # split correct and wrong idx
        correct_idx, wrong_idx = [], []

        # compute correlation between cor and y for analysis
        correct_cor, wrong_cor = [], []
        correct_y, wrong_y = [], []

        for i, y, y_hat, c in zip(idx, true, pred, cor):
            if y == y_hat:
                correct_idx.append(i)
                correct_cor.append(c)
                correct_y.append(y)
            else:
                wrong_idx.append(i)
                wrong_cor.append(c)
                wrong_y.append(y)

        return {
            'acc': acc,
            'loss': loss,
            'correct_idx': correct_idx,
            'correct_cor': correct_cor,
            'correct_y': correct_y,
            'wrong_idx': wrong_idx,
            'wrong_cor': wrong_cor,
            'wrong_y': wrong_y,
        }

    if att_idx_dict is not None:
        return utils.get_worst_acc(true, pred, idx, loss, att_idx_dict)

    return {
        'acc': acc,
        'loss': loss,
    }


def print_res(train_res, val_res, ep):
    print(("epoch {epoch}, train {acc} {train_acc:>7.4f} {train_worst_acc:>7.4f} "
           "{loss} {train_loss:>10.7f} {train_worst_loss:>10.7f} "
           "val {acc} {val_acc:>10.7f}, {loss} {val_loss:>10.7f}").format(
               epoch=ep,
               acc=colored("acc", "blue"),
               loss=colored("loss", "yellow"),
               regret=colored("regret", "red"),
               train_acc=train_res["avg_acc"],
               train_worst_acc=train_res["worst_acc"],
               train_loss=train_res["avg_loss"],
               train_worst_loss=train_res["worst_loss"],
               val_acc=val_res["acc"],
               val_loss=val_res["loss"]), flush=True)


def print_group_stats(pretrain_res, train_data=None):
    if train_data is None:
        for i in range(len(pretrain_res)):
            print('{}on{}_correct '.format(1-i, i), end='')
            print('len: {:7d}, correlation: {:>7.4f}'.format(
                len(pretrain_res[i]['correct_idx']),
                np.corrcoef(pretrain_res[i]['correct_cor'],
                            pretrain_res[i]['correct_y'])[0,1]))

            print('{}on{}_wrong   '.format(1-i, i), end='')
            print('len: {:7d}, correlation: {:>7.4f}'.format(
                len(pretrain_res[i]['wrong_idx']),
                np.corrcoef(pretrain_res[i]['wrong_cor'],
                            pretrain_res[i]['wrong_y'])[0,1]))
    else:
        # print per attribute correlation for each group
        results = []
        for i in range(len(pretrain_res)):
            # for env i
            attr_matrix = train_data.get_all_att(i)
            y_correct = attr_matrix[pretrain_res[i]['correct_idx'],
                                    train_data.label_idx]
            y_wrong = attr_matrix[pretrain_res[i]['wrong_idx'],
                                  train_data.label_idx]
            y_all = attr_matrix[train_data.envs[i]['idx_list'],
                                train_data.label_idx]

            # print(len(train_data.envs[i]['idx_list']))
            # print(len(pretrain_res[i]['correct_idx']))
            # print(len(pretrain_res[i]['wrong_idx']))

            for att_idx, attr in enumerate(torch.transpose(attr_matrix, 0, 1)):
                c_correct = attr[pretrain_res[i]['correct_idx']]
                c_wrong = attr[pretrain_res[i]['wrong_idx']]
                c_all = attr[train_data.envs[i]['idx_list']]
                rho_all = np.corrcoef(y_all.tolist(), c_all.tolist())[0, 1]
                rho_correct = np.corrcoef(y_correct.tolist(), c_correct.tolist())[0, 1]
                rho_wrong = np.corrcoef(y_wrong.tolist(), c_wrong.tolist())[0, 1]
                if i == 0:
                    results.append([rho_all, rho_correct, rho_wrong])
                else:
                    results[att_idx].append(rho_all)
                    results[att_idx].append(rho_correct)
                    results[att_idx].append(rho_wrong)

        print('0_all, 1_0_correct, 1_0_wrong, 1_all,  0_1_correct, 0_1_wrong')
        for i, rhos in enumerate(results):
            print(train_data.get_att_names(i), end=', ')
            for r in rhos:
                print('{:>7.4f}'.format(r), end=', ')
            print()


def print_pretrain_res(train_res, test_res, ep, i):
    print(("petrain {i}, epoch {epoch}, train {acc} {train_acc:>7.4f} "
           "{loss} {train_loss:>7.4f}, "
           "val {acc} {test_acc:>7.4f}, {loss} {test_loss:>7.4f} ").format(
               epoch=ep,
               i = i,
               acc=colored("acc", "blue"),
               loss=colored("loss", "yellow"),
               ebd=colored("ebd", "red"),
               train_acc=train_res["acc"],
               train_loss=train_res["loss"],
               test_acc=test_res["acc"],
               test_loss=test_res["loss"]), flush=True)


def ours(train_data, test_data, model, opt, writer, args):
    train_loaders = []
    for i in range(2):
        train_loaders.append(DataLoader(
            train_data,
            sampler=EnvSampler(args.num_batches, args.batch_size, i,
                               train_data.envs[i]['idx_list']),
        num_workers=10))

    test_loaders = []
    for i in range(4):
        test_loaders.append(DataLoader(
            test_data,
            sampler=EnvSampler(-1, args.batch_size, i,
                               test_data.envs[i]['idx_list']),
        num_workers=10))

    # training the environment-specific classifier
    models = []
    for i in range(2):
        if hasattr(train_data, 'vocab'):
            cur_model, cur_opt = get_model(args, train_data.vocab)
        else:
            cur_model, cur_opt = get_model(args)

        print("{}, Start training classifier on train env {}".format(
            datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S'), i),
              flush=True)

        best_acc = -1
        best_model = {}
        cycle = 0

        # start training the env specific model
        for ep in range(args.num_epochs):
            train_res = train_loop(train_loaders[i], cur_model, cur_opt, ep, args)

            with torch.no_grad():
                # evaluate on the other training environment
                val_res = test_loop(train_loaders[1-i], cur_model, ep, args)

            print_pretrain_res(train_res, val_res, ep, i)

            writer.add_scalar('pretrain_{}/train_env'.format(i), train_res['acc'], ep)
            writer.add_scalar('pretrain_{}/test_env'.format(i), val_res['acc'], ep)
            writer.add_scalar('pretrain_{}/train_env_ce'.format(i), train_res['loss'], ep)

            # if min(train_res['acc'], test_res['acc']) > best_acc:
            if val_res['acc'] > best_acc:
                best_acc = val_res['acc']
                cycle = 0
                # save best ebd
                for k in 'ebd', 'clf_all':
                    best_model[k] = copy.deepcopy(cur_model[k].state_dict())
            else:
                cycle += 1

            if cycle == args.patience:
                break

        # load best model
        for k in 'ebd', 'clf_all':
            cur_model[k].load_state_dict(best_model[k])

        models.append(cur_model)

    # load training data in test mode
    test_train_loaders = []
    for i in range(2):
        test_train_loaders.append(DataLoader(train_data,
            sampler=EnvSampler(-1, args.batch_size, i,
                               train_data.envs[i]['idx_list']), num_workers=10))

    # split the dataset based on the model predictions
    pretrain_res = []
    pretrain_res.append(test_loop(test_train_loaders[0], models[1], ep, args, True))
    pretrain_res.append(test_loop(test_train_loaders[1], models[0], ep, args, True))

    if args.dataset in ['pubmed', 'celeba']:
        print_group_stats(pretrain_res, train_data)
    else:
        print_group_stats(pretrain_res)

    # train a new unbiased model through dro
    train_loaders = []
    print('\n######################\nCreate New Groups')
    for i in range(len(pretrain_res)):
        train_loaders.append(DataLoader(
            train_data, sampler=EnvSampler(args.num_batches, args.batch_size, i,
                                           pretrain_res[i]['correct_idx']),
            num_workers=10))

        train_loaders.append(DataLoader(
            train_data, sampler=EnvSampler(args.num_batches, args.batch_size, i,
                                           pretrain_res[i]['wrong_idx']),
        num_workers=10))

    # start training
    best_acc = -1
    best_val_res = None
    best_model = {}
    cycle = 0
    for ep in range(args.num_epochs):
        train_res = train_dro_loop(train_loaders, model, opt, ep, args)

        with torch.no_grad():
            # validation
            val_res = test_loop(test_loaders[2], model, ep, args,
                                att_idx_dict=test_data.val_att_idx_dict)

        print_res(train_res, val_res, ep)

        writer.add_scalar('acc/avg', train_res['avg_acc'], ep)
        writer.add_scalar('acc/worst', train_res['worst_acc'], ep)
        writer.add_scalar('acc/val_env', val_res['acc'], ep)
        writer.add_scalar('loss/avg', train_res['avg_loss'], ep)
        writer.add_scalar('loss/worst', train_res['worst_loss'], ep)

        if min(train_res['worst_acc'], val_res['acc']) > best_acc:
            best_acc = min(train_res['worst_acc'], val_res['acc'])
            best_val_res = val_res
            best_train_res = train_res
            cycle = 0
            # save best ebd
            for k in 'ebd', 'clf_all':
                best_model[k] = copy.deepcopy(model[k].state_dict())
        else:
            cycle += 1

        if cycle == args.patience:
            break

    # load best model
    for k in 'ebd', 'clf_all':
        model[k].load_state_dict(best_model[k])

    # get the results
    test_res = test_loop(test_loaders[3], model, ep, args,
                         att_idx_dict=test_data.test_att_idx_dict)
    print('Best train')
    print(train_res)
    print('Best val')
    print(val_res)
    print('Test')
    print(test_res)

    return train_res, val_res, test_res
