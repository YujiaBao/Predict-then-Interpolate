import copy
import datetime
from termcolor import colored
from tqdm import tqdm
import torch
import numpy as np
import training.utils as utils
from data_utils import EnvSampler
from torch.utils.data import DataLoader


def train_loop(train_loaders, model, opt, ep, args):
    stats = {}
    for k in ['acc', 'loss', 'regret', 'loss_train']:
        stats[k] = []

    step = ep * args.num_batches
    for batch_0, batch_1 in zip(train_loaders[0], train_loaders[1]):
        # work on each batch
        model['ebd'].train()
        model['clf_all'].train()

        batch_0 = utils.to_cuda(utils.squeeze_batch(batch_0))
        batch_1 = utils.to_cuda(utils.squeeze_batch(batch_1))

        x_0 = model['ebd'](batch_0['X'])
        y_0 = batch_0['Y']
        x_1 = model['ebd'](batch_1['X'])
        y_1 = batch_1['Y']

        acc_0, loss_0, grad_0 = model['clf_all'](x_0, y_0, return_pred=False,
                                               grad_penalty=True)

        acc_1, loss_1, grad_1 = model['clf_all'](x_1, y_1, return_pred=False,
                                               grad_penalty=True)

        loss_ce = (loss_0 + loss_1) / 2.0
        regret = (grad_0 + grad_1) / 2.0

        acc = (acc_0 + acc_1) / 2.0

        weight = args.l_regret if step > args.anneal_iters else 1.0

        loss_total = loss_ce + weight * regret

        if weight > 1.0:
            loss_total /= weight

        opt.zero_grad()
        loss_total.backward()
        opt.step()

        stats['acc'].append(acc)
        stats['loss'].append(loss_total.item())
        stats['loss_train'].append(loss_ce.item())
        stats['regret'].append(regret.item())
        step += 1

    for k, v in stats.items():
        stats[k] = float(np.mean(np.array(v)))

    return stats


def test_loop(test_loader, model, ep, args, att_idx_dict=None):
    loss_list = []
    true, pred = [], []
    if att_idx_dict is not None:
        idx = []

    for batch in test_loader:
        # work on each batch
        model['ebd'].eval()
        model['clf_all'].eval()

        batch = utils.to_cuda(utils.squeeze_batch(batch))

        x = model['ebd'](batch['X'])
        y = batch['Y']

        y_hat, loss = model['clf_all'](x, y, return_pred=True)

        true.append(y)
        pred.append(y_hat)

        if att_idx_dict is not None:
            idx.append(batch['idx'])

        loss_list.append(loss.item())

    true = torch.cat(true)
    pred = torch.cat(pred)

    acc = torch.mean((true == pred).float()).item()
    loss = np.mean(np.array(loss_list))

    if att_idx_dict is not None:
        return utils.get_worst_acc(true, pred, idx, loss, att_idx_dict)

    return {
        'acc': acc,
        'loss': loss,
    }


def irm(train_data, test_data, model, opt, writer, args):
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

    # start training
    best_acc = -1
    best_val_res = None
    best_model = {}
    cycle = 0
    for ep in range(args.num_epochs):
        train_res = train_loop(train_loaders, model, opt, ep, args)

        with torch.no_grad():
            # validation
            val_res = test_loop(test_loaders[2], model, ep, args,
                                test_data.val_att_idx_dict)

        utils.print_res(train_res, val_res, ep)

        writer.add_scalar('acc/train_env', train_res['acc'], ep)
        writer.add_scalar('acc/val_env', val_res['acc'], ep)
        writer.add_scalar('loss/train_env_ce', train_res['loss_train'], ep)
        writer.add_scalar('loss/train_env_regret', train_res['regret'], ep)
        writer.add_scalar('loss/train_env_all', train_res['loss'], ep)

        if min(train_res['acc'], val_res['acc']) > best_acc:
            best_acc = min(train_res['acc'], val_res['acc'])
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
                         test_data.test_att_idx_dict)

    print('Best train')
    print(train_res)
    print('Best val')
    print(val_res)
    print('Test')
    print(test_res)

    return train_res, val_res, test_res
