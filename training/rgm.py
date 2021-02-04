import copy
import random
import datetime
from termcolor import colored
from tqdm import tqdm
import torch
import numpy as np
import training.utils as utils
from data_utils import EnvSampler
from torch.utils.data import DataLoader


def train_loop(train_loaders, model, opt_all, opt_0, opt_1, ep, args):
    stats = {}
    for k in ['acc', 'loss', 'regret', 'loss_train']:
        stats[k] = []

    step = ep * args.num_batches
    for batch_0, batch_1 in zip(train_loaders[0], train_loaders[1]):
        # work on each batch
        model['ebd'].train()
        model['clf_all'].train()
        model['clf_0'].train()
        model['clf_1'].train()

        batch_0 = utils.to_cuda(utils.squeeze_batch(batch_0))
        batch_1 = utils.to_cuda(utils.squeeze_batch(batch_1))

        x_0 = model['ebd'](batch_0['X'])
        y_0 = batch_0['Y']
        x_1 = model['ebd'](batch_1['X'])
        y_1 = batch_1['Y']

        # train clf_0 on x_0
        _, loss_0_0 = model['clf_0'](x_0.detach(), y_0)
        opt_0.zero_grad()
        loss_0_0.backward()
        opt_0.step()

        # train clf_1 on x_1
        _, loss_1_1 = model['clf_1'](x_1.detach(), y_1)
        opt_1.zero_grad()
        loss_1_1.backward()
        opt_1.step()

        # train clf_all on both, backprop to representation
        x = torch.cat([x_0, x_1], dim=0)
        y = torch.cat([y_0, y_1], dim=0)
        acc, loss_ce = model['clf_all'](x, y)

        # randomly sample a group, evaluate the validation loss
        # do not detach feature representation at this time
        if random.random() > 0.5:
            # choose env 1
            # apply clf 1 on env 1
            _, loss_1_1 = model['clf_1'](x_1, y_1)

            # apply clf 0 on env 1
            _, loss_0_1 = model['clf_0'](x_1, y_1)

            regret = loss_0_1 - loss_1_1
        else:
            # chosse env 0
            # apply clf 1 on env 0
            _, loss_1_0 = model['clf_1'](x_0, y_0)

            # apply clf 0 on env 0
            _, loss_0_0 = model['clf_0'](x_0, y_0)

            regret = loss_1_0 - loss_0_0

        weight = args.l_regret if step > args.anneal_iters else 1.0
        loss = loss_ce + weight * regret
        step += 1

        if weight > 1.0:
            loss /= weight

        opt_all.zero_grad()
        opt_0.zero_grad()
        opt_1.zero_grad()
        loss.backward()
        opt_all.step()

        stats['acc'].append(acc)
        stats['loss'].append(loss.item())
        stats['loss_train'].append(loss_ce.item())
        stats['regret'].append(regret.item())

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


def rgm(train_data, test_data, model, opt, writer, args):
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
    opt_all = opt[0]
    opt_0 = opt[1]
    opt_1 = opt[2]
    for ep in range(args.num_epochs):
        train_res = train_loop(train_loaders, model, opt_all, opt_0, opt_1, ep, args)

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
