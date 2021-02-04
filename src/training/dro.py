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


def train_loop(train_loaders, model, opt, ep, args):
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

        if args.dataset in ['beer_0', 'beer_1', 'beer_2', 'pubmed']:
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


def dro(train_data, test_data, model, opt, writer, args):
    # define groups by env-label pair
    train_loaders = []
    for env in range(2):
        # look at each environment, each label
        groups = {}
        label_list = train_data.get_all_y(env)
        for idx, label in zip(train_data.envs[env]['idx_list'], label_list):
            if label not in groups:
                groups[label] = [idx]
            else:
                groups[label].append(idx)

        for group in groups.values():
            train_loaders.append(DataLoader(
                train_data,
                sampler=EnvSampler(args.num_batches, args.batch_size, env,
                                   group),
            num_workers=5))

    test_loaders = []
    for i in range(4):
        test_loaders.append(DataLoader(
            test_data,
            sampler=EnvSampler(-1, args.batch_size, i,
                               test_data.envs[i]['idx_list']), num_workers=10))

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
                         test_data.test_att_idx_dict)
    print('Best train')
    print(train_res)
    print('Best val')
    print(val_res)
    print('Test')
    print(test_res)

    return train_res, val_res, test_res
