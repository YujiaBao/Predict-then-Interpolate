from training.irm import irm
from training.erm import erm
from training.rgm import rgm
from training.dro import dro
from training.ours import ours
from training.oracle import oracle


def train_val_test(train_data, test_data, model, opt, writer, args):
    if args.method == 'irm':
        return irm(train_data, test_data, model, opt, writer, args)

    if args.method == 'erm':
        return erm(train_data, test_data, model, opt, writer, args)

    if args.method == 'rgm':
        return rgm(train_data, test_data, model, opt, writer, args)

    if args.method == 'dro':
        return dro(train_data, test_data, model, opt, writer, args)

    if args.method == 'ours':
        return ours(train_data, test_data, model, opt, writer, args)

    if args.method == 'oracle':
        return oracle(train_data, test_data, model, opt, writer, args)

    raise ValueError('method {} is not impelmented'.format(args.method))

