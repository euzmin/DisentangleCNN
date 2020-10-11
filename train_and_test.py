import time

import torch
import torch.nn as nn


def train_or_test(model, loader, optimizer=None, log=print):

    is_train = optimizer is not None
    log(f'is_train : {is_train} and optimizer is {optimizer is not None}')
    start = time.time()
    n_items = 0
    n_correct = 0
    n_batches = 0
    total_cross_entropy = 0

    # for i, data in enumerate(loader):
    #     img, target = data
    #     img = img.cuda()
    #     target = target.cuda()
    for i, (img_id, img, target) in enumerate(loader):
        img = img.cuda()
        target = target.cuda()

        # if train then calculate grad else not
        grad_cal = torch.enable_grad() if is_train else torch.no_grad()

        with grad_cal:
            output = model(img)
            # output.shape:(batch_size,200) target.shape:(batch_size)
            # log(f'output type:{output.shape} target type:{target.shape}')
            loss = nn.functional.cross_entropy(output, target)

            pred_out, predicted = output.max(1)
            # log(f'output:{output.data[0]}')
            # log(f'predicted:{predicted.shape}')
            n_items += target.shape[0]

            # log(f'predicted:{pred_out}')
            # log(f'predicted:{predicted}')
            # log(f'target:{target}')
            # log(f'n_items:{n_items} n_correct:{n_correct}')

            n_correct += (predicted == target).sum().item()


            n_batches += 1
            total_cross_entropy += loss.item()

            if is_train:
                # log(f'\t\t start backward! ')
                optimizer.zero_grad()
                loss.backward()
                # log(f'before step param :{model.state_dict()["features.0.weight"]}')
                optimizer.step()
                # log(f'after step param :{model.state_dict()["features.0.weight"]}')
        # log(f'\t\t ce {loss.item()}')
        # log(f'\t\t correct:{n_correct}')
        del img
        del target
        del output
        del predicted


    end = time.time()

    log(f'\ttime:{end-start}')
    log(f'\tce per batch:{total_cross_entropy/n_batches}')
    log(f'\tn_correct:{n_correct} n_items:{n_items} acc: {n_correct/n_items * 100}%')
    return n_correct / n_items


def train(model, loader, optimizer, log=print):
    assert (optimizer is not None)
    log('\ttrain:')
    model.train()
    return train_or_test(model=model, loader=loader, optimizer=optimizer, log=log)


def test(model, loader, log):
    log('\ttest:')
    model.eval()
    return train_or_test(model=model, loader=loader, log=log)

