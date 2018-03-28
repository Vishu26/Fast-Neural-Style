from collections import defaultdict
from shutil import copyfile

import torch
from tqdm import tqdm_notebook
from torch.utils.data import DataLoader
from torch.autograd import Variable

def prep_img(img):
    return Variable(img.unsqueeze(0)).cuda()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def _fit_epoch(model, loader, criterion, optimizer):
    model.train()
    style_loss_meter = AverageMeter()
    content_loss_meter = AverageMeter()
    total_loss_meter = AverageMeter()
    t = tqdm_notebook(loader, total=len(loader))
    for data, target in t:
        data = Variable(data.cuda())
        target = Variable(target.cuda())
        output = model(data)
        style_loss, content_loss = criterion(output, data, target)
        loss = style_loss + content_loss
        style_loss_meter.update(style_loss.data[0])
        content_loss_meter.update(content_loss.data[0])
        total_loss_meter.update(loss.data[0])
        t.set_description("[ loss: {:.4f} | style loss: {:.4f} | content loss: {:.4f} ]".format(
            total_loss_meter.avg, style_loss_meter.avg, content_loss_meter.avg))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return total_loss_meter.avg, style_loss_meter.avg, content_loss_meter.avg


def fit(model, train, criterion, optimizer, batch_size=32,
        shuffle=True, nb_epoch=1, validation_data=None, cuda=True, num_workers=0):
    # TODO: implement CUDA flags, optional metrics and lr scheduler
    if validation_data:
        print('Train on {} samples, Validate on {} samples'.format(len(train), len(validation_data)))
        val_loader = DataLoader(validation_data, batch_size, shuffle, num_workers=num_workers)
    else:
        print('Train on {} samples'.format(len(train)))
    train_loader = DataLoader(train, batch_size, shuffle, num_workers=num_workers)
    history = defaultdict(list)
    t = tqdm_notebook(range(nb_epoch), total=nb_epoch)
    for epoch in t:
        loss, style_loss, content_loss = _fit_epoch(model, train_loader, criterion, optimizer)
        history['loss'].append(loss)
        history['style_loss'].append(style_loss)
        history['content_loss'].append(content_loss)
        if validation_data:
            val_loss, val_acc = validate(model, val_loader, criterion)
            print("[Epoch {} - loss: {:.4f} - val_loss: {:.4f}]".format(epoch + 1, loss, val_loss))
            history['val_loss'].append(val_loss)
        else:
            print('[ loss: {:.4f} | style loss: {:.4f} | content loss: {:.4f} ]'.format(
                loss, style_loss, content_loss))
    return history


def validate(model, val_loader, criterion):
    model.eval()
    val_loss = AverageMeter()
    for data, target in val_loader:
        data = Variable(data.cuda())
        target = Variable(target.cuda())
        output = model(data)
        loss = criterion(output, target)
        val_loss.update(loss.data[0])
    return val_loss.avg


def save_checkpoint(model_state, optimizer_state, filename, epoch=None, is_best=False):
    state = dict(model_state=model_state,
                 optimizer_state=optimizer_state,
                 epoch=epoch)
    torch.save(state, filename)
    if is_best:
        copyfile(filename, 'model_best.pth.tar')


def anneal_lr(optimizer, decay):
    for param_group in optimizer.param_groups:
        param_group['lr'] /= decay
