import shutil

from data.cub_data import *
from model.vgg_model import *
from dis_model import *
from log import create_logger
import train_and_test as tnt
import hydra
import torchvision.transforms as transforms
from  torchvision.utils import make_grid
import time
from torch.utils.tensorboard import SummaryWriter
from helpers import *


@hydra.main(config_path='conf', config_name='config')
def run(cfg):
    data_path = cfg.root_path.path
    model_cfg = cfg.model.cfg
    model_path = cfg.model.path
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    root_dir = '/home/zhuminqin/Code/DisentangleCNN'
    model_dir = os.path.join(root_dir,'saved_model'+str(int(time.time())))
    makedir(model_dir)
    shutil.copy(os.path.join(root_dir, 'dis_model.py'), os.path.join(model_dir, 'dis_model.py'))
    shutil.copy(os.path.join(root_dir, './train_and_test.py'), os.path.join(model_dir, 'train_and_test.py'))
    shutil.copy(os.path.join(root_dir, './main.py'), os.path.join(model_dir, 'main.py'))
    log = SummaryWriter(log_dir=os.path.join(model_dir, 'log'))

    trans = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    data = CUB_dataset(data_path, trans=trans)

    train_data, test_data = data_split(data, os.path.join(data.root, 'CUB_200_2011/train_test_split.txt'))

    train_loader = DataLoader(
        train_data,
        batch_size=32,
        shuffle=False,
        collate_fn=case_collete,
        num_workers=4
    )
    test_loader = DataLoader(
        test_data,
        batch_size=32,
        shuffle=False,
        collate_fn=case_collete
    )
    # a visualize demo
    dataiter = iter(train_loader)
    _, images, labels = dataiter.next()

    # create grid of images
    img_grid = make_grid(images)

    # write to tensorboard
    log.add_image('four_fashion_mnist_images', img_grid)

    net = Dis_features(model_cfg, model_path, pretrained=False, batch_norm=True)
    net = net.cuda()

    log.add_graph(net, images.cuda())

    optimizer = torch.optim.Adam(
        params=net.parameters(),
        lr=0.0001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    for epoch in range(50):
        print('epoch: \t{0}'.format(epoch))
        acc = tnt.train(model=net, loader=train_loader, optimizer=optimizer, log=log, epoch=epoch)
        log.add_scalar('pretrain_' + 'train' + '/epoch_acc', acc * 100, epoch)
        acc = tnt.test(model=net, loader=test_loader, log=log, epoch=epoch)
        log.add_scalar('pretrain_' + 'test' + '/epoch_acc', acc * 100, epoch)

        torch.save(net.state_dict(), os.path.join(model_dir, 'cub'+str(epoch)+'_dis_adam''.pth'))


if __name__ == '__main__':
    run()
