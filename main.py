from cub_data import *
from vgg_model import *
from log import create_logger
import train_and_test as tnt
import hydra
from torchvision import datasets, transforms


@hydra.main(config_path='conf', config_name='config')
def run(cfg):
    data_path = cfg.root_path.path
    model_cfg = cfg.model.cfg
    model_path = cfg.model.path
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    # get_root_path()
    # get_model_cfg()
    model_dir = '/home/zhuminqin/Code/DisentangleCNN/saved_model'
    log, log_close = create_logger(os.path.join(model_dir, 'train.log'))
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
    # transform = transforms.Compose([transforms.ToTensor(),
    #                                 transforms.Normalize([0.5], [0.5])])

    # data_train = datasets.CIFAR10(root=data_path,
    #                               transform=trans,
    #                               train=True,
    #                               download=False)

    # data_test = datasets.CIFAR10(root=data_path,
    #                              transform=trans,
    #                              train=False,
    #                              download=False)
    # train_loader = torch.utils.data.DataLoader(dataset=data_train,
    #                                            batch_size=64,
    #                                            shuffle=True)
    #
    # test_loader = torch.utils.data.DataLoader(dataset=data_test,
    #                                           batch_size=64,
    #                                           shuffle=True)
    # net = VGG11_bn_features(model_cfg, model_path, pretrained=False)
    log(f'model_cfg:{model_cfg}')
    log(f'model_path:{model_path}')
    net = VGG_features(model_cfg, model_path, pretrained=False, batch_norm=False)
    net = net.cuda()

    # log('start check param!')
    # param = torch.load('/home/zhuminqin/Code/DisentangleCNN/pretrained_models/cub49_vgg16.pth')
    # param_list = [[k, v] for k, v in param.items()]
    # model_list = [[k, v] for k, v in net.state_dict().items()]
    # for i, data in enumerate(model_list):
    #     k = data[0]
    #     v = data[1]
    #     # print(v.shape)
    #     # print(pred_list[i][1].shape)
    #     if not v.equal(param_list[i][1]):
    #         log(k)
    # log('check done~')
    # net_multi = torch.nn.DataParallel(net)

    optimizer = torch.optim.SGD(
        params=net.parameters(),
        lr=0.001, momentum=0.9)
    for epoch in range(50):
        log('epoch: \t{0}'.format(epoch))
        tnt.train(model=net, loader=train_loader, optimizer=optimizer, log=log)
        tnt.test(model=net, loader=test_loader, log=log)
    torch.save(net.state_dict(), os.path.join(model_dir, 'cub150_vgg16.pth'))
    log_close()


if __name__ == '__main__':
    run()
