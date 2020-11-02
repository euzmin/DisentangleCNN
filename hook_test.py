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
import PIL.Image as Image

# 全局变量，用于存储中间层的 feature
total_feat_out = []
total_feat_in = []
total_feat_name = []

# 定义 forward hook function
def hook_fn_forward(module, input, output):
    print(module) # 用于区分模块
    # print('input', input) # 首先打印出来
    print('output', output)
    total_feat_out.append(output) # 然后分别存入全局 list 中

    # total_feat_in.append(input)

@hydra.main(config_path='conf', config_name='config')
def test(cfg):
    data_path = cfg.root_path.path
    model_cfg = cfg.model.cfg
    model_path = cfg.model.path
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    root_dir = '/home/zhuminqin/Code/DisentangleCNN'
    model_dir = os.path.join(root_dir,'test_model'+str(int(time.time())))
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
    # data = CUB_dataset(data_path, trans=trans)
    #
    # train_data, test_data = data_split(data, os.path.join(data.root, 'CUB_200_2011/train_test_split.txt'))
    #
    # train_loader = DataLoader(
    #     train_data,
    #     batch_size=4,
    #     shuffle=False,
    #     collate_fn=case_collete,
    # )
    # test_loader = DataLoader(
    #     test_data,
    #     shuffle=False,
    #     collate_fn=case_collete
    # )

    net = Dis_features(model_cfg, model_path, pretrained=False, batch_norm=True)
    net = net.cuda()
    modules = net.named_children()  #
    for name, module in modules:
        print(name)
        total_feat_name.append(name)
        if name == 'features3':
            module.register_forward_hook(hook_fn_forward)

    optimizer = torch.optim.Adam(
        params=net.parameters(),
        lr=0.0001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    img_path = os.path.join(data_path, 'CUB_200_2011/images/001.Black_footed_Albatross/Black_Footed_Albatross_0003_796136.jpg')
    img = trans(Image.open(img_path).convert('RGB'))
    img = img.unsqueeze(0).cuda()
    output = net(img)

    print('==========Saved inputs and outputs==========')
    with open(os.path.join(model_dir, 'hook_feat.txt'), 'w') as f:
        for idx in range(len(total_feat_out)):
            # print('input: ', total_feat_in[idx])
            print('output: ', total_feat_out[idx])
            if idx < len(total_feat_name):
                f.write('name:'+str(total_feat_name[idx])+'\n')
           #  f.write('input:'+str(total_feat_in[idx])+'\n')
            save_pkl(os.path.join(model_dir, 'hook_feat.pkl'), total_feat_out[idx])
            f.write('output:'+str(total_feat_out[idx])+'\n')
            # module.register_backward_hook(hook_fn_backward)


if __name__ == '__main__':
    test()
