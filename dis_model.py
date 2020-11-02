import os

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.functional as F


def local_filter(x):
    # reshape in_fts from [b,c,3,3] to [b,c, 3*3]
    in_fts = x.reshape(x.shape[0], x.shape[1], -1)

    # center feature
    cent_f = in_fts[:, :, in_fts.shape[2] // 2]
    # lf is local filter,shape is [batch,channel,3*3]
    lf = cent_f.reshape((x.shape[0], x.shape[1], -1))
    # print(f"lf.shape:{lf.shape}")
    lf = lf.expand((-1, -1, in_fts.shape[2]))
    # print(f"lf.shape:{lf.shape}")
    # l2 distance
    dis = -torch.sum(((lf - in_fts) ** 2), 1)
    ep = 1000
    dis[:, in_fts.shape[2] // 2] -= ep
    # use softmax to get score
    sc = torch.softmax(dis, -1)
    sc = sc.unsqueeze(1)
    sc = sc.expand(-1, in_fts.shape[1], -1)
    # cent value move to neighbor value
    out_fts = torch.sum(sc.mul(in_fts - lf), -1)
    out_fts += cent_f

    return out_fts


class Dis_Net(nn.Module):

    def __init__(self, cfg, num_classes=200, batch_norm=False, init_weights=True):
        super(Dis_Net, self).__init__()

        self.batch_norm = batch_norm

        self.kernel_sizes = []
        self.strides = []
        self.paddings = []

        self.features0 = self._make_layers(cfg[0], batch_norm, 3)
        self.features1 = self._make_layers(cfg[1], batch_norm, cfg[0][0])
        self.features2 = self._make_layers(cfg[2], batch_norm, cfg[1][0])
        self.features3 = self._make_layers(cfg[3], batch_norm, cfg[2][0])
        self.features4 = self._make_layers(cfg[4], batch_norm, cfg[3][0])
        self.bn = torch.nn.BatchNorm2d(cfg[3][0])
        self.avgpool = nn.AdaptiveAvgPool2d((7,7))
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )
        if init_weights:
            self._init_weights()

    def forward(self, x):

        x = self.features0(x)
        # x = cluster(x)
        x = self.features1(x)
        # x = cluster(x)
        x = self.features2(x)
        # x = cluster(x)
        x = self.features3(x)
        x = cluster(x)
        x = self.bn(x)
        x = self.features4(x)

        # x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _make_layers(self, cfg, batch_norm, in_channels):

        self.n_layers = 0

        layers = []
        # in_channels = 3
        for m in cfg:
            if m == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

                self.kernel_sizes.append(2)
                self.strides.append(2)
                self.paddings.append(0)
            # # soft cluster
            # elif m == 'C':
            #     layers += [cluster()]

            else:
                conv2d = nn.Conv2d(in_channels, m, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(m), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]

                self.n_layers += 1

                self.kernel_sizes.append(3)
                self.paddings.append(1)
                self.strides.append(1)

                in_channels = m

        return nn.Sequential(*layers)

    def conv_info(self):
        return self.kernel_sizes, self.strides, self.paddings

    def num_layers(self):
        return self.n_layers

    def __repr__(self):
        template = 'VGG: {} batch_norm: {}'
        return template.format(self.num_layers() + 3, self.batch_norm)


def cluster(features):
    in_fts = features
    out_fts = torch.empty(features.shape).cuda()
    for i in range(features.shape[2]-3):
        for j in range(features.shape[3]-3):
            local_in_fts = features[:, :, i:i+3, j:j+3]
            local_out_fts = local_filter(local_in_fts)
            # print(f'out_fts[:,:,i,j]:{out_fts[:,:,i,j].shape}, local_out_fts:{local_out_fts.shape}')
            out_fts[:,:,i,j] = local_out_fts
    return out_fts.cuda()


def Dis_features(cfg, model_path, pretrained=True, batch_norm=True, **kwargs):

    if pretrained:
        kwargs["init_weights"] = False
    print(f'cfg[0]:{cfg[0]}')
    model = Dis_Net(cfg, batch_norm=batch_norm, **kwargs)

    if pretrained:
        # my_dict = model_zoo.load_url(model_urls['vgg11_bn'], model_dir=model_dir)
        # TODO: my operation
        model_dict = model.state_dict()
        for i, k in enumerate(model_dict):
            print(model_dict[k].shape)
            if i > 4:
                break
        model_dict = torch.load(model_path)
        param_map = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4]
        param_map2 = [0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2, 4, 4, 0, 0, 2, 2, 4, 4, 0, 0, 2, 2, 4, 4]
        param = {}
        for i, k in enumerate(model_dict.keys()):
            new_key = k
            k_list = k.split('.')
            # print(k_list[-1])
            if i < len(param_map):
                new_key = 'features' + str(param_map[i]) + '.' + str(param_map2[i]) + '.' + k_list[-1]
            # print(new_key)
            param[new_key] = model_dict[k]
        model.load_state_dict(param)
    else:
    #     # pass
        pass
        # params = {k: v for k, v in pret_dcit.items() if k in model_dict and not k.startswith('classifier.6')}
        # model_dict.update(params)
        # model.load_state_dict(model_dict)
        # model_dict = model.state_dict()
        # torch.save(model_dict, os.path.join('/home/zhuminqin/Code/DisentangleCNN/pretrained_models/', 'demo.pth'))
        # for k,v in model_dict.items():
        #     print(f'k:{k}, v:{v.shape}, pret_dcit:{pret_dcit[k].shape}')
        #     if k in pret_dcit:
        #         if pret_dcit[k] == v:
        #             continue

        # print(f'load successfully!')
    return model
