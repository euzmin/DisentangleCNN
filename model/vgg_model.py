import os

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class VGG_Net(nn.Module):

    def __init__(self, cfg, num_classes=200, batch_norm=False, init_weights=True):
        super(VGG_Net, self).__init__()

        self.batch_norm = batch_norm

        self.kernel_sizes = []
        self.strides = []
        self.paddings = []

        self.features = self._make_layers(cfg, batch_norm)
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
        x = self.features(x)
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

    def _make_layers(self, cfg, batch_norm):

        self.n_layers = 0

        layers = []
        in_channels = 3
        for m in cfg:
            if m == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

                self.kernel_sizes.append(2)
                self.strides.append(2)
                self.paddings.append(0)

            elif m == 'I':
                pass

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


def VGG_features(cfg, model_path, pretrained=True, batch_norm=True, **kwargs):

    if pretrained:
        kwargs["init_weights"] = False

    model = VGG_Net(cfg, batch_norm=batch_norm, **kwargs)

    if pretrained:
        # my_dict = model_zoo.load_url(model_urls['vgg11_bn'], model_dir=model_dir)
        # TODO: my operation
        pass
    else:
    #     # pass
        model_dict = model.state_dict()
        pret_dcit = torch.load(model_path)
        params = {k: v for k, v in pret_dcit.items() if k in model_dict and not k.startswith('classifier.6')}
        model_dict.update(params)
        model.load_state_dict(model_dict)
        # model_dict = model.state_dict()
        # torch.save(model_dict, os.path.join('/home/zhuminqin/Code/DisentangleCNN/pretrained_models/', 'demo.pth'))
        # for k,v in model_dict.items():
        #     print(f'k:{k}, v:{v.shape}, pret_dcit:{pret_dcit[k].shape}')
        #     if k in pret_dcit:
        #         if pret_dcit[k] == v:
        #             continue

        # print(f'load successfully!')
    return model
