import os
from helpers import *
from torch.utils.data import Dataset, DataLoader, Subset
import torch
import random
from torchvision import transforms


def disc2cons(label, distribution, class_num, is_target):
    """
    :param label:
    for target, label = target_id
    for attribution, label={attr_id:[is_present, certainty, time]}


    :param distribution:
    for target, [mean, std] is distribution
    for attribution, distribuiton is a dict related to certainty

    :param class_num: number of targets/attributions
    :param is_target: 1 is target, 0 is attribution
    :return: continuous values, shape is (class_num,1)
    """

    con_vals = torch.zeros(class_num, 1)
    if is_target:
        for i in range(class_num):
            if i == label:
                con_vals[i] = random.gauss(1 * distribution[0], distribution[1])
            else:
                con_vals[i] = random.gauss(0 * distribution[0], distribution[1])
    else:
        for attr_id, attr_detail in enumerate(label):
            # calculate score according to is_present and certainty
            score = 0
            if int(attr_detail[0]) == 0:
                score = (4-int(attr_detail[1])) / 6
            else:
                score = (2 + int(attr_detail[1])) / 6
            con_vals[attr_id] = random.gauss(score * distribution[0], distribution[1])

    return con_vals


class CUB_dataset(Dataset):

    def __init__(self, path, trans=None, target_trans=None, attr_trans=None):
        super().__init__()
        self.root = path


        # 1. get the map of image id  to img path
        self.samples = []
        print(self.root)
        with open(os.path.join(self.root, 'CUB_200_2011/images.txt')) as f:
            for line in f:
                image_id, img_path = line.split()

                # eg: [/.../CUB_200_2011/images/001.xx/xx.jpg, 001.xx/xx.jpg]
                self.samples.append((int(image_id) - 1, os.path.join(self.root, 'CUB_200_2011/images', img_path), img_path))
        key = lambda x: x[0]

        # 2. get targets and attributions
        self.targets = []
        self.attrs = []
        with open(os.path.join(self.root, 'CUB_200_2011/classes.txt')) as f:
            for line in f.readlines():
                class_id, class_name = line.split()
                self.targets.append(class_name)
        with open(os.path.join(self.root,'attributes.txt')) as f:
            for line in f.readlines():
                attr_id, attr_name = line.split()
                self.attrs.append(attr_name)

        # 4. get the map of image to target
        self.targets_ids = []
        with open(os.path.join(self.root, 'CUB_200_2011/image_class_labels.txt')) as f:
            for line in f:
                image_id, target_id = line.split()
                self.targets_ids.append(int(target_id) - 1)

        # 5. get the map of img to attribution
        # a 2-d dictself.attrs_ids[image_id][attr_id] = [is_present, certainty, time]
        self.attrs_ids = [[] for i in range(len(self.samples))]
        with open(os.path.join(self.root, 'CUB_200_2011/attributes/my_image_attribute_labels.txt')) as f:
            for line in f:
                # print(line.split())
                image_id, attr_id, is_present, certainty, time = line.split()
                self._build_attrs_ids(int(image_id)-1, [is_present, certainty, time])

        # 6. trans is transform(torchvision.Transform)
        self.transform = trans
        self.target_transform = target_trans
        self.attr_transform = attr_trans

    def __getitem__(self, index):
        """

        :param index: data_id, not img_id
        :return:
        """
        # # 1. get img_id, target_id and attrs_id(all attribution ids of this img
        # img_id = self.data_ids[index]

        target_id = self.targets_ids[index]

        # attr_ids is a dict like {attr_id:[is_present, certainty, time]}
        attr_ids = self.attrs_ids[index]

        # 2. get image
        img_id, dir_path, dir_name = self.samples[index]
        image = pil_loader(os.path.join(self.root, 'images', dir_path))

        # 3. transform image
        if self.transform:
            image = self.transform(image)

        # 4. transform target/attribution to continuous values
        # demo distribution
        distribution = [100, 1]
        if self.target_transform:

            target_id = self.target_transform(target_id, distribution, 200, 1)

        if self.attr_transform:

            attr_ids = self.attr_transform(attr_ids, distribution, 312, 0)

        res = {
               "img_id": img_id,
               "img": image,
               "target": target_id,
               "attrs": attr_ids,
               "img_name": self.samples[index][2],
               "img_path": self.samples[index][1]}

        return res

    def __len__(self):
        return len(self.samples)

    # build the map of img_id to attr_d
    def _build_attrs_ids(self, img_id, att_detail):
        self.attrs_ids[img_id].append(att_detail)


def case_collete(cases):
    img_id = [c["img_id"] for c in cases]
    target = torch.LongTensor([c["target"] for c in cases])
    img = torch.stack([c["img"] for c in cases], 0)
    return img_id, img, target


# os.path.join(data.root, 'CUB_200_2011/train_test_split.txt')
def data_split(data: CUB_dataset, split_txt: str):
    train_list = []
    test_list = []
    with open(split_txt) as f:
        for data_id, line in enumerate(f):
            image_id, is_train = line.split()
            if is_train == '1':
                train_list.append(data_id)
            else:
                test_list.append(data_id)
    train_dataset = Subset(data, train_list)
    test_dataset = Subset(data, test_list)
    train_dataset.__setattr__("samples", [data.samples[i] for i in train_list])
    test_dataset.__setattr__("samples", [data.samples[i] for i in test_list])
    return train_dataset, test_dataset


if __name__ == '__main__':
    path = '/Usedars/zmin/PycharmProjects/Birds-200-2011/Caltech-UCSD-Birds-200-2011/data/CUB_200_2011/'
    trans = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    data = CUB_dataset(
        path,
        trans=trans,
        # target_trans=disc2cons,
        # attr_trans=disc2cons
    )
    train_data, test_data = data_split(data, os.path.join(data.root, 'CUB_200_2011/train_test_split.txt'))
    #
    train_loader = DataLoader(
        train_data,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        collate_fn=case_collete
    )
    for i, (img_id, img, target) in enumerate(train_loader):
        print(f'for img_id:{img_id} img.shape:{img.shape} target:{target}')
        break
        # img = img.cuda()
        # target = target.cuda()

    #
    # img_id_list = []
    # target_list = []
    # attrs_list = []
    # for img_id, target, attrs in loader:
    #     # print(f'target.shape:{target.shape}, attr.shape:{attrs.shape}')
    #     img_id_list += img_id
    #     target_list += target
    #     attrs_list += attrs
    #     # print(f'len:{len(res["img_id"])}')
    # img2target = dict(zip(img_id_list, target_list))
    # img2attr = dict(zip(img_id_list, attrs_list))
    #
    # # save_pkl('/Users/zmin/PLANB/util/data/CUB_img2target.pkl', img2target)
    # # save_pkl('/Users/zmin/PLANB/util/data/CUB_img2attr.pkl', img2attr)

