import torch.utils.data as data
import numpy as np
from imageio import imread
from path import Path
import random


def load_as_float(path):
    return imread(path).astype(np.float32)


class SequenceFolder(data.Dataset):#对数据集进一步包装，完了返回给dataloader
    """A sequence data loader where the files are arranged in this way:
        root/scene_1/0000000.jpg
        root/scene_1/0000001.jpg
        ..
        root/scene_1/cam.txt
        root/scene_2/0000000.jpg
        .

        transform functions must take in a list a images and a numpy array (usually intrinsics matrix)
    """

    def __init__(self, root, seed=None, train=True, sequence_length=3, transform=None, target_transform=None):
        np.random.seed(seed)#不按顺序随机抽取数据作为训练/测试集
        random.seed(seed)
        self.root = Path(root)#precessed_data/
        scene_list_path = self.root/'train.txt' if train else self.root/'val.txt'#processed_data/train.txt
        self.scenes = [self.root/folder[:-1] for folder in open(scene_list_path)]#list of [Path(processed_data/scenes_1),processed_data/scenes_2...

        self.transform = transform#compose 一个函数
        self.crawl_folders(sequence_length)

    def crawl_folders(self, sequence_length):
        sequence_set = []
        demi_length = (sequence_length-1)//2
        shifts = list(range(-demi_length, demi_length + 1))
        shifts.pop(demi_length)
        for scene in self.scenes:
            #这里了
            intrinsics = np.genfromtxt(scene/'cam.txt').astype(np.float32).reshape((3, 3))#这个intrinsics是kitti自带的，
                        # calib_cam_to_cam.txt projection matrix after rectification
            imgs = sorted(scene.files('*.jpg'))
            if len(imgs) < sequence_length:
                continue
            for i in range(demi_length, len(imgs)-demi_length):
                sample = {'intrinsics': intrinsics, 'tgt': imgs[i], 'ref_imgs': []}
                for j in shifts:
                    sample['ref_imgs'].append(imgs[i+j])
                sequence_set.append(sample)
        random.shuffle(sequence_set)
        self.samples = sequence_set#list of dict ,lenth 306 dict={intrinsics:ndarray;tgt:Path; ref_imgs:list} list of ref_imgs: sql-1 lenth Paths

    def __getitem__(self, index):
        sample = self.samples[index]#get a dict={intrinsics:ndar;tgt:Path; ref_imgs:list} list of ref_imgs: sql-1 lenth Paths
        tgt_img = load_as_float(sample['tgt'])
        ref_imgs = [load_as_float(ref_img) for ref_img in sample['ref_imgs']]
        if self.transform is not None:
                                            #(H,W,3)        sql-1 list (H,W,3)
            imgs, intrinsics = self.transform([tgt_img] + ref_imgs, np.copy(sample['intrinsics']))#深复制一个nda（3,3）
            tgt_img = imgs[0]
            ref_imgs = imgs[1:]
        else:
            intrinsics = np.copy(sample['intrinsics'])
        return tgt_img, ref_imgs, intrinsics, np.linalg.inv(intrinsics)

    def __len__(self):
        return len(self.samples)
