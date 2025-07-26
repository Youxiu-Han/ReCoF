import logging
import math
import os
import random
import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset
from .randaugment import RandAugmentMC
import torch.utils.data as data
logger = logging.getLogger(__name__)

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)

def get_cifar(args, root_path, data_path):

    if args.dataset == 'cifar10':
        base_dataset = datasets.CIFAR10(root_path, train=True, download=False)
        test_dataset = datasets.CIFAR10(root_path, train=False, download=False)
        mean = cifar10_mean
        std = cifar10_std
        all_classes = 10
        num_unlabeled = 20000
        

        args.num_classes = 6
        
        base_dataset.targets = np.array(base_dataset.targets)
        test_dataset.targets = np.array(test_dataset.targets)
        

        base_dataset.targets -= 2
        base_dataset.targets[np.where(base_dataset.targets == -2)[0]] = 8  # 0→8
        base_dataset.targets[np.where(base_dataset.targets == -1)[0]] = 9  # 1→9
        
        test_dataset.targets -= 2
        test_dataset.targets[np.where(test_dataset.targets == -2)[0]] = 8  # 0→8
        test_dataset.targets[np.where(test_dataset.targets == -1)[0]] = 9  # 1→9

        actual_classes = 10
        
    elif args.dataset == 'cifar100':
        base_dataset = datasets.CIFAR100(root_path, train=True, download=False)
        test_dataset = datasets.CIFAR100(root_path, train=False, download=False)
        mean = cifar100_mean
        std = cifar100_std
        all_classes = 100
        num_unlabeled = 20000
        args.num_classes = 50
        
        base_dataset.targets = np.array(base_dataset.targets)
        test_dataset.targets = np.array(test_dataset.targets)
        actual_classes = all_classes
        

    train_labeled_idxs, train_unlabeled_idxs, val_idxs = split_l_u_scomatch(
        args, base_dataset.targets, actual_classes, num_unlabeled)
    

    norm_func = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    norm_func_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    

    if args.dataset == 'cifar10':
        train_labeled_dataset = CIFAR10SS(
            root_path, train_labeled_idxs, train=True, transform=norm_func,
            mapped_targets=base_dataset.targets)
        train_unlabeled_dataset = CIFAR10SS(
            root_path, train_unlabeled_idxs, train=True, 
            transform=TransformFixMatch(mean=cifar10_mean, std=cifar10_std, norm=True),
            mapped_targets=base_dataset.targets)
        val_dataset = CIFAR10SS(
            root_path, val_idxs, train=True, transform=norm_func_test,
            mapped_targets=base_dataset.targets)
    elif args.dataset == 'cifar100':
        train_labeled_dataset = CIFAR100SS(
            root_path, train_labeled_idxs, train=True, transform=norm_func)
        train_unlabeled_dataset = CIFAR100SS(
            root_path, train_unlabeled_idxs, train=True, 
            transform=TransformFixMatch(mean=cifar100_mean, std=cifar100_std, norm=True))
        val_dataset = CIFAR100SS(
            root_path, val_idxs, train=True, transform=norm_func_test)
    

    test_dataset.transform = norm_func_test
    

    id_mask = test_dataset.targets < args.num_classes
    test_dataset.data = test_dataset.data[id_mask]
    test_dataset.targets = test_dataset.targets[id_mask]
    
    return train_labeled_dataset, train_unlabeled_dataset, val_dataset, test_dataset


class TransformFixMatch(object):
    def __init__(self, mean, std, norm=True, size_image=32):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=size_image,
                                  padding=int(size_image*0.125),
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=size_image,
                                  padding=int(size_image*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)
        ])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])
        self.norm = norm

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        if self.norm:
            return self.normalize(weak), self.normalize(strong)
        else:
            return weak, strong

class CIFAR10SS(datasets.CIFAR10):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False, mapped_targets=None):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        

        if mapped_targets is not None:
            self.targets = mapped_targets
            
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

class CIFAR100SS(datasets.CIFAR100):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index



def split_l_u_scomatch(args, labels, all_classes, num_unlabeled):

    if args.dataset == 'cifar10':
        label_per_class = 400
    elif args.dataset == 'cifar100':
        label_per_class = 100
    elif args.dataset == 'tinyimagenet':
        label_per_class = 100
    elif args.dataset == 'mnist':
        label_per_class = 10
    else:
        label_per_class = args.num_labeled
        
    val_per_class = args.num_val if hasattr(args, 'num_val') else 20
    mismatch_ratio = args.mismatch_ratio if hasattr(args, 'mismatch_ratio') else 0.3
    
    labels = np.array(labels)
    labeled_idx = []
    val_idx = []
    unlabeled_idx = []
    

    unlabel_number_per_id_class = int(num_unlabeled * (1.0 - mismatch_ratio)) // args.num_classes
    n_unlabels_shift = (num_unlabeled - (unlabel_number_per_id_class * args.num_classes)) // \
                       (all_classes - args.num_classes)
    

    for i in range(args.num_classes):
        idx = np.where(labels == i)[0]

        labeled_idx.extend(idx[:label_per_class])

        val_idx.extend(idx[label_per_class:label_per_class+val_per_class])

        start_idx = label_per_class + val_per_class
        end_idx = start_idx + unlabel_number_per_id_class
        unlabeled_idx.extend(idx[start_idx:end_idx])
    

    for i in range(args.num_classes, all_classes):
        idx = np.where(labels == i)[0]
        unlabeled_idx.extend(idx[:n_unlabels_shift])
    
    labeled_idx = np.array(labeled_idx)
    unlabeled_idx = np.array(unlabeled_idx)
    val_idx = np.array(val_idx)
    
    logger.info(f"Mismatch ratio: {mismatch_ratio}")
    logger.info(f"Labeled examples: {len(labeled_idx)}")
    logger.info(f"Unlabeled examples: {len(unlabeled_idx)}")
    logger.info(f"Validation examples: {len(val_idx)}")
    
    return labeled_idx, unlabeled_idx, val_idx

# ----------------------------- TinyImageNet -----------------------------
def default_loader(path):
    return Image.open(path).convert('RGB')

class TinyImageFolderSS(data.Dataset):

    def __init__(self, image_list, label_list, root, 
                 transform=None, target_transform=None, loader=default_loader):
        self.imgs = image_list
        self.labels = np.array(label_list)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.root = root

    def __getitem__(self, index):

        path = os.path.join(self.root, self.imgs[index])
        target = self.labels[index]
        img = self.loader(path)
        
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target, index

    def __len__(self):
        return len(self.imgs)

def get_tinyimagenet(args, root_path, data_path):

    mean = normal_mean
    std = normal_std
    all_classes = 200
    args.num_classes = 100
    
    labeled_per_class = 100
    unlabeled_number = 40000
    mismatch_ratio = args.mismatch_ratio if hasattr(args, 'mismatch_ratio') else 0.3
    

    unlabel_number_per_id_class = int(unlabeled_number * (1.0 - mismatch_ratio)) // args.num_classes
    n_unlabels_shift = (unlabeled_number - (unlabel_number_per_id_class * args.num_classes)) // \
                       (all_classes - args.num_classes)
    

    txt_file = os.path.join(root_path, "tiny-imagenet/train_all.txt")
    val_txt_file = os.path.join(root_path, "tiny-imagenet/val_all.txt")
    

    if not os.path.exists(txt_file):
        raise FileNotFoundError(f"TinyImageNet train file not found: {txt_file}")
    
    labeled_idxs = []
    labeled_labs = []
    unlabeled_idxs = []
    unlabeled_labs = []
    
    new_anno = [[] for i in range(all_classes)]
    with open(txt_file,'r') as f:
        annos = [[line.split(' ')[0], int(line.split(' ')[1])] for line in f.readlines()]
    random.shuffle(annos)
    for anno in annos:
        new_anno[anno[1]].append(anno[0])
    

    for i in range(args.num_classes):
        labeled_idxs += new_anno[i][:labeled_per_class]
        labeled_labs += [i for j in range(labeled_per_class)]
    
    for i in range(args.num_classes):
        unlabeled_idxs += new_anno[i][labeled_per_class:labeled_per_class+unlabel_number_per_id_class]
        unlabeled_labs += [i for j in range(unlabel_number_per_id_class)]
    
    for i in range(args.num_classes, all_classes):
        unlabeled_idxs += new_anno[i][:n_unlabels_shift]
        unlabeled_labs += [i for j in range(n_unlabels_shift)]
    
    del annos
    del new_anno
    

    val_idxs = []
    val_labs = []
    if os.path.exists(val_txt_file):
        with open(val_txt_file, 'r') as f:
            annos = [[line.split(' ')[0], int(line.split(' ')[1])] for line in f.readlines()]

        for anno in annos:
            if anno[1] < args.num_classes:
                val_idxs.append(anno[0])
                val_labs.append(anno[1])

    norm_func_labeled = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),  
        transforms.Normalize(mean=mean, std=std)
    ])
    norm_func_unlabeled = TransformFixMatch_TinyImagenet(mean=mean, std=std, norm=True, size_image=32)
    norm_func_test = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),  
        transforms.Normalize(mean=mean, std=std)
    ])
    

    train_labeled_dataset = TinyImageFolderSS(
        labeled_idxs, labeled_labs, 
        root=os.path.join(root_path, "tiny-imagenet-200/"),
        transform=norm_func_labeled)
    train_unlabeled_dataset = TinyImageFolderSS(
        unlabeled_idxs, unlabeled_labs,
        root=os.path.join(root_path, "tiny-imagenet-200/"),
        transform=norm_func_unlabeled)
    val_dataset = TinyImageFolderSS(
        val_idxs, val_labs,
        root=os.path.join(root_path, "tiny-imagenet-200/"),
        transform=norm_func_test)

    test_dataset = TinyImageFolderSS(
        val_idxs, val_labs,
        root=os.path.join(root_path, "tiny-imagenet-200/"),
        transform=norm_func_test)
    
    logger.info(f"TinyImageNet Mismatch ratio: {mismatch_ratio}")
    logger.info(f"Labeled examples: {len(train_labeled_dataset)}")
    logger.info(f"Unlabeled examples: {len(train_unlabeled_dataset)}")
    logger.info(f"Validation examples: {len(val_dataset)}")
    
    return train_labeled_dataset, train_unlabeled_dataset, val_dataset, test_dataset


class TransformFixMatch_TinyImagenet(object):
    def __init__(self, mean, std, norm=True, size_image=32):
        self.weak = transforms.Compose([
            transforms.Resize((size_image, size_image)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=size_image,
                                  padding=int(size_image*0.125),
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.Resize((size_image, size_image)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=size_image,
                                  padding=int(size_image*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])
        self.norm = norm

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        if self.norm:
            return self.normalize(weak), self.normalize(strong)
        else:
            return weak, strong

DATASET_GETTERS_MISMATCH = {
    'cifar10': get_cifar,
    'cifar100': get_cifar,
    'tinyimagenet': get_tinyimagenet,
}