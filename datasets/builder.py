# ------------------------------------------------------------------------------
# CoDe
# Copyright (C) 2024 by Ji-Jia Wu. All Rights Reserved.
# ------------------------------------------------------------------------------
# Modified from TCL (https://github.com/kakaobrain/tcl)
# Copyright (c) 2023 Kakao Brain. All Rights Reserved.
# ------------------------------------------------------------------------------
import os.path as osp
import random
import warnings
from functools import partial

import numpy as np
import torch.distributed as dist
import webdataset as wds
from braceexpand import braceexpand
from timm.data import create_transform
from torchvision import transforms as T
import us
import json
import io

from torch.utils.data._utils.collate import default_collate as torch_default_collate
from .noun_parser import WordAugTokenizeWrapper
import random
import torch
from sclip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from sclip import tokenize
from datasets.transforms import *

import base64
import zlib


def decode_mask(encoded_data):
    compressed_data = base64.b64decode(encoded_data['seg'])
    decompressed_data = zlib.decompress(compressed_data)
    mask_array = np.frombuffer(decompressed_data, dtype=np.uint8).reshape(encoded_data['shape']) > 0
    return mask_array


def collate(data):
    data_new = []
    for sample in data:
        assert sample["mask"].shape[0] >= 2
        index = [random.randint(0, sample["mask"].shape[0]-1) for _ in range(2)]
        while index[0] == index[1]:
            index = [random.randint(0, sample["mask"].shape[0]-1) for _ in range(2)]
        # -1代表的是不作加强的embedding
        selected_num = random.randint(0, 1)
        category = (sample['category'][index, selected_num, :] + sample['category'][index, -1, :]) / 2
        # category = sample['category'][index, :]
        text = [sample['text'][index[0]], sample['text'][index[1]]]
        data_new.append((sample['image'], category, sample['mask'][index, :], text))

    # data = [(sample['image'], sample['category'], sample['mask']) for sample in data]
    output = torch_default_collate(data_new)
    image, category, mask, text = output
    # image:    <B, 3, 224, 224>
    # category: <B, 5, 512>
    # mask:     <B, 5, 224, 224>

    return {
        "image": image,
        "category": category,
        "mask_gt": mask,
        "text": text,
    }


class NounNotEnoughError(Exception):
    pass


def worker_init_fn(worker_id, num_workers, rank, seed):
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_loader(config):
    dataset_train = build_dataset(config=config)
    us.dprint("successfully build train dataset")

    init_fn = partial(
        worker_init_fn, num_workers=config.num_workers, rank=dist.get_rank(), seed=config.seed
    )
    data_loader_train = wds.WebLoader(
        dataset_train.batched(config.batch_size, collate, partial=False),
        batch_size=None,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.num_workers > 0,
        worker_init_fn=init_fn,
    )

    train_len = len(dataset_train)
    train_nbatches = max(
        1, train_len // (config.batch_size * dist.get_world_size()))
    data_loader_train = data_loader_train.with_epoch(
        train_nbatches).with_length(train_nbatches)

    return dataset_train, data_loader_train

def warn_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning,
    and continue."""
    if isinstance(exn, NounNotEnoughError):
        return True
    warnings.warn(repr(exn))
    return True

def img_json_npz_decoder(key, value):
    if key.endswith(".jpg") or key.endswith(".png"):
        return wds.imagehandler("pil")(key, value)
    elif key.endswith(".npz"):
        masks = np.load(io.BytesIO(value))['masks']
        per_numclass = masks.shape[0]
        if per_numclass < 2:
            # print("yes")
            raise NounNotEnoughError()
        return masks
    elif key.endswith(".npy"):
        return np.load(io.BytesIO(value))
    elif key.endswith(".json"):
        value = json.loads(value)
        if len(value) < 2:
            raise NounNotEnoughError()
        return value
    return value

def build_dataset(config):
    """
    Args:
        config: CONFIG.data (CONFIG = global config)
    """
    # text_transform = TextPreprocess(
    #     num_words=config.num_words, word_type=config.word_type)
    image_mask_transform = build_img_mask_transform(config.img_aug)
    split = "train"
    dataset_type = None
    tar_file_list = []
    total_length = 0
    for ds in config.dataset[split]:
        ds_meta = config.dataset.meta[ds]
        if dataset_type is None:
            dataset_type = ds_meta.type
        else:
            assert dataset_type == ds_meta.type, "All datasets must be of the same type"

        prefix = ds_meta.prefix
        path = ds_meta.path
        length = ds_meta.length
        cur_tar_file_list = []
        for tar_file in braceexpand(osp.join(path, prefix)):
            if osp.exists(tar_file):
                cur_tar_file_list.append(tar_file)
        print(f"Found {len(cur_tar_file_list)} files for dataset {ds}")
        tar_file_list.extend(cur_tar_file_list)
        total_length += length

    print(f"Found {len(tar_file_list)} files in total for split {split}")

    dataset = (
        wds.WebDataset(tar_file_list, repeat=True, handler=warn_and_continue)
        .shuffle(40000)  # datapoint-level shuffle
        .decode(img_json_npz_decoder, handler=warn_and_continue)
        .rename(
            image="jpg",
            mask="npz",
            category="npy",
            text="json",
            keep=False,
            handler=warn_and_continue,
        )
        .map(image_mask_transform, handler=warn_and_continue)
        .with_length(total_length)
    )

    return dataset


def build_img_mask_transform(config):
    transform = Compose(
        [
            ToTensor(),
            RandomResizedCrop(config.img_size, config.img_scale),
            RandomHorizontalFlip(0.5),
            ColorJitter(0.4, 0.4, 0.4, 0.4, 0.1),
            Normalize(mean=us.DEFAULT_MEAN, std=us.DEFAULT_STD)
        ]
    )
    return transform