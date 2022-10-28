"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import transforms
    from torchvision import datasets
except ImportError:
    raise ImportError("Please install pytorch and torchvision to use this module.")

from distutils.log import warn
from imagenet_x.utils import files
import imagenet_x.annotations

import numpy as np
import pandas as pd
from io import BytesIO
import base64


try:
    from IPython.display import HTML
except ImportError:
    import warnings
    warn("IPython not installed, cannot display images")
    
from imagenet_x import load_annotations, FACTORS


class ImageNetXImageFolder(datasets.ImageFolder):
    """Loads ImageNetX annotations along with Imagenet validation samples"""
    
    def __init__(self, imagenet_path, *args, which_factor="top", partition="val", filter_prototypes=True, **kwargs):
        super().__init__(imagenet_path, *args, **kwargs)
        self.annotations_ = load_annotations(which_factor=which_factor, partition=partition, filter_prototypes=filter_prototypes).set_index('file_name')
        # Filter out unanotated samples
        self.samples = [(path, label) for (path, label) in self.samples if path.split("/")[-1] in self.annotations_.index]

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        img_id = self.samples[index][0].split("/")[-1]
        img_annotations = self.annotations_.loc[img_id]
        return img, target, img_annotations[FACTORS].values.astype(np.bool)
    
class ImageNetX(datasets.ImageNet):
    """Loads ImageNetX annotations along with Imagenet validation samples"""
    
    def __init__(self, imagenet_path, *args, which_factor="top", partition="val", filter_prototypes=True, **kwargs):
        super().__init__(imagenet_path, split=partition, **kwargs)
        self.annotations_ = load_annotations(which_factor=which_factor, partition=partition, filter_prototypes=filter_prototypes).set_index('file_name')
        # Filter out unanotated samples
        self.samples = [(path, label) for (path, label) in self.samples if path.split("/")[-1] in self.annotations_.index]

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        img_id = self.samples[index][0].split("/")[-1]
        img_annotations = self.annotations_.loc[img_id]
        return img, target, img_annotations[FACTORS].values.astype(np.bool)


def get_vanilla_transform():
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )

def format_dataset_entry(inp):
    img, label, factors = inp
    imagenet_classes = pd.read_csv((files(imagenet_x.annotations) / 'imagenet_labels.txt'), names=['class', 'name']).name.to_list()
    label = imagenet_classes[label]
    factor = FACTORS[np.nonzero(factors)[0][0]]
    return dict(Class=label, Factor=factor, image=img)

def image_base64(im):
    with BytesIO() as buffer:
        im.resize((128,128)).save(buffer, 'jpeg')
        return base64.b64encode(buffer.getvalue()).decode()

def image_formatter(im):
    return f'<img src="data:image/jpeg;base64,{image_base64(im)}">'

def reshape(df, rows=3):
    length = len(df)
    cols = np.ceil(length / rows).astype(int)
    df = df.assign(rows=np.tile(np.arange(rows), cols)[:length], 
                   cols=np.repeat(np.arange(cols), rows)[:length]) \
           .pivot('rows', 'cols', df.columns.tolist()) \
           .sort_index(level=1, axis=1).droplevel(level=1, axis=1).rename_axis(None)
    return df

# sample 10 random images from the dataset with numpy indices
def display_sample_df(dataset, nb_rows=3, nb_cols=3):
    sample = np.random.choice(len(dataset), nb_rows*nb_cols, replace=False)
    sample_df = pd.DataFrame([format_dataset_entry(dataset[i]) for i in sample])
    sample_df = reshape(sample_df, nb_rows)
    return HTML(sample_df.to_html(formatters={'image': image_formatter}, escape=False, index=False))