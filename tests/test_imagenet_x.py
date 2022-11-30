"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from imagenet_x import load_annotations, get_factor_accuracies, FACTORS
from imagenet_x.plots import model_comparison

def test_load_annotations():
    annotations = load_annotations()
    assert annotations.file_name.is_unique
    assert annotations[FACTORS].notna().all().all()
    
def test_compute_factor_accuracies():
    annotations = get_factor_accuracies('model_predictions/base')
    
def test_plot_model_comparison():
    factor_accs = get_factor_accuracies('model_predictions/base').reset_index()
    model_comparison(factor_accs, remove_underscore=False)
    