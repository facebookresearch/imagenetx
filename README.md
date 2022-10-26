## Installation

If you just want to load the annotations :

```bash
pip install imagenet-x
```

Or if you've cloned this repo 

```bash
pip install -e .
```

If you want to reproduce plots from the paper or use the plots subpackage

```bash
pip install imagenet-x[plot]
```

If you want to use the evaluate subpackage you will need a working installation of pytorch

```bash
pip install imagenet-x[evaluate]
```


## Usage

### To load the annotations

```python
from imagenet_x import load_annotations

annotations = load_annotations()
```

This will output the following table

| file_name                    |   class |   multiple_objects |   background |   color |   brighter |   darker |   style |   larger |   smaller |   object_blocking |   person_blocking |   partial_view |   pattern |   pose |   shape |   subcategory |   texture | justification                                   | one_word          | metaclass   |
|:-----------------------------|--------:|-------------------:|-------------:|--------:|-----------:|---------:|--------:|---------:|----------:|------------------:|------------------:|---------------:|----------:|-------:|--------:|--------------:|----------:|:------------------------------------------------|:------------------|:------------|
| ILSVRC2012_val_00004487.JPEG |     762 |                  0 |            0 |       0 |          0 |        0 |       0 |        1 |         0 |                 0 |                 0 |              0 |         0 |      0 |       0 |             0 |         0 | close up of a pan fried sea bass.               | sea bass close up | structure   |
| ILSVRC2012_val_00003963.JPEG |     292 |                  0 |            0 |       1 |          0 |        0 |       0 |        0 |         0 |                 0 |                 0 |              0 |         0 |      0 |       0 |             0 |         0 | sepia image of tiger                            | digitally altered | other       |
| ILSVRC2012_val_00041992.JPEG |     718 |                  0 |            0 |       0 |          0 |        0 |       0 |        0 |         0 |                 0 |                 0 |              0 |         0 |      1 |       0 |             0 |         0 | the bridge is brown                             | rare view         | device      |
| ILSVRC2012_val_00028056.JPEG |     635 |                  0 |            0 |       0 |          0 |        0 |       0 |        0 |         0 |                 0 |                 0 |              0 |         1 |      0 |       0 |             0 |         0 | the magnetic compass is on the bronze container | wood shape        | device      |

See this [notebook](notebooks/load_samples.ipynb) for some sample images and annotations

One can also directly download the raw annotation files stored in the `annotations` folder.
There are 4 json line files `imagenet_x_[train|val]_[multi|top]_factor.jsonl` that have entries such as the following:
```
{"file_name":"ILSVRC2012_val_00004487.JPEG","class":762,"multiple_objects":0,"background":1,"color":0,"brighter":0,"darker":0,"style":0,"larger":1,"smaller":0,"object_blocking":0,"person_blocking":0,"partial_view":0,"pattern":0,"pose":1,"shape":0,"subcategory":0,"texture":0,"justification":"close up of a pan fried sea bass. ","one_word":"sea bass close up"}
```
### To generate plots for a new model

Generate the predictions of your model on the Imagenet Validation set in a csv file with 3 columns 

| file_name                    |   predicted_class |   predicted_probability |
|:-----------------------------|------------------:|------------------------:|
| ILSVRC2012_val_00000293.JPEG |                 0 |                0.634764 |
| ILSVRC2012_val_00002138.JPEG |               391 |                0.360206 |
| ILSVRC2012_val_00003014.JPEG |                 0 |                0.951837 |
| ILSVRC2012_val_00006697.JPEG |                 0 |                0.999731 |
| ILSVRC2012_val_00007197.JPEG |                 0 |                0.998473 |

Then store the list of model CSVs in a directory `"path/to/model/predictions"`

```python
from imagenet_x import get_factor_accuracies, error_ratio

factor_accs = get_factor_accuracies("path/to/model/predictions")
error_ratios = error_ratio(factor_accs)
```

This gives the following tables

#### Factor accuracies

| model    |     pose |   background |   pattern |    color |   smaller |    shape |   partial_view |   subcategory |   texture |   larger |   darker |   object_blocking |   person_blocking |    style |   brighter |   multiple_objects |   worst_factor |   average |
|:---------|---------:|-------------:|----------:|---------:|----------:|---------:|---------------:|--------------:|----------:|---------:|---------:|------------------:|------------------:|---------:|-----------:|-------------------:|---------------:|----------:|
| DINO     | 0.821561 |     0.737577 |  0.772103 | 0.710569 |  0.62069  | 0.596465 |       0.722571 |      0.519658 |  0.471631 | 0.693333 | 0.639344 |          0.525641 |          0.5      | 0.581395 |   0.772727 |              0.65  |       0.471631 |  0.754283 |
| ResNet50 | 0.824018 |     0.710799 |  0.804588 | 0.698862 |  0.62069  | 0.558174 |       0.713166 |      0.466667 |  0.393617 | 0.666667 | 0.565574 |          0.512821 |          0.45     | 0.511628 |   0.704545 |              0.5   |       0.393617 |  0.746693 |
| SimCLR   | 0.739976 |     0.63494  |  0.693905 | 0.623902 |  0.482069 | 0.505155 |       0.659875 |      0.45641  |  0.308511 | 0.66     | 0.622951 |          0.410256 |          0.316667 | 0.604651 |   0.704545 |              0.6   |       0.308511 |  0.664064 |
| ViT      | 0.827868 |     0.746458 |  0.799565 | 0.732358 |  0.617931 | 0.642121 |       0.786834 |      0.531624 |  0.524823 | 0.7      | 0.57377  |          0.538462 |          0.55     | 0.627907 |   0.818182 |              0.625 |       0.524823 |  0.767599 |

#### Error ratios

| model    |     pose |   background |   pattern |   color |   smaller |   shape |   partial_view |   subcategory |   texture |   larger |   darker |   object_blocking |   person_blocking |   style |   brighter |   multiple_objects |
|:---------|---------:|-------------:|----------:|--------:|----------:|--------:|---------------:|--------------:|----------:|---------:|---------:|------------------:|------------------:|--------:|-----------:|-------------------:|
| DINO     | 0.726197 |      1.06799 |  0.927478 | 1.1779  |   1.54369 | 1.64228 |       1.12906  |       1.95486 |   2.15032 |  1.24805 |  1.46777 |           1.93051 |           2.03486 | 1.70361 |   0.924938 |            1.4244  |
| ResNet50 | 0.694739 |      1.1417  |  0.771442 | 1.18883 |   1.49743 | 1.74423 |       1.13236  |       2.10548 |   2.39386 |  1.31592 |  1.71502 |           1.92327 |           2.17128 | 1.92798 |   1.16639  |            1.97389 |
| SimCLR   | 0.774029 |      1.0867  |  0.911171 | 1.11955 |   1.54176 | 1.47304 |       1.01247  |       1.61814 |   2.0584  |  1.0121  |  1.12238 |           1.75552 |           2.03412 | 1.17686 |   0.879497 |            1.1907  |
| ViT      | 0.74067  |      1.09097 |  0.862456 | 1.15164 |   1.64401 | 1.53992 |       0.917235 |       2.01538 |   2.04465 |  1.29087 |  1.83403 |           1.98596 |           1.93631 | 1.60108 |   0.782348 |   

We also provide some plotting utilities

```python
from imagenet_x import plots

plots.model_comparison(factor_accs.reset_index(), fname="/path/to/save/fig.pdf|png")
```

Finally, we also provide a `ImagenetX` pytorch dataset that loads the imagenet samples and appends the factors of validation as a one hot encoded vector of 16 elements.

```python

from imagenet_x.evaluate import ImageNetX, get_vanilla_transform

# Declare dataset
imagenet_val_path = '/datasets01/imagenet_full_size/061417/val/'
transforms = get_vanilla_transform()
dataset = ImageNetX(imagenet_val_path, transform=transforms)
```

See this [notebook](notebooks/evaluate_model.ipynb) to run the previous commands and for a sample evaluation loop on a resnet-18.

## Paper results

### To reproduce plots for models in the paper

You need python>=3.8 for plots and evaluation to work

```bash
pip install imagenet-x[all]
python -m imagenet_x plots [--use-tex]
```

or if you cloned this repo

```bash
pip install -e .[all]
python -m imagenet_x plots [--use-tex]
```

### Generate aggregate results from model predictions

```bash
python -m imagenet_x aggregate --model-dirs path/to/model/predictions 
```