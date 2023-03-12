
# Cellulus

A self-supervised learning method for spatial instance embedding applied to microscopy images for cell segmentation
 - a simple post-processing step cells can be segmented fully
unsupervised
 - support supervised learning by adding the generated (unsupervised) segmentation to your dataset

---

## Requirements and Setup

Install the required packages with conda
```
conda create --name autospem --file environment.yml
```

### External Datasets

Models were trained on cell segmentation datasets that are part of the [tissuenet dataset](https://datasets.deepcell.org/) and the [cell tracking challenge datasets](http://celltrackingchallenge.net/2d-datasets/)


## Train Spatial Instance Embedding Networks

![](.assets/autospem.webp)

```
python colocseg/train_ssl.py --shape 252 252 --in_channels 2 --out_channels 2 --dspath <path to tissuenet files> --initial_lr 4e-05 --output_shape 236 236 --positive_radius 10 --regularization 1e-05 --check_val_every_n_epoch 10 --limit_val_batches 256 --max_epochs 50 --temperature 10 --lr_milestones 20 30 --batch_size 8 --loader_workers 8 --gpu 1
```

## Download pretrained models


## Infer Mean and Std of Spatial Embeddings

```
python colocseg/infer_spatial_embeddings.py <path_to_model>/model.torch output.zarr spatial_embedding <path_to_tissuenet>/tissuenet_v1.0_test.npz 102 raw 2 32 transpose
```

## Infer Segmentation from Spatial Embedding

```
python colocseg/infer_pseudo_gt_from_mean_std.py output.zarr <path_to_tissuenet>/tissuenet_v1.0_test.npz spatial_embedding meanshift_segmentation 0 0.21
```
## Postprocess Embeddings (Shrinking Instances by Fixed Distance)

```
python scripts/postprocess.py output.zarr meanshift_segmentation
```



