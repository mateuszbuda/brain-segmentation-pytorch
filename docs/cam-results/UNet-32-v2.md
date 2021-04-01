

```bash
python -m steps.train_cam --config-name=unet/train_cam \
  mid_ch=32 batch_size=64 +early_stopping=50
```

```bash
python -m steps.make_cam --config-name=unet/make_cam \
  mid_ch=32 weights=$(pwd)/outputs/train-cam/2021-03-31/unet-32-v2/weights/best_model.pt
```

```bash
python -m steps.eval_cam --config-name=unet/eval_cam \
  cam_out_dir=$(pwd)/outputs/make-cam/2021-03-31/unet-32-v2/cam_outputs
```
```yaml
iou: [0.68583074 0.30775186 0.196904   0.1948477  0.17820888 0.29436888
 0.5314692  0.33577453 0.3680818  0.14117331 0.42156312 0.34941597
 0.35554949 0.33935344 0.40337183 0.31180429 0.22394363 0.34827347
 0.20325554 0.3135852  0.32586658]	
miou: 0.3252568311750386
```

```bash
python -m steps.make_cam_crf --config-name=unet/make_cam_crf \
  cam_out_dir=$(pwd)/outputs/make-cam/2021-03-31/unet-32-v2/cam_outputs
```

```bash
python -m steps.eval_png --config-name=unet/eval_png \
  output_dir=$(pwd)/outputs/make-cam-crf/2021-03-31/unet-32-v2/crf_outputs
```
```yaml
iou: [0.72821    0.38155628 0.22383577 0.21907859 0.18803969 0.32575893
        0.55976599 0.35711374 0.38326861 0.15752747 0.45859894 0.36586923
        0.34532034 0.36427113 0.4356635  0.31844752 0.244561   0.3924668
        0.19508603 0.30268196 0.37105457]	
miou: 0.34848457573070946
```

```
Git commit d32138dc353ed363242654753405c7968c53ce14
```