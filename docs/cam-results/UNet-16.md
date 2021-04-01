

```bash
python -m steps.train_cam --config-name=unet/train_cam \
  mid_ch=16 batch_size=128
```

```bash
python -m steps.make_cam --config-name=unet/make_cam \
  mid_ch=16 weights=$(pwd)/outputs/train-cam/2021-03-31/unet-16/weights/best_model.pt
```

```bash
python -m steps.eval_cam --config-name=unet/eval_cam \
  cam_out_dir=$(pwd)/outputs/make-cam/2021-03-31/unet-16/cam_outputs
```
```yaml
iou: [0.68429404 0.32592865 0.19045575 0.15442053 0.20517302 0.21143245
 0.47748155 0.33492868 0.40411381 0.1285854  0.40908516 0.30737528
 0.38004162 0.31947931 0.4367879  0.33964815 0.23210624 0.42181571
 0.25678082 0.3479898  0.26252121]	
miou: 0.325259290469362
```

```bash
python -m steps.make_cam_crf --config-name=unet/make_cam_crf \
  cam_out_dir=$(pwd)/outputs/make-cam/2021-03-31/unet-16/cam_outputs
```

```bash
python -m steps.eval_png --config-name=unet/eval_png \
  output_dir=$(pwd)/outputs/make-cam-crf/2021-03-31/unet-16/crf_outputs
```
```yaml
iou: [0.72834173 0.41004248 0.20835724 0.16383911 0.21253757 0.22278407
 0.5066628  0.3593036  0.43900482 0.1348789  0.43925203 0.31469249
 0.38853366 0.33406148 0.48095362 0.3725973  0.25283987 0.52823357
 0.26310573 0.3600513  0.28296201]	
miou: 0.35252549586473175
```

```
Git commit d32138dc353ed363242654753405c7968c53ce14
```