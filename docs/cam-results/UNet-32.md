

```bash
python -m steps.train_cam --config-name=unet/train_cam \
  mid_ch=32 batch_size=64
```

```bash
python -m steps.make_cam --config-name=unet/make_cam \
  mid_ch=32 weights=$(pwd)/outputs/train-cam/2021-03-31/unet-32/weights/best_model.pt
```

```bash
python -m steps.eval_cam --config-name=unet/eval_cam \
  cam_out_dir=$(pwd)/outputs/make-cam/2021-03-31/unet-32/cam_outputs
```
```yaml
iou: [0.68691831 0.27402306 0.20144877 0.14512615 0.14015046 0.21435765
 0.48758067 0.33060065 0.39320863 0.12342919 0.48214674 0.25162936
 0.33992057 0.37781778 0.42598039 0.3522709  0.21719984 0.3859505
 0.21999853 0.37128902 0.27790673]	
miou: 0.3189978044448517
```

```bash
python -m steps.make_cam_crf --config-name=unet/make_cam_crf \
  cam_out_dir=$(pwd)/outputs/make-cam/2021-03-31/unet-32/cam_outputs
```

```bash
python -m steps.eval_png --config-name=unet/eval_png \
  output_dir=$(pwd)/outputs/make-cam-crf/2021-03-31/unet-32/crf_outputs
```
```yaml
iou: [0.72682985 0.33272382 0.21915772 0.15137177 0.13759223 0.2273586
 0.5221365  0.34799951 0.42814036 0.12595025 0.53906035 0.25376448
 0.32974457 0.41063108 0.44456043 0.3698362  0.22779434 0.45601662
 0.20095949 0.39011597 0.28679066]	
miou: 0.3394540385199172
```

```
Git commit d32138dc353ed363242654753405c7968c53ce14
```