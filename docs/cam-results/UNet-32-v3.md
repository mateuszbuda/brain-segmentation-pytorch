

```bash
python -m steps.train_cam --config-name=unet/train_cam \
  mid_ch=32 batch_size=64 +early_stopping=100
```

```bash
python -m steps.make_cam --config-name=unet/make_cam \
  mid_ch=32 weights=$(pwd)/outputs/train-cam/2021-03-31/unet-32-v3/weights/best_model.pt
```

```bash
python -m steps.eval_cam --config-name=unet/eval_cam \
  cam_out_dir=$(pwd)/outputs/make-cam/2021-03-31/unet-32-v3/cam_outputs
```
```yaml
iou: [0.69467797 0.30632112 0.19252105 0.15563878 0.20198752 0.25927783
 0.42663088 0.35449893 0.42020063 0.13979498 0.44013049 0.32796027
 0.34652618 0.40487731 0.4606483  0.3470291  0.21383493 0.36804811
 0.23057803 0.29608008 0.31825723]	
miou: 0.3288342718342704
```

```bash
python -m steps.make_cam_crf --config-name=unet/make_cam_crf \
  cam_out_dir=$(pwd)/outputs/make-cam/2021-03-31/unet-32-v3/cam_outputs
```

```bash
python -m steps.eval_png --config-name=unet/eval_png \
  output_dir=$(pwd)/outputs/make-cam-crf/2021-03-31/17-51-46/crf_outputs
```
```yaml
iou: [0.73704118 0.35873461 0.22461205 0.16334701 0.2166576  0.27811738
 0.44573307 0.38280006 0.45322134 0.15101187 0.47917684 0.33832336
 0.33045383 0.47857348 0.48836582 0.36538224 0.23773346 0.41360092
 0.22955254 0.29790048 0.34747   ]	
miou: 0.35322900659370543
```

```
Git commit d32138dc353ed363242654753405c7968c53ce14
```