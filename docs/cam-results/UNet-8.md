

```bash
python -m steps.train_cam --config-name=unet/train_cam \
  mid_ch=8 batch_size=256
```

```bash
python -m steps.make_cam --config-name=unet/make_cam \
  mid_ch=8 weights=$(pwd)/outputs/train-cam/2021-03-31/unet-8/weights/best_model.pt
```

```bash
python -m steps.eval_cam --config-name=unet/eval_cam \
  cam_out_dir=$(pwd)/outputs/make-cam/2021-03-31/unet-8/cam_outputs
```
```yaml
iou: [0.68303975 0.31897522 0.18771888 0.16542793 0.15144261 0.27548563
 0.46018724 0.34734514 0.40891009 0.10930123 0.37482989 0.24658109
 0.33678973 0.3473072  0.43210221 0.33907136 0.21905939 0.38336686
 0.21910488 0.3156682  0.27644053]	
miou: 0.3141978601375306
```

```bash
python -m steps.make_cam_crf --config-name=unet/make_cam_crf \
  cam_out_dir=$(pwd)/outputs/make-cam/2021-03-31/unet-8/cam_outputs
```

```bash
python -m steps.eval_png --config-name=unet/eval_png \
  output_dir=$(pwd)/outputs/make-cam-crf/2021-03-31/unet-8/crf_outputs
```
```yaml
iou: [0.72688534 0.40026172 0.21753479 0.17661744 0.1656173  0.32375911
 0.49227967 0.36825202 0.43594979 0.11172923 0.40575986 0.25452137
 0.32219972 0.38560349 0.47835779 0.35859012 0.23607359 0.46309185
 0.21495909 0.29985676 0.2930311 ]	
miou: 0.3395681494506045
```

```
Git commit d32138dc353ed363242654753405c7968c53ce14
```