

```bash
python -m steps.train_cam --config-name=unet/train_cam \
  mid_ch=64 batch_size=32
```

```bash
python -m steps.make_cam --config-name=unet/make_cam \
  mid_ch=64 weights=$(pwd)/outputs/train-cam/2021-03-31/unet-64/weights/best_model.pt
```

```bash
python -m steps.eval_cam --config-name=unet/eval_cam \
  cam_out_dir=$(pwd)/outputs/make-cam/2021-03-31/unet-64/cam_outputs
```
```yaml
iou: [0.69296721 0.34577913 0.20822271 0.16812929 0.14744637 0.20160285
 0.45423467 0.36993881 0.39785367 0.11277776 0.39316021 0.25512225
 0.34086968 0.37626193 0.48720295 0.32588718 0.20298203 0.40676824
 0.25535988 0.24394574 0.26214415]	
miou: 0.31660269949340386
```

```bash
python -m steps.make_cam_crf --config-name=unet/make_cam_crf \
  cam_out_dir=$(pwd)/outputs/make-cam/2021-03-31/unet-64/cam_outputs
```

```bash
python -m steps.eval_png --config-name=unet/eval_png \
  output_dir=$(pwd)/outputs/make-cam-crf/2021-03-31/unet-64/crf_outputs
```
```yaml
iou: [0.73125911 0.41876454 0.23992761 0.18202355 0.15131658 0.19322785
 0.48739503 0.39325456 0.43759704 0.11176381 0.43777432 0.25594296
 0.31287327 0.42944977 0.52356261 0.32773961 0.21962071 0.47263511
 0.27188519 0.22609453 0.27710792]	
miou: 0.3381531281224007
```

```
Git commit d32138dc353ed363242654753405c7968c53ce14
```