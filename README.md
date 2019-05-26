# U-Net for brain segmentation in PyTorch

U-Net implementation in PyTorch for FLAIR abnormality segmentation in brain MRI based on a deep learning segmentation algorithm used in [Association of genomic subtypes of lower-grade gliomas with shape features automatically extracted by a deep learning algorithm](https://doi.org/10.1016/j.compbiomed.2019.05.002).

This repository is a PyTorch port of official MATLAB/Keras implementation [brain-segmentation](https://github.com/mateuszbuda/brain-segmentation).
Weights for trained models are provided and can be used for testing or fine-tuning on a different dataset.
If you use our code or weights, please consider citing:

```
@article{buda2019association,
  title={Association of genomic subtypes of lower-grade gliomas with shape features automatically extracted by a deep learning algorithm},
  author={Buda, Mateusz and Saha, Ashirbani and Mazurowski, Maciej A},
  journal={Computers in Biology and Medicine},
  volume={109},
  year={2019},
  publisher={Elsevier},
  doi={10.1016/j.compbiomed.2019.05.002}
}
```

## docker

```
docker build -t brainseg .
```
