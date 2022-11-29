# RobustNets
RobustNets benchmark models and code.

Code and model release for the NeurIPS 2022 paper:

[**Models Out of Line: A Fourier Lens on Distribution Shift Robustness**](https://arxiv.org/abs/2207.04075)

## Getting started
We recommend cloning this repository and running the data download script `download_RobustNets.sh`. Alternatively, you can manually download the RobustNets model state dicts here: https://github.com/sarafridov/RobustNets/releases.

After downloading the RobustNets dataset and code, running the program `RobustNets.py` will ensure you have downloaded the whole dataset and everything is working correctly. This program expects access to a GPU and takes about 10 minutes to run due to its computation of CIFAR-10 test accuracy for each model. You can skip the accuracy checks if they're too burdensome, but they do help confirm that we're working with the same data. Assuming you downloaded the RobustNets assets to the directory `RobustNets` and want to use the directory `tempC` to store `torchvision`'s CIFAR-10 data, enter the following command:

```
python RobustNets.py --PATH_TO_RobustNets=RobustNets --PATH_TO_c10=tempC
```

`RobustNets.py` contains the function `iterate_over_RobustNets`, which should give you all the information you need to understand how to iterate over identifiers for each model in the RobustNets dataset. This program also contains `check_RobustNets_c10_accuracy`, which illustrates how to load these models given their identifiers. In particular, you must use the function `instantiate_model`, which takes the model identifier and the path to RobustNets as arguments: `model = instantiate_model(model_string, PATH_TO_RobustNets)`.

## Computing metrics

All metrics applied in our paper to the RobustNets models are in the dictionary `RobustNets/metric_and_OOD_var_dict.json`, but you may want to recompute these metrics or compute them on other models. The following examples illustrate how to do this. To use a model that isn't the default model, you will have to specify that model via the args (see `get_args` in `utilities.py`). To use a model outside of the RobustNets dataset, you will have to modify the metric-computation programs to load your desired model rather than a RobustNets model. Finally, note that the interpolation programs create and save additional data at the specified `PATH_TO_interp`. 

Compute Fourier interpolation metrics:

```
python fourier_interpolation.py --PATH_TO_RobustNets=RobustNets --PATH_TO_interp=tempI --PATH_TO_c10=tempC
```

Compute pixel interpolation metrics:

```
python pixel_interpolation.py --PATH_TO_RobustNets=RobustNets --PATH_TO_interp=tempI --PATH_TO_c10=tempC
```

Compute Jacobian norm:

```
python jacobian_norm.py --PATH_TO_RobustNets=RobustNets --PATH_TO_c10=tempC
```

## Citation
```
@inproceedings{modelsoutofline,
      title={Models Out of Line: A Fourier Lens on Distribution Shift Robustness},
      author={Fridovich-Keil, Sara and Bartoldson, Brian R. and Diffenderfer, James and Kailkhura, Bhavya and Bremer, Peer-Timo},
      year={2022},
      booktitle={NeurIPS},
}
```
