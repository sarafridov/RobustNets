# Confirm RobustNets dataset is set up correctly with various checks.
# Illustrate how to access RobustNets models (see `iterate_over_RobustNets`).
# Expected output: "All checks passed!"

from models import (Conv8, ResNet18, VGG16, cConv8, cResNet18, cVGG16,
                                instantiate_model)
from utilities import get_model_string, get_args
import torch
from torchvision import datasets, transforms
import json
from functools import partial
import os
from pathlib import Path
from tqdm import tqdm

def iterate_over_RobustNets(function_applied_to_each_model):
    """
    Iterates over all models in the RobustNets dataset, applying the
    `function_applied_to_each_model` to the unique identifier of
    each RobustNets model (`model_string`).

    As illustrated in `check_RobustNets_c10_accuracy`, `model_string`
    and the path to the RobustNets folder are required to instantiate
    a model (via `instantiate_model`).

    The following for loops and if conditions show the span of RobustNets.
    """
    model_names = ['Conv8', 'ResNet18', 'VGG16']
    pruning_approaches = ['biprop', 'edgepopup', 'GMP', 'FT', 'lrr', 'lth']
    sparsity_levels = [0.0, 0.5, 0.6, 0.8, 0.9, 0.95]
    sparsity_types = ['globally', 'layerwise']
    data_augmentations = ['augmix', 'gaussian', 'clean']

    for pruning_approach in tqdm(pruning_approaches):
        for model_name in model_names:
            for sparsity in sparsity_levels:
                for sparsity_type in sparsity_types:
                    for data_augmentation in data_augmentations:
                        if sparsity == 0:
                            if pruning_approach not in ['lrr']:
                                continue # we only have 1 model with 0 sparsity (i.e., 1 unpruned model)
                        if sparsity_type == 'layerwise':
                            if (pruning_approach in ['lrr', 'lth']) or (sparsity==0.95 and model_name=='Conv8'):
                                continue # 'lth' and 'lrr' pruning was always done globally; Conv8 layerwise 0.95 sparsity excluded

                        # define unique model string in terms of variable values
                        model_string = get_model_string(model_name, data_augmentation, pruning_approach, sparsity_type, sparsity)
                        function_applied_to_each_model(model_string)

def check_RobustNets_c10_accuracy(test_loader, PATH_TO_RobustNets, metric_and_OOD_var_dict, model_string):
    """
    For each CIFAR-10 model analyzed in "Models Out of Line...", load the model,
    then compute its test accuracy. The model loaded correctly if this test accuracy
    matches the accuracy we used in the paper, which was computed after training.
    """
    state_dict_name = model_string + '_state_dict.pt'
    # build model and load its state dict from the RobustNets location
    model = instantiate_model(model_string, PATH_TO_RobustNets)
    # confirm loaded model's accuracy matches accuracy found during training
    test_acc = compute_test_accuracy(test_loader, model)
    c10_acc = metric_and_OOD_var_dict[model_string]['cifar10_acc']
    acc_string = f'c10_acc was {c10_acc}, computed acc was {test_acc}'
    assert(test_acc == c10_acc), acc_string
    print(model_string + f' c10 acc matches precomputed acc ({c10_acc}%)')

def check_RobustNets_existence(PATH_TO_RobustNets, metric_and_OOD_var_dict):
    """
    Confirm 1) there's a model for each `metric_and_OOD_var_dict` key,
    2) there's a key for each model, and 3) the iterator is comprehensive.
    """
    # checks 1 and 2
    files = os.listdir(PATH_TO_RobustNets)
    count = 0
    expected_count = len(metric_and_OOD_var_dict)
    for f in files:
        if f[-len('_state_dict.pt'):] == '_state_dict.pt':
            assert f.replace('_state_dict.pt',
                '') in metric_and_OOD_var_dict, f'RobustNets model {f} not in dictionary keys.'
            count += 1
    if count != expected_count:
        print(f'Expected {expected_count} RobustNets models but found {count}. Your download may be incomplete.')
        # figure out which model is missing
        for key in metric_and_OOD_var_dict:
            assert os.path.exists(
                PATH_TO_RobustNets/key+'_state_dict.pt'), f'Model {key} not in RobustNets directory.'

    # check 3, is the iterator comprehensive?
    global iterator_count 
    iterator_count = 0
    def check_vals_in_iterator(model_string):
        global iterator_count
        if model_string in metric_and_OOD_var_dict:
            iterator_count+=1
        else:
            assert False, f'iterator created unexpected value {model_string}'
    iterate_over_RobustNets(check_vals_in_iterator)
    assert iterator_count == expected_count, f'Expected {expected_count} RobustNets models but iterated over {iterator_count}. The iterator may have been modified.'

def get_c10_test_loader(data_dir):
    normalize = transforms.Normalize(
                mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])
    c10_transforms = transforms.Compose([transforms.ToTensor(), normalize])             
    test_set = datasets.CIFAR10(root=data_dir, 
                                            train=False,
                                            download=True,
                                            transform=c10_transforms)
    return torch.utils.data.DataLoader(test_set, batch_size=400, num_workers=4, pin_memory=True)

def compute_test_accuracy(test_loader, model):
    """
    Compute CIFAR-10 test accuracy on GPU
    """
    model.cuda()
    model.eval()
    y_hats = torch.tensor([], dtype=torch.int64).cuda()
    y_s = torch.tensor([], dtype=torch.int64)
    with torch.no_grad():
        for x, y in test_loader:
            y_hat = model(x.cuda())
            y_hats = torch.cat((y_hats, y_hat.argmax(1)))
            y_s = torch.cat((y_s, y))
    return round( (y_hats.cpu() == y_s).sum().item() / len(y_s) * 100, 2)

if __name__=='__main__':
    args = get_args()
    PATH_TO_RobustNets = Path(args.PATH_TO_RobustNets)
    assert args.PATH_TO_c10, 'you must specify a location for the CIFAR-10 data we will create using the arg --PATH_TO_c10'
    PATH_TO_c10_data = Path(args.PATH_TO_c10)
    PATH_TO_metric_and_OOD_var_dict = 'RobustNets/metric_and_OOD_var_dict.json'
    with open(PATH_TO_metric_and_OOD_var_dict, 'r') as f:
        metric_and_OOD_var_dict = json.load(f)
    print('**********************************\nRunning RobustNets existence checks.')
    check_RobustNets_existence(PATH_TO_RobustNets, metric_and_OOD_var_dict)
    print('RobustNets existence checks passed.')
    print('**********************************\nRunning RobustNets accuracy checks.')
    iterate_over_RobustNets(
        partial(check_RobustNets_c10_accuracy, get_c10_test_loader(PATH_TO_c10_data),
        PATH_TO_RobustNets, metric_and_OOD_var_dict))
    print('RobustNets accuracy checks passed.')
    print('**********************************\nAll checks passed!')
