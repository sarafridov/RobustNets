from argparse import ArgumentParser
import torch

def sparse_to_dense(sparse_tuple):
    """
    converts a sparse representation of a weight tensor to a dense one
    """
    indices, weights, shape, dense_shape = sparse_tuple
    sparse_tensor = torch.sparse_coo_tensor(
        indices, weights, shape)
    return sparse_tensor.to_dense().reshape(dense_shape)

def save_sparse_state_dict(dense_state_dict_path, sparse_state_dict_path):
    """
    if sparsity > 0.5, then it's more memory efficient to 
    store weight tensors as sparse tensors.

    args
    ----
    dense_state_dict_path: path to a dense_state_dict
    sparse_state_dict_path: path used to save the created sparse_state_dict
    """
    dense_state_dict = torch.load(dense_state_dict_path)
    if 'state_dict' in dense_state_dict: # case when other keys exist
        dense_state_dict = dense_state_dict['state_dict']
    sparse_state_dict = {}
    sparsity = float(dense_state_dict_path.split('_')[-3])
    if sparsity > 0.5:
        for k,v in dense_state_dict.items():
            if 'weight' in k:
                # replace the key's value with a sparse tuple
                flat_v = v.reshape(-1) # allows us to store indices with 1 int
                nz = nonzero_indices = torch.nonzero(flat_v).T
                sparse_state_dict[k] = sparse_tuple = (
                        nz.int(), # indices,
                        flat_v[nz[0]], # nonzero weights,
                        flat_v.shape, v.shape) # sparse and dense shapes
                # make sure the sparse tuple can give us back the value
                dense_weight_tensor = sparse_to_dense(sparse_tuple)
                assert torch.isclose(dense_weight_tensor, v).min()
            else:
                # use the original value if it's not a weight tensor
                sparse_state_dict[k] = v
        torch.save(sparse_state_dict, sparse_state_dict_path)
    else:
        print('Saving original dictionary because sparsity <= 0.5')
        torch.save(dense_state_dict, sparse_state_dict_path)
        
def sparse_dict_to_dense_dict(sparse_state_dict_path, 
                              dense_state_dict_path=None):
    """
    build and return a dense_state_dict from a sparse_state_dict, OR
    run a check for equivalence between a dense_state_dict (built from a
    sparse_state_dict) and the original dense_state_dict.

    args
    -----
    sparse_state_dict_path: path to sparse_state_dict
    dense_state_dict_path: path to original dense_state_dict, optional
    """
    sparse_state_dict = torch.load(sparse_state_dict_path)
    dense_state_dict = {}
    if dense_state_dict_path is not None:
        dense_state_dict = torch.load(dense_state_dict_path)
    sparsity = float(sparse_state_dict_path._str.split('_')[-3])

    if sparsity > 0.5:
        if dense_state_dict:
            # a dense dict was provided, so just check that
            # it can be rebuilt from the sparse dict
            for (k,v), (d_k, d_v) in zip(sparse_state_dict.items(), 
                                        dense_state_dict.items()):
                if 'weight' in k:
                    assert torch.isclose(sparse_to_dense(v), d_v).min()
                else:
                    assert torch.isclose(v, d_v).min()
            print('All equality checks passed!')
        else:
            # rebuild the dense dict from the sparse dict
            for (k,v) in sparse_state_dict.items():
                if 'weight' in k:
                    dense_state_dict[k] = sparse_to_dense(v)
                else:
                    dense_state_dict[k] = v    
            return dense_state_dict
    else:
        return sparse_state_dict

def get_model_string(model_name, data_augmentation, pruning_approach, sparsity_type, sparsity):
    return f'{model_name}_{data_augmentation}_{pruning_approach}_{sparsity_type}_{sparsity}'

def args_to_model_string(args):
    return f'{args.model_name}_{args.data_augmentation}_{args.pruning_approach}_{args.sparsity_type}_{args.sparsity}'

def get_args():
    args = ArgumentParser()
    args.add_argument(
        '--PATH_TO_RobustNets',
        type=str,
        required=True,
        help='Path to location of RobustNets dataset'
    )
    args.add_argument(
        '--model_name',
        type=str,
        default='Conv8',
        help='Name of the model to study, can be Conv8, ResNet18, or VGG16'
    )
    args.add_argument(
        '--sparsity',
        type=float,
        default=0.0,
        help='Percentage of the model weights to prune, can be 0.0, 0.5, 0.6, 0.8, 0.9, or 0.95'
    )
    args.add_argument(
        '--sparsity_type',
        type=str,
        default='globally',
        help='globally or layerwise (latter is only for biprop, edgepopup, GMP, or FT).'
    )
    args.add_argument(
        '--data_augmentation',
        type=str,
        default='clean',
        help='Type of data augmentation: clean, augmix, or gaussian.'
    )
    args.add_argument(
        '--pruning_approach',
        type=str,
        default='lrr',
        help='Type of pruning to apply: lrr, lth, edgepopup, biprop, GMP, or FT.'
    )
    args.add_argument(
        '--batch_size',
        type=int,
        default=500,
        help='Batch size for interpolation codes.'
    )
    args.add_argument(
        '--PATH_TO_c10',
        default=None,
        help='Desired location for CIFAR-10 data.'
    )
    args.add_argument(
        '--PATH_TO_interp_c10',
        default=None,
        help='Desired location for interpolated datasets.'
    )

    parsed = args.parse_args()

    model_names = ['Conv8', 'ResNet18', 'VGG16']
    pruning_approaches = ['biprop', 'edgepopup', 'GMP', 'FT', 'lrr', 'lth']
    sparsity_levels = [0.0, 0.5, 0.6, 0.8, 0.9, 0.95]
    sparsity_types = ['globally', 'layerwise']
    data_augmentations = ['augmix', 'gaussian', 'clean']

    assert '.' in str(parsed.sparsity), f'got {parsed.sparsity} but need sparsity to be a float, e.g. 0.0 or 0.95'
    assert parsed.model_name in model_names, f'model_name must be one of {model_names}'
    assert parsed.pruning_approach in pruning_approaches, f'pruning_approach must be one of {pruning_approaches}'
    assert parsed.sparsity in sparsity_levels, f'sparsity must be one of {sparsity_levels}'
    assert parsed.sparsity_type in sparsity_types, f'sparsity_type must be one of {sparsity_types}'
    assert parsed.data_augmentation in data_augmentations, f'data_augmentation must be one of {data_augmentations}'

    return parsed
