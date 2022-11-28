# The purpose of this file is to construct a dataset of interpolated cifar10
# to use for function frequency analysis via FFT of linear interpolation paths
# between validation images
import torch, torchvision
import torchvision.transforms as transforms
import numpy as np
import os, sys, json
from pathlib import Path
from models import instantiate_model
from utilities import args_to_model_string, get_args

FLAGS = get_args()

class InterpData:
    # If a dir is provided, reload the interpolation dataset from there
    # Otherwise, create an interpolation dataset using the provided data and labels
    def __init__(self, data=None, labels=None, dir=None, num_classes=10, m=200, delta=1, seed=0, batch_size=FLAGS.batch_size, logdir=None):
        # m is the number of paths for each pair of classes
        # delta is the sampling distance along each path
        self.num_classes = num_classes
        self.m = m
        self.delta = delta
        np.random.seed(seed)
        if data is None:
            self.reload(dir)
        else:
            print(f'Constructing the interpolation dataset')
            assert data is not None and labels is not None
            (self.n, self.h, self.w, self.c) = data.shape
            assert self.n % self.num_classes == 0  # Expect an equal number of examples per class

            # Sort by class label
            self.data = np.zeros((self.num_classes, self.n // self.num_classes, self.h, self.w, self.c))
            for i in range(self.num_classes):
                idx = (labels == i)
                self.data[i, ...] = data[idx, ...]
            
            self.setup_class_pairs()
            self.make_within_paths()
            self.make_between_paths()
            if dir is not None:
                self.save(dir)
        self.reset(batch_size, logdir)
    
    def make_within_paths(self):
        # Construct the within-class paths
        within_paths = []
        within_path_ids = []
        for i in range(self.num_classes):
            for j in range(self.m):
                x0 = self.data[i, self.within_pairs[i, j, 0], ...]
                x1 = self.data[i, self.within_pairs[i, j, 1], ...]
                distance = np.linalg.norm((x0 - x1).flatten())
                ts = np.arange(start=0, stop=distance, step=self.delta)
                path = x0[np.newaxis, ...] + ts[:, np.newaxis, np.newaxis, np.newaxis] * (x1[np.newaxis, ...] - x0[np.newaxis, ...]) / distance
                within_paths = within_paths + list(path)
                within_path_ids = within_path_ids + list(ts)
        self.within_paths = np.asarray(within_paths)
        self.within_path_ids = np.asarray(within_path_ids)
        print(f'within_paths has shape {self.within_paths.shape} with ids of shape {self.within_path_ids.shape}')

    def make_between_paths(self):
        # Construct the between_class paths
        between_paths = []
        between_path_ids = []
        for i in range(len(self.between_class_pairs)):
            for j in range(self.m):
                x0 = self.data[self.between_class_pairs[i, 0], self.between_pairs[i, j, 0], ...]
                x1 = self.data[self.between_class_pairs[i, 1], self.between_pairs[i, j, 1], ...]
                distance = np.linalg.norm((x0 - x1).flatten())
                ts = np.arange(start=0, stop=distance, step=self.delta)
                path = x0[np.newaxis, ...] + ts[:, np.newaxis, np.newaxis, np.newaxis] * (x1[np.newaxis, ...] - x0[np.newaxis, ...]) / distance
                between_paths = between_paths + list(path)
                between_path_ids = between_path_ids + list(ts)
        self.between_paths = np.asarray(between_paths)
        self.between_path_ids = np.asarray(between_path_ids)
        print(f'between_paths has shape {self.between_paths.shape} with ids of shape {self.between_path_ids.shape}')

    def setup_class_pairs(self):
        # Choose within-class pairs
        self.within_pairs = np.zeros((self.num_classes, self.m, 2), dtype=int)
        for i in range(self.num_classes):
            self.within_pairs[i, ...] = np.random.choice(a=self.n // self.num_classes, size=(self.m, 2), replace=False)

        # Choose between-class pairs
        class_pairs = []
        for i in range(self.num_classes):
            for j in range(1, self.num_classes - i):
                class_pairs.append([i, i + j])
        self.between_class_pairs = np.asarray(class_pairs)
        num_class_pairs = self.num_classes * (self.num_classes - 1) // 2  # num_classes choose 2
        assert len(self.between_class_pairs) == num_class_pairs
        self.between_pairs = np.zeros((num_class_pairs, self.m, 2), dtype=int)
        for i in range(num_class_pairs):
            # Technically these don't all need to be without replacement, but this should be ok
            self.between_pairs[i, ...] = np.random.choice(a=self.n // self.num_classes, size=(self.m, 2), replace=False) 

    def save(self, dir):
        os.makedirs(dir, exist_ok=True)
        np.save(os.path.join(dir, 'within_pairs.npy'), self.within_pairs)
        np.save(os.path.join(dir, 'between_class_pairs.npy'), self.between_class_pairs)
        np.save(os.path.join(dir, 'between_pairs.npy'), self.between_pairs)
        np.save(os.path.join(dir, 'within_paths.npy'), self.within_paths)
        np.save(os.path.join(dir, 'within_path_ids.npy'), self.within_path_ids)
        np.save(os.path.join(dir, 'between_paths.npy'), self.between_paths)
        np.save(os.path.join(dir, 'between_path_ids.npy'), self.between_path_ids)

    def reload(self, dir):
        print(f'Reloading the interpolation dataset from {dir}')
        self.within_pairs = np.load(os.path.join(dir, 'within_pairs.npy'))
        self.between_class_pairs = np.load(os.path.join(dir, 'between_class_pairs.npy'))
        self.between_pairs = np.load(os.path.join(dir, 'between_pairs.npy'))
        self.within_paths = np.load(os.path.join(dir, 'within_paths.npy'))
        self.within_path_ids = np.load(os.path.join(dir, 'within_path_ids.npy'))
        self.between_paths = np.load(os.path.join(dir, 'between_paths.npy'))
        self.between_path_ids = np.load(os.path.join(dir, 'between_path_ids.npy'))
        print(f'within_paths has shape {self.within_paths.shape} with ids of shape {self.within_path_ids.shape}')
        print(f'between_paths has shape {self.between_paths.shape} with ids of shape {self.between_path_ids.shape}')

    def reset(self, batch_size=None, logdir=None):
        self.idx = 0
        self.logdir = logdir
        if batch_size is not None:
            self.batch_size = batch_size
        if logdir is not None:
            os.makedirs(logdir, exist_ok=True)
            placeholder_within_preds = np.zeros((len(self.within_paths), self.num_classes))
            placeholder_between_preds = np.zeros((len(self.between_paths), self.num_classes))
            np.save(os.path.join(self.logdir, 'within_preds.npy'), placeholder_within_preds)
            np.save(os.path.join(self.logdir, 'between_preds.npy'), placeholder_between_preds)

    def next_batch(self):
        # Figure out if we are working within_class or between_class
        if self.idx < len(self.within_paths):
            last_idx = min(len(self.within_paths), self.idx + self.batch_size)
            batch = self.within_paths[self.idx:last_idx, ...]
        elif self.idx < len(self.within_paths) + len(self.between_paths):
            last_idx = min(len(self.between_paths) + len(self.within_paths), self.idx + self.batch_size)
            batch = self.between_paths[self.idx - len(self.within_paths):last_idx - len(self.within_paths), ...]
        else:
            return None, None, None
        start_idx = self.idx
        self.idx = last_idx
        return batch, start_idx, last_idx

    def apply_model_to_batch(self, model):
        batch, start_idx, stop_idx = self.next_batch()
        if batch is None:
            return None
        preds = model(torch.from_numpy(batch).permute((0,3,1,2)).float().cuda())  # [batch_size, num_classes]
        # Apply softmax
        soft = torch.nn.Softmax(dim=1)
        preds = soft(preds).detach().cpu().numpy()
        # Save the predictions to the logfiles
        if self.logdir is not None:
            if start_idx < len(self.within_paths):
                within_preds = np.load(os.path.join(self.logdir, 'within_preds.npy'))
                within_preds[start_idx:stop_idx, ...] = preds
                np.save(os.path.join(self.logdir, 'within_preds.npy'), within_preds)
            else:
                between_preds = np.load(os.path.join(self.logdir, 'between_preds.npy'))
                between_preds[start_idx - len(self.within_paths):stop_idx - len(self.within_paths), ...] = preds
                np.save(os.path.join(self.logdir, 'between_preds.npy'), between_preds)  
        return preds  

    def eval_model(self, model, FLAGS, base_logdir='/usr/workspace/freqml/logs'):
        strategy = ''
        if np.isin(FLAGS.pruning_approach, ['biprop', 'edgepopup']):
            strategy = FLAGS.sparsity_type
        self.logdir = os.path.join(base_logdir, f'{FLAGS.model_name}_{FLAGS.data_augmentation}_{FLAGS.pruning_approach}{FLAGS.sparsity}{strategy}')
        if self.verify_results():
            print('This model has already been evaluated')
            return
        self.reset(batch_size=FLAGS.batch_size, logdir=self.logdir)
        while(True):
            preds = self.apply_model_to_batch(model)
            if preds is None:
                break
        self.verify_results()            

    def verify_results(self):
        correct = True
        if not os.path.exists(os.path.join(self.logdir, 'within_preds.npy')):
            print(f'This model has not been evaluated yet')
            return False
        if not os.path.exists(os.path.join(self.logdir, 'between_preds.npy')):
            print(f'This model has not been evaluated yet')
            return False
        # Check that the model predictions are all (close to) valid probability distributions
        within_preds = np.load(os.path.join(self.logdir, 'within_preds.npy'))
        if np.amin(within_preds.flatten()) < 0: 
            correct = False
            print('Found a predicted within-class probability less than 0')
        within_sum = np.sum(within_preds, axis=1)
        if np.amin(within_sum) < 0.99: 
            correct = False
            print(f'Within-class probabilities sum to less than 1')
        if np.amax(within_sum) > 1.01: 
            correct = False
            print(f'Within-class probabilities sum to more than 1')
        within_std = np.std(within_preds, axis=1)
        if np.amin(within_std) < 1e-5:
            correct = False
            print(f'Found a perfectly uniform within-class probability, probably evaluation was stopped partway')
        between_preds = np.load(os.path.join(self.logdir, 'between_preds.npy'))
        if np.amin(between_preds.flatten()) < 0: 
            correct = False
            print('Found a predicted between-class probability less than 0')
        between_sum = np.sum(between_preds, axis=1)
        if np.amin(between_sum) < 0.99: 
            correct = False
            print(f'Between-class probabilities sum to less than 1')
        if np.amax(between_sum) > 1.01: 
            correct = False
            print(f'Between-class probabilities sum to more than 1')
        between_std = np.std(between_preds, axis=1)
        if np.amin(between_std) < 1e-5:
            correct = False
            print(f'Found a perfectly uniform between-class probability, probably evaluation was stopped partway')
        return correct

    def process_results(self, FLAGS, freq_thresh_frac=0.1, base_logdir='/usr/workspace/freqml/logs'):
        strategy = ''
        if np.isin(FLAGS.pruning_approach, ['biprop', 'edgepopup']):
            strategy = FLAGS.sparsity_type
        self.logdir = os.path.join(base_logdir, f'{FLAGS.model_name}_{FLAGS.data_augmentation}_{FLAGS.pruning_approach}{FLAGS.sparsity}{strategy}')
        if not self.verify_results():
            print('Error: Cannot process results without first evaluating model')
            assert False
        # Load the results
        within_preds = np.load(os.path.join(self.logdir, 'within_preds.npy'))
        between_preds = np.load(os.path.join(self.logdir, 'between_preds.npy'))
        # Separate out the paths
        within_paths = InterpData.separate_paths(within_preds, self.within_path_ids)
        between_paths = InterpData.separate_paths(between_preds, self.between_path_ids)
        assert len(within_paths) == self.num_classes * self.m
        assert len(between_paths) == len(self.between_class_pairs) * self.m
        # Process each path and average over the paths
        high_freq_fracs_within = []
        for path in within_paths:
            high_freq_fracs_within.append(self.process_path(path=path, freq_thresh_frac=freq_thresh_frac))
        high_freq_fracs_between = []
        for path in between_paths:
            high_freq_fracs_between.append(self.process_path(path=path, freq_thresh_frac=freq_thresh_frac))
        high_freq_fracs_within = np.asarray(high_freq_fracs_within)
        high_freq_fracs_between = np.asarray(high_freq_fracs_between)
        np.save(os.path.join(self.logdir, 'high_freq_fracs_within.npy'), high_freq_fracs_within)
        np.save(os.path.join(self.logdir, 'high_freq_fracs_between.npy'), high_freq_fracs_between)
        return high_freq_fracs_within, high_freq_fracs_between

    def process_path(self, path, freq_thresh_frac):
        # Compute the DFT magnitude for each class along the path
        fftmag = np.absolute(np.fft.rfft(path, axis=0))
        # Average the magnitudes over the classes
        fftmag = np.mean(fftmag, axis=1)
        # Compute the fraction above the threshold
        threshold = int(np.ceil(freq_thresh_frac * len(fftmag)))
        high_freq_frac = np.sum(fftmag[threshold:-1]) / np.sum(fftmag)
        return high_freq_frac

    @staticmethod
    def separate_paths(preds, ids):
        # preds should have shape [n, num_classes] and contain model predictions
        # ids should have shape [n] and contain delta values from interpolation
        assert len(preds) == len(ids)
        path_starts = np.where(ids == 0)[0]
        paths = []
        for idx in range(len(path_starts)):
            startpt = path_starts[idx]
            endpt = len(ids)
            if idx + 1 < len(path_starts):
                endpt = path_starts[idx + 1]
            path = preds[startpt:endpt, ...]
            paths.append(path)
        return paths

if __name__=='__main__':
    # Load and preprocess the test images before doing interpolation, unless we've already done so
    assert FLAGS.PATH_TO_interp_c10, 'you must specify a location for the interpolated data we will create using the arg --PATH_TO_interp_c10'
    assert FLAGS.PATH_TO_c10, 'you must specify a location for the CIFAR-10 data we will create using the arg --PATH_TO_c10'
    FLAGS.PATH_TO_interp_c10 = Path(FLAGS.PATH_TO_interp_c10)
    FLAGS.PATH_TO_c10 = Path(FLAGS.PATH_TO_c10)
    FLAGS.PATH_TO_RobustNets = Path(FLAGS.PATH_TO_RobustNets)
    data = None
    labels = None
    os.makedirs(FLAGS.PATH_TO_c10, exist_ok=True)
    os.makedirs(FLAGS.PATH_TO_interp_c10, exist_ok=True)
    paths = os.listdir(FLAGS.PATH_TO_interp_c10)
    if not all([x in paths for x in ['within_pairs.npy', 'between_class_pairs.npy',
            'between_pairs.npy', 'within_paths.npy', 'within_path_ids.npy',
            'between_paths.npy', 'between_path_ids.npy']]):
        cifar_mean = [0.491, 0.482, 0.447]
        cifar_std = [0.247, 0.243, 0.262]
        testset = torchvision.datasets.CIFAR10(root=FLAGS.PATH_TO_c10, 
                                                train=False,
                                                download=True,
                                                transform=None)
        data = testset.data  # [10000, 32, 32, 3]
        data = (data / 255. - cifar_mean) / cifar_std
        labels = np.asarray(testset.targets)  # [10000]

    # Prepare the interpolation dataset
    interp_data = InterpData(data=data, labels=labels, dir=FLAGS.PATH_TO_interp_c10)

    # Load the model
    model = instantiate_model(args_to_model_string(FLAGS), FLAGS.PATH_TO_RobustNets)
    model.eval()
    model.cuda()

    # Evaluate the model
    interp_data.eval_model(model=model, FLAGS=FLAGS)
    high_freq_fracs_within, high_freq_fracs_between = interp_data.process_results(FLAGS=FLAGS, freq_thresh_frac=0.1)
    print(f'avg_high_freq_frac_within is {np.mean(high_freq_fracs_within)} +- {np.std(high_freq_fracs_within)}')
    print(f'avg_high_freq_frac_between is {np.mean(high_freq_fracs_between)} +- {np.std(high_freq_fracs_between)}')


    with open('RobustNets/metric_and_OOD_var_dict.json', 'r') as f:
        metric_dict = json.load(f)
    print(f'avg_high_freq_frac_within in "Models Out of Line..." was {metric_dict[args_to_model_string(FLAGS)]["high_freq_fracs_within"]}')
    print(f'avg_high_freq_frac_between in "Models Out of Line..." was {metric_dict[args_to_model_string(FLAGS)]["high_freq_fracs_between"]}')
