# The purpose of this file is to construct a dataset of interpolated cifar10
# for function frequency analysis via FFT of Fourier amplitude and 
# phase interpolation paths between validation images
import torch, torchvision
import torchvision.transforms as transforms
import numpy as np
import os, sys, json
from pathlib import Path
from models import instantiate_model
from utilities import args_to_model_string, get_args

FLAGS = get_args()

class FourierInterpData:
    # If a dir is provided, reload the interpolation dataset from there
    # Otherwise, create an interpolation dataset using the provided data and labels
    def __init__(self, data=None, labels=None, dirname=None, num_classes=10, m=5000, delta=1, seed=0, batch_size=FLAGS.batch_size, logdir=None):
        # m is the number of paths for each pair of classes
        # delta is the sampling distance along each path
        self.num_classes = num_classes
        self.m = m
        self.delta = delta
        np.random.seed(seed)
        if data is None:
            self.reload(dirname)
        else:
            print(f'Constructing the interpolation dataset')
            assert data is not None and labels is not None
            (self.n, self.h, self.w, self.c) = data.shape

            # Create m random pairs of images, to use for both types of interpolation
            self.ids = np.random.choice(self.n, size=(self.m, 2), replace=False)
            self.data = np.zeros((self.m, 2, self.h, self.w, self.c))
            for i in range(self.m):
                for j in range(2):
                    self.data[i,j,...] = data[self.ids[i,j], ...]
            
            self.make_amplitude_paths()
            self.make_phase_paths()
            if dirname is not None:
                self.save(dirname)
        self.reset(batch_size, logdir)
    
    def make_amplitude_paths(self):
        # Construct the varying-amplitude paths
        within_paths = []
        within_path_ids = []
        
        for i in range(self.m):
            x0 = self.data[i, 0, ...]
            x1 = self.data[i, 1, ...]
            ts = list(reversed(np.linspace(start=0, stop=1, num=100, endpoint=True)))
            # L is how much source, (1-L) is how much target, so we reversed ts to make path[0] all src and path[-1] all trg
            path = [(FourierInterpData.amplitude_interpolation(x0.T, x1.T, 0.2, L=t)).T for t in ts] 
            within_paths = within_paths + path
            within_path_ids = within_path_ids + ts
        self.within_paths = np.asarray(within_paths)
        self.within_path_ids = np.asarray(within_path_ids)
        print(f'within_paths has shape {self.within_paths.shape} with ids of shape {self.within_path_ids.shape}')

    def make_phase_paths(self):
        # Construct the varying-phase paths
        between_paths = []
        between_path_ids = []
        for i in range(self.m):
            x0 = self.data[i, 0, ...]
            x1 = self.data[i, 1, ...]
            ts = list(reversed(np.linspace(start=0, stop=1, num=100, endpoint=True)))
            # L is how much source, (1-L) is how much target, so we reversed ts to make path[0] all src and path[-1] all trg
            path = [(FourierInterpData.phase_interpolation(x0.T, x1.T, 0.2, L=t)).T for t in ts] 
            between_paths = between_paths + path
            between_path_ids = between_path_ids + ts
        self.between_paths = np.asarray(between_paths)
        self.between_path_ids = np.asarray(between_path_ids)
        print(f'between_paths has shape {self.between_paths.shape} with ids of shape {self.between_path_ids.shape}')
    
    @staticmethod
    def low_freq_mutate( amp_src, amp_trg, window_size=0.1, L = 0.1):
        a_src = np.fft.fftshift( amp_src, axes=(-2, -1) )
        a_trg = np.fft.fftshift( amp_trg, axes=(-2, -1) )

        _, h, w = a_src.shape
        b = (np.floor(np.amin((h,w))*window_size)).astype(int)
        c_h = np.floor(h/2.0).astype(int)
        c_w = np.floor(w/2.0).astype(int)

        h1 = c_h-b
        h2 = c_h+b+1
        w1 = c_w-b
        w2 = c_w+b+1

        a_src[:,h1:h2,w1:w2] = L*a_src[:,h1:h2,w1:w2] + (1.-L)*a_trg[:,h1:h2,w1:w2]
        a_src = np.fft.ifftshift( a_src, axes=(-2, -1) )
        return a_src
        
    @staticmethod
    def amplitude_interpolation( src_img, trg_img, window_size=0.1, L=0.1 ):
        # exchange magnitude
        # input: src_img, trg_img

        src_img_np = src_img
        trg_img_np = trg_img

        # get fft of both source and target
        fft_src_np = np.fft.fft2( src_img_np, axes=(-2, -1) )
        fft_trg_np = np.fft.fft2( trg_img_np, axes=(-2, -1) )

        # extract amplitude and phase of both ffts
        amp_src, pha_src = np.abs(fft_src_np), np.angle(fft_src_np)
        amp_trg, pha_trg = np.abs(fft_trg_np), np.angle(fft_trg_np)

        # mutate the amplitude part of source with target
        #amp_src_ = L*amp_src + (1-L)*amp_trg
        amp_src_ = FourierInterpData.low_freq_mutate( amp_src, amp_trg, window_size, L)  # window_size=0.5 will interpolate all frequencies

        # mutated fft of source
        fft_src_ = amp_src_ * np.exp( 1j * pha_src )

        # get the mutated image
        src_in_trg = np.fft.ifft2( fft_src_, axes=(-2, -1) )
        src_in_trg = np.real(src_in_trg)

        return src_in_trg

    @staticmethod
    def phase_interpolation( src_img, trg_img, window_size=0.1, L=0.1 ):
        # exchange magnitude
        # input: src_img, trg_img

        src_img_np = src_img
        trg_img_np = trg_img

        # get fft of both source and target
        fft_src_np = np.fft.fft2( src_img_np, axes=(-2, -1) )
        fft_trg_np = np.fft.fft2( trg_img_np, axes=(-2, -1) )

        # extract amplitude and phase of both ffts
        amp_src, pha_src = np.abs(fft_src_np), np.angle(fft_src_np)
        amp_trg, pha_trg = np.abs(fft_trg_np), np.angle(fft_trg_np)

        # mutate the phase part of source with target
        #amp_src_ = L*amp_src + (1-L)*amp_trg
        pha_src_ = FourierInterpData.low_freq_mutate( pha_src, pha_trg, window_size, L)  # window_size=0.5 will interpolate all frequencies

        # mutated fft of source
        fft_src_ = amp_src * np.exp( 1j * pha_src_ )

        # get the mutated image
        src_in_trg = np.fft.ifft2( fft_src_, axes=(-2, -1) )
        src_in_trg = np.real(src_in_trg)

        return src_in_trg

    def save(self, dir):
        os.makedirs(dir, exist_ok=True)
        np.save(os.path.join(dir, 'amp_paths.npy'), self.within_paths)
        np.save(os.path.join(dir, 'amp_path_ids.npy'), self.within_path_ids)
        np.save(os.path.join(dir, 'phase_paths.npy'), self.between_paths)
        np.save(os.path.join(dir, 'phase_path_ids.npy'), self.between_path_ids)

    def reload(self, dir):
        print(f'Reloading the interpolation dataset from {dir}')
        self.within_paths = np.load(os.path.join(dir, 'amp_paths.npy'))
        self.within_path_ids = np.load(os.path.join(dir, 'amp_path_ids.npy'))
        self.between_paths = np.load(os.path.join(dir, 'phase_paths.npy'))
        self.between_path_ids = np.load(os.path.join(dir, 'phase_path_ids.npy'))
        print(f'within_paths has shape {self.within_paths.shape} with ids of shape {self.within_path_ids.shape}')
        print(f'between_paths has shape {self.between_paths.shape} with ids of shape {self.between_path_ids.shape}')

    def reset(self, batch_size=None, logdir=None, idx=0):
        self.idx = idx
        self.logdir = logdir
        if batch_size is not None:
            self.batch_size = batch_size
        if logdir is not None:
            os.makedirs(logdir, exist_ok=True)
            placeholder_within_preds = np.zeros((len(self.within_paths), self.num_classes))
            placeholder_between_preds = np.zeros((len(self.between_paths), self.num_classes))
            # don't overwrite existing files
            if not os.path.exists(os.path.join(self.logdir, 'amp_preds.npy')):
                np.save(os.path.join(self.logdir, 'amp_preds.npy'), placeholder_within_preds)
            if not os.path.exists(os.path.join(self.logdir, 'phase_preds.npy')):
                np.save(os.path.join(self.logdir, 'phase_preds.npy'), placeholder_between_preds)

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
        preds = model(torch.from_numpy(batch).permute((0,3,1,2)).float().to("cuda"))  # [batch_size, num_classes]
        # Apply softmax
        soft = torch.nn.Softmax(dim=1)
        preds = soft(preds).detach().cpu().numpy()
        # Save the predictions to the logfiles
        if self.logdir is not None:
            if start_idx < len(self.within_paths):
                within_preds = np.load(os.path.join(self.logdir, 'amp_preds.npy'))
                within_preds[start_idx:stop_idx, ...] = preds
                np.save(os.path.join(self.logdir, 'amp_preds.npy'), within_preds)
            else:
                between_preds = np.load(os.path.join(self.logdir, 'phase_preds.npy'))
                between_preds[start_idx - len(self.within_paths):stop_idx - len(self.within_paths), ...] = preds
                np.save(os.path.join(self.logdir, 'phase_preds.npy'), between_preds)  
        return preds  

    def eval_model(self, model, FLAGS, base_logdir='/usr/workspace/freqml/logs'):
        strategy = ''
        if np.isin(FLAGS.pruning_approach, ['biprop', 'edgepopup', 'GMP', 'FT']):
            strategy = FLAGS.sparsity_type
        self.logdir = os.path.join(base_logdir, f'{FLAGS.model_name}_{FLAGS.data_augmentation}_{FLAGS.pruning_approach}{FLAGS.sparsity}{strategy}')
        status, idx = self.verify_results()
        if status:
            print('This model has already been evaluated')
            return
        self.reset(batch_size=FLAGS.batch_size, logdir=self.logdir, idx=idx)
        while(True):
            preds = self.apply_model_to_batch(model)
            if preds is None:
                break
        status, idx = self.verify_results() 
        assert status

    def verify_results(self):
        correct = True
        idx = None
        if not os.path.exists(os.path.join(self.logdir, 'amp_preds.npy')):
            print(f'This model has not been evaluated yet')
            return False, 0
        if not os.path.exists(os.path.join(self.logdir, 'phase_preds.npy')):
            print(f'This model has not been evaluated yet')
            return False, 0
        # Check that the model predictions are all (close to) valid probability distributions
        within_preds = np.load(os.path.join(self.logdir, 'amp_preds.npy'))
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
            idx = np.where(within_std < 1e-5)[0][0]  # Resume partial evaluation
            print(f'Found a perfectly uniform within-class probability, probably evaluation was stopped partway')
            print(f'Resuming at index {idx}', flush=True)
        between_preds = np.load(os.path.join(self.logdir, 'phase_preds.npy'))
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
            if idx == None:
                idx = np.where(between_std < 1e-5)[0][0] + len(within_std)  # Resume partial evaluation
            print(f'Found a perfectly uniform between-class probability, probably evaluation was stopped partway')
            print(f'Resuming at index {idx}', flush=True)
        if idx == None:
            idx = 0
            print(f'Resuming at index {idx}', flush=True)
        return correct, idx

    def process_results(self, FLAGS, freq_thresh_frac=0.1, base_logdir='/usr/workspace/freqml/logs'):
        strategy = ''
        if np.isin(FLAGS.pruning_approach, ['biprop', 'edgepopup', 'GMP', 'FT']):
            strategy = FLAGS.sparsity_type
        self.logdir = os.path.join(base_logdir, f'{FLAGS.model_name}_{FLAGS.data_augmentation}_{FLAGS.pruning_approach}{FLAGS.sparsity}{strategy}')
        status, idx = self.verify_results()
        if not status:
            print('Error: Cannot process results without first evaluating model')
            assert False
        # Load the results
        within_preds = np.load(os.path.join(self.logdir, 'amp_preds.npy'))
        between_preds = np.load(os.path.join(self.logdir, 'phase_preds.npy'))
        # Separate out the paths
        within_paths = FourierInterpData.separate_paths(within_preds, self.within_path_ids)
        between_paths = FourierInterpData.separate_paths(between_preds, self.between_path_ids)
        assert len(within_paths) == self.m
        assert len(between_paths) == self.m
        # Process each path and average over the paths
        # also count distance before top prediction changes along each path
        high_freq_fracs_within = []
        consistent_distances_amp = []
        for path in within_paths:
            consistent_distances_amp.append(self.measure_path_consistency(path=path))
            high_freq_fracs_within.append(self.process_path(path=path, freq_thresh_frac=freq_thresh_frac))
        high_freq_fracs_between = []
        consistent_distances_phase = []
        for path in between_paths:
            consistent_distances_phase.append(self.measure_path_consistency(path=path))
            high_freq_fracs_between.append(self.process_path(path=path, freq_thresh_frac=freq_thresh_frac))
        high_freq_fracs_within = np.asarray(high_freq_fracs_within)
        high_freq_fracs_between = np.asarray(high_freq_fracs_between)
        consistent_distances_amp = np.asarray(consistent_distances_amp)
        consistent_distances_phase = np.asarray(consistent_distances_phase)
        np.save(os.path.join(self.logdir, 'high_freq_fracs_amp.npy'), high_freq_fracs_within)
        np.save(os.path.join(self.logdir, 'high_freq_fracs_phase.npy'), high_freq_fracs_between)
        np.save(os.path.join(self.logdir, 'consistent_distances_amp.npy'), consistent_distances_amp)
        np.save(os.path.join(self.logdir, 'consistent_distances_phase.npy'), consistent_distances_phase)
        return high_freq_fracs_within, high_freq_fracs_between, consistent_distances_amp, consistent_distances_phase

    def process_path(self, path, freq_thresh_frac):
        # Compute the DFT magnitude for each class along the path
        fftmag = np.absolute(np.fft.rfft(path, axis=0))
        # Average the magnitudes over the classes
        fftmag = np.mean(fftmag, axis=1)
        # Compute the fraction above the threshold
        threshold = int(np.ceil(freq_thresh_frac * len(fftmag)))
        high_freq_frac = np.sum(fftmag[threshold:-1]) / np.sum(fftmag)
        return high_freq_frac
    
    # Count how many images along the path starting at the original image produce the same top prediction
    def measure_path_consistency(self, path):
        predictions = np.argmax(path, axis=1)
        deviations = np.where(predictions != predictions[0])
        result = len(predictions)
        if len(deviations[0]) > 0:
            result = deviations[0][0]
        return result

    @staticmethod
    def separate_paths(preds, ids):
        # preds should have shape [n, num_classes] and contain model predictions
        # ids should have shape [n] and contain delta values from interpolation
        assert len(preds) == len(ids)
        path_starts = np.where(ids == 1)[0]  # Note: this is different than for pixel interpolation
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
    if not ('amp_paths.npy' in paths and 'amp_path_ids.npy' in paths
            and 'phase_paths.npy' in paths and 'phase_path_ids.npy' in paths):
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
    interp_data = FourierInterpData(data=data, labels=labels, dirname=FLAGS.PATH_TO_interp_c10)

    # Load the model
    model = instantiate_model(args_to_model_string(FLAGS), FLAGS.PATH_TO_RobustNets)
    model.eval()
    model.cuda()

    # Evaluate the model
    interp_data.eval_model(model=model, FLAGS=FLAGS)
    high_freq_fracs_within, high_freq_fracs_between, consistent_distances_amp, consistent_distances_phase = interp_data.process_results(FLAGS=FLAGS, freq_thresh_frac=0.1)
    print(f'avg_high_freq_frac_amp is {np.mean(high_freq_fracs_within)} +- {np.std(high_freq_fracs_within)}')
    print(f'avg_high_freq_frac_phase is {np.mean(high_freq_fracs_between)} +- {np.std(high_freq_fracs_between)}')
    print(f'avg_consistent_dist_amp is {np.mean(consistent_distances_amp)} += {np.std(consistent_distances_amp)}')
    print(f'avg_consistent_dist_phase is {np.mean(consistent_distances_phase)} += {np.std(consistent_distances_phase)}')

    with open('RobustNets/metric_and_OOD_var_dict.json', 'r') as f:
        metric_dict = json.load(f)
    print(f'avg_high_freq_frac_amp in "Models Out of Line..." was {metric_dict[args_to_model_string(FLAGS)]["high_freq_fracs_amp"]}')
    print(f'avg_high_freq_frac_phase in "Models Out of Line..." was {metric_dict[args_to_model_string(FLAGS)]["high_freq_fracs_phase"]}')
    print(f'avg_consistent_dist_amp in "Models Out of Line..." was {metric_dict[args_to_model_string(FLAGS)]["consistent_distances_amp"]}')
    print(f'avg_consistent_dist_phase in "Models Out of Line..." was {metric_dict[args_to_model_string(FLAGS)]["consistent_distances_phase"]}')
