import argparse
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from glob import glob

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#Set up log file
log_file = os.path.join(os.path.dirname(__file__), 'mmd.log')
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def rbf_numerator(X,Y):
    return X@X.T + Y@Y.T - 2*X@Y.T


# def rbf_kernel(X, Y, sigma=1.0):
def mmd_rbf(X, Y, sigma=1.0):
    # Compute Gram matrices for the same-set terms.

    m = X.shape[0]
    n = Y.shape[0]

    XX = torch.matmul(X, X.T)
    YY = torch.matmul(Y, Y.T)
    
    # Correct computation of pairwise squared distances for same sets.
    # Since we also need to compute XX and YY, we can use the diagonal.
    r2_xx = (XX.diag().unsqueeze(0) + XX.diag().unsqueeze(1) - 2 * XX) # XX.diag().unsqueeze(0) + XX.diag().unsqueeze(1)  this creates all pairwise sums of xi,xj
    r2_yy = (YY.diag().unsqueeze(0) + YY.diag().unsqueeze(1) - 2 * YY)
    
    # For the cross term, compute squared norms explicitly.
    X_norm = (X ** 2).sum(dim=1).unsqueeze(1)  # (n, 1)
    Y_norm = (Y ** 2).sum(dim=1).unsqueeze(0)  # (1, m)
    r2_xy = X_norm + Y_norm - 2 * X@Y.T
    
    # Compute the kernel matrices.
    k_xx = torch.exp(-r2_xx / (2 * sigma**2)).fill_diagonal_(0)
    k_yy = torch.exp(-r2_yy / (2 * sigma**2)).fill_diagonal_(0)
    k_xy = torch.exp(-r2_xy / (2 * sigma**2))
    
    return k_xx.sum()/(m*(m-1)) + k_yy.sum()/(n*(n-1)) - 2*k_xy.sum()/(m*n)


def load_h5_as_array(file_path,key = 'logits_df',retries=5, wait=5):
    # Open the file in read mode and load the desired dataset.
    for attempt in range(retries):
        try:
            with h5py.File(file_path, 'r',locking=False) as f:
                # Assuming data is stored in the first dataset
                data = f[key]['block0_values'][:]
            return data
        except OSError as e:
            logger.info(f"Attempt {attempt+1}/{retries} failed on {file_path}: {e}")
            time.sleep(wait)
    raise RuntimeError(f"Failed after {retries} retries for {file_path}")


def get_h5_shape(file_path,key = 'logits_df'):
    # Open the file in read mode and load the desired dataset.
    
    with h5py.File(file_path, 'r') as f:
        # Assuming data is stored in the first dataset
        return f[key]['block0_values'].shape


def multiple_h5s_to_tensor(file_paths,max_workers=8):
    # Create an empty list to store the data
    # Use ThreadPoolExecutor to read files concurrently
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        arrays = list(executor.map(load_h5_as_array, file_paths))

    return torch.from_numpy(np.concatenate(arrays, axis=0))


class ProteinLabelDistributions(Dataset):
    def __init__(self, dataset_name: str, h5_paths: list, cache: bool = True, max_workers: int = 8):

        files_dir = os.path.dirname(h5_paths[0])
        pt_path = f'{files_dir}/{dataset_name}.pt'
        
        logger.info(pt_path)
        if not os.path.exists(pt_path) or not cache:
            logger.info("Reading from HDF5 files...")
            self.data = multiple_h5s_to_tensor(h5_paths, max_workers=max_workers)

            logger.info("Saving to PT file...")
            torch.save(self.data, pt_path)
        else:
            logger.info("Loading from PT file...")
            self.data = torch.load(pt_path)
        

    def __len__(self):
        return len(self.h5_paths)

    def __getitem__(self, idx):
        return torch.sigmoid(self.data[idx])


class H5BatchedDataset(Dataset):
    def __init__(self, h5_paths: list, transform=None):
        self.h5_paths = h5_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.h5_paths)

    def __getitem__(self, idx):
        
        output = torch.from_numpy(load_h5_as_array(self.h5_paths[idx]))

        if self.transform:
            output = self.transform(output)
        return output
        

class RunningMeanVariance:
    """
    An implementation of Welford's algorithm for computing running mean and variance.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0
        self.mean = 0
        self.M2 = 0
        self.sample_variance = 0
        
    def __call__(self,new_value):
        self.count += 1

        delta = new_value - self.mean
        self.mean += delta / self.count

        delta2 = new_value - self.mean
        self.M2 += delta * delta2

    def compute(self):
        if self.count < 2:
            return float("nan")
        else:
            (self.mean, self.sample_variance) = (self.mean, self.M2 / (self.count - 1))
            return (self.mean, self.sample_variance)


def get_sigma_median(dl_Y, dl_X, num_iterations,subbatch_size,device,debug=False):

    dl_Y_iter, dl_X_iter = subbatch_iterator(dl_Y, subbatch_size, device='cpu'), subbatch_iterator(dl_X, subbatch_size, device='cpu')

    iteration = 0
    pbar = tqdm(total = num_iterations)
    rs = RunningMeanVariance()
    median_dist_means = []
    median_dist_stds = []

    while iteration + 1 <= num_iterations:
        try:
            aggregate = torch.cat(
                (
                    next(dl_Y_iter),
                    next(dl_X_iter)),
                dim=0
            ).to(device) # Concatenate the two batches
        except StopIteration:
            logger.info("End of dataset reached.")
            break
            
        # For the cross term, compute squared norms explicitly.
        aggregate_norm = (aggregate ** 2).sum(dim=1, keepdim=True)  # (n, 1)
        pairwise_squared_distances = (aggregate_norm + aggregate_norm.T - 2 * aggregate@aggregate.T)
        median_dist = pairwise_squared_distances.median().sqrt()

        # if median dist is na, skip and print
        if median_dist.isnan():
            logger.info(f"Median distance is NaN for iteration {iteration}, skipping...")
            continue

        rs(median_dist) # Update running stats

        if iteration +1 >=2:
            median_dist_mean, median_dist_var = rs.compute() # Compute iteration stats
            median_dist_means.append(median_dist_mean.item())
            median_dist_stds.append(median_dist_var.item()**0.5) # Append running std

            if median_dist_mean.isnan():
                logger.info(f"Median distance MEAN is NaN for iteration {iteration}, skipping...")
                continue

        iteration+=1
        pbar.update(1)

        if debug:
            if iteration % ( num_iterations // 10) == 0:
                logger.info(f"Iteration {iteration}/{num_iterations} -- Median Dist: {median_dist.item():.4f}")

    median_dist_means = np.array(median_dist_means)
    median_dist_stds = np.array(median_dist_stds)
    # get std error by dividing by Ns using length of len(median_dist_stds)
    median_dist_std_errors = median_dist_stds/np.sqrt(np.arange(2,len(median_dist_stds) + 2))
    pbar.close()
    return median_dist_means, median_dist_stds, median_dist_std_errors

        
def get_running_stats(data_loader, num_iterations,device):

    data_loader_iter = iter(data_loader)

    iteration = 0
    pbar = tqdm(total = num_iterations)
    rs = RunningMeanVariance()
    mean_means = np.zeros(num_iterations - 1)
    mean_stds = np.zeros(num_iterations - 1)

    while iteration + 1 <= num_iterations:
        mean = next(data_loader_iter).double().mean() # Compute mean on sample
        rs(mean) # Update running stats

        if iteration +1 >=2:
            mean_mean, mean_var = rs.compute() # Compute iteration stats
            mean_means[iteration - 1] = mean_mean.item()
            mean_stds[iteration - 1] = mean_var.item()**0.5 # Append running std

        iteration+=1
        pbar.update(1)

    mean_std_errors = mean_stds/np.sqrt(np.arange(2, num_iterations + 1))
    pbar.close()
    return mean_means, mean_stds, mean_std_errors


def get_mmd_stats(dl_Y, dl_X, num_iterations,subbatch_size,sigma,device):

    dl_Y_iter, dl_X_iter = subbatch_iterator(dl_Y, subbatch_size, device=device), subbatch_iterator(dl_X, subbatch_size, device=device)

    rs = RunningMeanVariance()
    iteration = 0
    pbar = tqdm(total = num_iterations)
    mmds = []
    mmd_means = []
    mmd_stds = []
    
    while iteration + 1 <= num_iterations:
        try:
            batch_Y = next(dl_Y_iter).double()
            batch_X = next(dl_X_iter).double()
        except StopIteration:
            logger.info("End of dataset reached.")
            break

        mmd = mmd_rbf(batch_Y, batch_X,sigma=sigma) #Compute mmd on sample

        if mmd.isnan() or mmd.isinf():
            logger.info(f"MMD is NaN for iteration {iteration}, skipping...")
            continue
        
        rs(mmd) # Update running stats
        mmds.append(mmd.item())
        
        if iteration +1 >=2:
            mmd_mean, mmd_var = rs.compute() # Compute iteration stats
            mmd_means.append(mmd_mean.item()) # Append running mean 
            mmd_stds.append(mmd_var.item()**0.5) # Append running std

        iteration+=1
        pbar.update(1)

    mmd_means = np.array(mmd_means)
    mmd_stds = np.array(mmd_stds)
    mmd_std_errors = mmd_stds/np.sqrt(np.arange(2,len(mmd_stds) + 2))
    pbar.close()
    return mmd_means, mmd_stds, mmd_std_errors, mmds


def subbatch_iterator(dataloader, subbatch_size, device):
    for batch in dataloader:
        batch = batch.squeeze(0)
        for i in range(0, batch.size(0), subbatch_size):
            subbatch = batch[i:i+subbatch_size]
            yield subbatch.to(device)

# Test Cases #
# X = torch.tensor([[0.0, 0.0],
#                     [1.0, 1.0]])
# Y = torch.tensor([[1.0, 0.0],
#                     [0.0, 1.0]])

# result = mmd_rbf(X, Y, sigma=1.0)
# logger.info(f"Computed MMD: {result.item():.4f}")

# # Expected from manual computation
# expected = 0.3679 + 0.3679 - 2 * 0.6065  # Since k_xx and k_yy only have one off-diagonal term each
# logger.info(f"Expected MMD (manual): {expected:.4f}")

# assert abs(result.item() - expected) < 1e-3, "Mismatch with manual MMD calculation"

# X = torch.tensor([[1.0, 2.0],
#                     [3.0, 4.0]])
# Y = torch.tensor([[5.0, 6.0],
#                     [7.0, 8.0]])

# result = mmd_rbf(X, Y, sigma=4.0)
# logger.info(f"Computed MMD: {result.item():.6f}")

# # Manual computation
# k_same = torch.exp(torch.tensor(-8.0 / 32)).item()*2  
# k_cross_sum = (torch.exp(torch.tensor(-32.0 / 32)) +  # ≈ 1.125e-7
#                 torch.exp(torch.tensor(-72.0 / 32)) +  # ≈ 2.32e-16
#                 torch.exp(torch.tensor(-8.0 / 32)) +   # ≈ 0.0183
#                 torch.exp(torch.tensor(-32.0 /32))).item()  # ≈ 1.125e-7

# expected = k_same/2 + k_same/2 - 0.5 * k_cross_sum
# logger.info(f"Expected MMD (manual): {expected:.6f}")

# assert abs(result.item() - expected) < 1e-6, "Mismatch with manual MMD calculation"

if __name__ == "__main__":
    '''
    sample usage:   python mmd.py --input-dir /data/generation_annotations --output-dir /data/mmd_results --subbatch-size 1000 --num-iterations 1
    '''

    parser = argparse.ArgumentParser(description='MMD Analysis')
    parser.add_argument('--input-dir', type=str, help='Input directory where all datasets are located (e.g. /data/generation_annotations)')
    parser.add_argument('--Y-datasets', type=str, nargs='+', help='List of Y datasets to use',default=['uniref50','gigaref_clustered'])
    parser.add_argument('--X-datasets', type=str, nargs='+', help='List of X datasets to use',default=['dayhoff','rfdiffusion_both_filter','rfdiffusion_scrmsd','rfdiffusion_unfiltered'])
    parser.add_argument('--output-dir', type=str, help='Output directory for results')
    parser.add_argument('--subbatch-size', type=int, default=1000, help='Subbatch size for MMD computation')
    parser.add_argument('--num-iterations', type=int, default=10_000, help='Number of iterations for MMD computation')
    parser.add_argument('--sigma', type=float, default=None, help='Sigma for RBF kernel')
    parser.add_argument('--not-save', action='store_true', help='Do not save results')
    parser.add_argument('--debug', action='store_true', help='Debug mode')

    args = parser.parse_args()

    is_amlt = os.environ.get("AMLT_OUTPUT_DIR", None) is not None

    if is_amlt:
        args.output_dir = os.path.join(os.environ["AMLT_OUTPUT_DIR"], args.output_dir)

    # create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    #print args
    logger.info(args)

    d = 32102

    transform = transforms.Compose(
        [
            transforms.Lambda(lambda x: x / float(round((d/10)**0.5)))
        ]
    )

    
    ds_to_pattern = {
        'dayhoff': os.path.join(args.input_dir,'DAYHOFF_GENERATIONS/test_1_logits*'),
        'rfdiffusion_both_filter': os.path.join(args.input_dir,'RFDIFFUSION_GENERATIONS_BOTH_FILTER/test_1_logits*'),
        'rfdiffusion_scrmsd': os.path.join(args.input_dir,'RFDIFFUSION_GENERATIONS_SCRMSD/test_1_logits*'),
        'rfdiffusion_unfiltered': os.path.join(args.input_dir,'RFDIFFUSION_GENERATIONS_UNFILTERED/test_1_logits*'),
        'uniref50_first_half': os.path.join(args.input_dir,'UNIREF50_10M_first_half/test_1_logits*'),
        'uniref50_second_half': os.path.join(args.input_dir,'UNIREF50_10M_second_half/test_1_logits*'),
        'gigaref_clustered_first_half': os.path.join(args.input_dir,'GIGAREF_CLUSTERED_10M_first_half/test_1_logits*'),
        'gigaref_clustered_second_half': os.path.join(args.input_dir,'GIGAREF_CLUSTERED_10M_second_half/test_1_logits*'),
        'uniref50': os.path.join(args.input_dir,'UNIREF50_10M/test_1_logits*'),
        'gigaref_clustered': os.path.join(args.input_dir,'GIGAREF_CLUSTERED_10M/test_1_logits*'),
        'gigaref_singletons': os.path.join(args.input_dir,'GIGAREF_SINGLETONS_10M/test_1_logits*')
    }

    Y_datasets_patterns = [
        ds_to_pattern[ds] for ds in args.Y_datasets
    ]
    
    X_datasets_patterns = [
        ds_to_pattern[ds] for ds in args.X_datasets
    ]


    # Verify X dataset patters and Y dataset patterns exist
    for X_glob_pattern in X_datasets_patterns:
        assert glob(X_glob_pattern), f"X dataset pattern {X_glob_pattern} does not exist"
           
    for Y_glob_pattern in Y_datasets_patterns:
        assert glob(Y_glob_pattern), f"Y dataset pattern {Y_glob_pattern} does not exist"


    for X_glob_pattern in X_datasets_patterns:
        for Y_glob_pattern in Y_datasets_patterns:
            
            X_dataset_name = X_glob_pattern.split('/')[-2]
            Y_dataset_name = Y_glob_pattern.split('/')[-2]

            logger.info(f"X dataset: {X_dataset_name}")
            logger.info(f"Y dataset: {Y_dataset_name}")

            
            ds_X = H5BatchedDataset(
                h5_paths = glob(X_glob_pattern), 
                transform = transform
            ) 
            ds_Y = H5BatchedDataset(
                h5_paths = glob(Y_glob_pattern),
                transform = transform
            ) 

            dl_X = torch.utils.data.DataLoader(
                ds_X,
                batch_size=1,
                num_workers=2,
                pin_memory=True,
                shuffle=False
            )

            dl_Y = torch.utils.data.DataLoader(
                ds_Y,
                batch_size=1,
                num_workers=2,
                pin_memory=True,
                shuffle=False
            )

            #Params
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            if args.sigma is None:
                logger.info("getting median sigma")
                (median_dist_means, 
                median_dist_stds,
                median_dist_std_errors) = get_sigma_median(dl_Y, dl_X, args.num_iterations,subbatch_size=args.subbatch_size, device = device, debug=args.debug)
                sigma = median_dist_means[-1] #Setting sigma to median heuristic
                if args.debug:
                    logger.info(f"Median Dist Means: {median_dist_means[-1]}")

                logger.info(f"Setting sigma to {sigma}")
            else:
                sigma = args.sigma
                logger.info(f"Using sigma {sigma}")

            logger.info("getting mmd")
            (mmd_means,
            mmd_stds,
            mmd_std_errors, 
            mmds) = get_mmd_stats(dl_Y, dl_X, args.num_iterations, subbatch_size=args.subbatch_size,sigma = sigma, device = device)

            if args.debug:
                logger.info(f"MMD Means: {mmd_means[-1]}")

            if not args.not_save:
                logger.info("Saving results")
                #Save MMD info
                np.save(
                    os.path.join(args.output_dir,f'mmd_means_{X_dataset_name}_{Y_dataset_name}.npy'),
                    np.array(mmd_means)
                    )
                np.save(
                    os.path.join(args.output_dir,f'mmd_stds_{X_dataset_name}_{Y_dataset_name}.npy'),
                    np.array(mmd_stds)
                    )
                np.save(
                    os.path.join(args.output_dir,f'mmd_std_errors_{X_dataset_name}_{Y_dataset_name}.npy'),
                    np.array(mmd_std_errors)
                    )
                
                #Save median info
                if args.sigma is None:
                        
                    np.save(
                        os.path.join(args.output_dir,f'median_dist_means_{X_dataset_name}_{Y_dataset_name}.npy'),
                        np.array(median_dist_means)
                        )
                    np.save(
                        os.path.join(args.output_dir,f'median_dist_std_errors_{X_dataset_name}_{Y_dataset_name}.npy'),
                        np.array(median_dist_std_errors)
                        )