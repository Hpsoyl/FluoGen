import os
import numpy as np
from scipy.stats import spearmanr
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from utils.bounds import HB_mu_plus
import glob
import os
import yaml
import tifffile

from tqdm import trange
from torch.utils.data import Dataset
from skimage import io
import matplotlib as plt

def quantile_regression_nested_sets_from_output(output, lam=None):
    output[:,0,:,:] = torch.minimum(output[:,0,:,:], output[:,1,:,:]-1e-6)
    output[:,2,:,:] = torch.maximum(output[:,2,:,:], output[:,1,:,:]+1e-6)
    upper_edge = lam * (output[:,2,:,:] - output[:,1,:,:]) + output[:,1,:,:] 
    lower_edge = output[:,1,:,:] - lam * (output[:,1,:,:] - output[:,0,:,:])

    return lower_edge, output[:,1,:,:], upper_edge 

def nested_sets_from_output(output, lam=None):
    lower_edge, prediction, upper_edge = quantile_regression_nested_sets_from_output(output, lam)
    upper_edge = torch.maximum(upper_edge, prediction + 1e-6) # set a lower bound on the size.
    lower_edge = torch.minimum(lower_edge, prediction - 1e-6)

    return lower_edge, prediction, upper_edge 

def get_rcps_losses(model, dataset, rcps_loss_fn, lam, device):
    losses = []
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=True) 
    for batch, labels in dataloader:
        sets = model.nested_sets_from_output(batch,lam) 
        losses = losses + [rcps_loss_fn(sets, labels),]
    return torch.cat(losses,dim=0)

def get_rcps_losses_from_outputs(out_dataset, rcps_loss_fn, lam, device):
    losses = []
    dataloader = DataLoader(out_dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)
    for batch in dataloader:
        x, labels = batch
        sets = nested_sets_from_output(x.to(device),lam) 
        losses = losses + [rcps_loss_fn(sets, labels.to(device)).cpu(),]
    return torch.cat(losses,dim=0)

def get_rcps_metrics_from_outputs(lhat, dataset, rcps_loss_fn, device, specimen):
    losses = []
    sizes = []
    residuals = []
    spatial_miscoverages = []
    colormap = plt.colormaps.get_cmap('jet')
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)
    for batch in dataloader:
        x, labels, output_path = batch
        labels = labels.to(device)
        sets = nested_sets_from_output(x.to(device), lhat)
        sets_range = sets[2]-sets[0]

        for i in range(len(output_path)):
            mapped_data = colormap(sets_range[i].cpu().numpy())
            mapped_data = (mapped_data[..., :3] * 255).astype(np.uint8)
            save_path = os.path.join(f"/data0/syhong/BioDiff/validation_output/Uncertainty_{specimen}", os.path.basename(os.path.dirname(output_path[i])))
            os.makedirs(save_path, exist_ok=True)
            io.imsave(os.path.join(save_path, os.path.basename(output_path[i])), mapped_data)

        losses = losses + [rcps_loss_fn(sets, labels),]
        sets_full = sets_range.flatten(start_dim=1).detach().cpu().numpy()
        size_random_idxs = np.random.choice(sets_full.shape[1],size=sets_full.shape[0])
        size_samples = sets_full[range(sets_full.shape[0]),size_random_idxs]
        residuals = residuals + [(labels - sets[1]).abs().flatten(start_dim=1)[range(sets_full.shape[0]),size_random_idxs]]
        spatial_miscoverages = spatial_miscoverages + [(labels > sets[2]).float() + (labels < sets[0]).float()]
        sizes = sizes + [torch.tensor(size_samples),]

    losses = torch.cat(losses,dim=0)
    sizes = torch.cat(sizes,dim=0)
    sizes = sizes + torch.rand(size=sizes.shape).to(sizes.device)*1e-6
    residuals = torch.cat(residuals,dim=0).detach().cpu().numpy() 
    spearman = spearmanr(residuals, sizes)[0]
    mse = (residuals*residuals).mean().item()
    spatial_miscoverage = torch.cat(spatial_miscoverages, dim=0).detach().cpu().numpy().mean(axis=0).mean(axis=0)
    size_bins = torch.tensor([0, torch.quantile(sizes, 0.25), torch.quantile(sizes, 0.5), torch.quantile(sizes, 0.75)])
    buckets = torch.bucketize(sizes, size_bins)-1
    stratified_risks = torch.tensor([losses[buckets == bucket].mean() for bucket in range(size_bins.shape[0])])
    print(f"Model output shape: {x.shape}, label shape: {labels.shape}, Sets shape: {sets[2].shape}, sizes: {sizes}, size_bins:{size_bins}, stratified_risks: {stratified_risks}, mse: {mse}")
    return losses, sizes, spearman, stratified_risks, mse, spatial_miscoverage

def evaluate_from_loss_table(loss_table,n,alpha,delta):
  with torch.no_grad():
    perm = torch.randperm(loss_table.shape[0])
    loss_table = loss_table[perm]
    calib_table, val_table = loss_table[:n], loss_table[n:]
    Rhats = calib_table.mean(dim=0)
    RhatPlus = torch.tensor([HB_mu_plus(Rhat, n, delta) for Rhat in Rhats])
    try:
        idx_lambda = (RhatPlus <= delta).nonzero()[0]
    except:
        print("No rejections made!")
        idx_lambda = 0
    return val_table[:,idx_lambda].mean()
  
def fraction_missed_loss(pset,label):
    misses = (pset[0].squeeze() > label.squeeze()).float() + (pset[2].squeeze() < label.squeeze()).float()
    misses[misses > 1.0] = 1.0
    d = len(misses.shape)
    return misses.mean(dim=tuple(range(1,d)))

def get_rcps_loss_fn(config):
    string = config['rcps_loss']
    if string == 'fraction_missed':
        return fraction_missed_loss
    else:
        raise NotImplementedError

def calibrate_model(dataset, config):
    with torch.no_grad():
        print(f"Calibrating...")
        alpha = config['alpha']
        delta = config['delta']
        device = config['device']
        print("Initialize lambdas")
        lambdas = torch.linspace(config['minimum_lambda'],config['maximum_lambda'],config['num_lambdas'])
        print("Initialize loss")
        rcps_loss_fn = get_rcps_loss_fn(config)

        labels_shape = list(dataset[0][1].unsqueeze(0).shape)
        labels_shape[0] = len(dataset)
        labels = torch.zeros(tuple(labels_shape), device='cpu')
        outputs_shape = list(dataset[0][0].unsqueeze(0).to(device).shape)
        outputs_shape[0] = len(dataset)
        outputs = torch.zeros(tuple(outputs_shape),device='cpu')
        print("Collecting dataset")
        tempDL = DataLoader(dataset, num_workers=0, batch_size=config['batch_size'], pin_memory=True) 
        counter = 0
        for batch in tqdm(tempDL):
            outputs[counter:counter+batch[0].shape[0],:,:,:] = batch[0].to(device).cpu()
            labels[counter:counter+batch[1].shape[0]] = batch[1]
            counter += batch[0].shape[0]

        print("Output dataset")
        out_dataset = TensorDataset(outputs,labels.cpu())
        dlambda = lambdas[1]-lambdas[0]
        lhat = lambdas[-1]+dlambda-1e-9
        print("Computing losses")
        calib_loss_table = torch.zeros((outputs.shape[0],lambdas.shape[0]))
        for lam in reversed(lambdas):
            losses = get_rcps_losses_from_outputs(out_dataset, rcps_loss_fn, lam-dlambda, device)
            calib_loss_table[:,np.where(lambdas==lam)[0]] = losses[:,None]
            Rhat = losses.mean()
            RhatPlus = HB_mu_plus(Rhat.item(), losses.shape[0], delta)
            print(f"\rLambda: {lam:.4f}  |  Rhat: {Rhat:.4f}  |  RhatPlus: {RhatPlus:.4f}  ",end='')
            if Rhat >= alpha or RhatPlus > alpha:
                lhat = lam
                print("")
                print(f"lhat is {lhat}")
                break
        return lhat, calib_loss_table

def eval_set_metrics(lhat, dataset, config, specimen):
    device = config['device']
    rcps_loss_fn = get_rcps_loss_fn(config)

    labels = torch.cat([x[1].unsqueeze(0).to(device).to('cpu') for x in dataset], dim=0).cpu()
    outputs_shape = list(dataset[0][0].unsqueeze(0).to(device).shape)
    outputs_shape[0] = len(dataset)
    outputs = torch.zeros(tuple(outputs_shape),device='cpu')
    for i in trange(len(dataset)):
        # print(f"Validation output {i}")
        outputs[i,:,:,:] = dataset[i][0].unsqueeze(0).to(device).cpu()
        outputs_path.append(dataset[i][2])
    out_dataset = TensorDataset(outputs,labels)

    print("GET RCPS METRICS FROM OUTPUTS")
    losses, sizes, spearman, stratified_risks, mse, spatial_miscoverage = get_rcps_metrics_from_outputs(lhat, dataset, rcps_loss_fn, device, specimen)
    print("DONE!")
    return losses.mean(), sizes, spearman, stratified_risks, mse, spatial_miscoverage
  

class calib_dataset(Dataset):
    def __init__(self, data_path, specimen):
        self.calib_path = glob.glob(os.path.join(data_path, "*.tif"))
        self.spcimen = specimen

    def __getitem__(self, index):
        basename = os.path.basename(self.calib_path[index])
        gt_path = os.path.join(f"/lab310/enhance/BioDiffuse_Base/bioSR_512/{self.spcimen}/validate/validate_gt", basename)
        calib_img = tifffile.imread(self.calib_path[index])
        gt_img = tifffile.imread(gt_path) / 65535

        return torch.from_numpy(calib_img), torch.from_numpy(gt_img)

    def __len__(self):
        return len(self.calib_path)
    
class test_dataset(Dataset):
    def __init__(self, data_path, specimen):
        self.calib_path = glob.glob(os.path.join(data_path, "*.tif"))
        self.spcimen = specimen

    def __getitem__(self, index):
        basename = os.path.basename(self.calib_path[index])
        gt_path = os.path.join(f"/lab310/enhance/BioDiffuse_Base/BioSR_daijk/test/{self.spcimen}/testing_gt", basename)
        calib_img = tifffile.imread(self.calib_path[index])
        gt_img = tifffile.imread(gt_path) / 65535

        return torch.from_numpy(calib_img), torch.from_numpy(gt_img), self.calib_path[index]

    def __len__(self):
        return len(self.calib_path)

if __name__ == "__main__":
    specimen = "F-actin"
    CLIP_path="stable-diffusion-v1-5"
    calib_path = f"validation_output/Uncertainty_{specimen}_prediction_val"
    test_path = f"validation_output/Uncertainty_{specimen}_prediction_test"

    config_file = "/data0/syhong/BioDiff/src/utils/config.yml"
    with open(config_file) as file:
        config = yaml.safe_load(file)

    weight_dtype = torch.float16
    device = "cuda:5"

    dataset_calib = calib_dataset(calib_path, specimen)
    dataset_test = test_dataset(test_path, specimen)

    lhat, _ = calibrate_model(dataset_calib, config)
    # lhat = torch.tensor(0.3684210479259491)
    risk, sizes, spearman, stratified_risk, mse, spatial_miscoverage = eval_set_metrics(lhat, dataset_test, config, specimen)
    print(f"Risk: {risk}  |  Mean size: {sizes.mean()}  |  Spearman: {spearman}  |  stratified risk: {stratified_risk}  | MSE: {mse}")


