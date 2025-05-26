import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from model import create_model
from data.dataset_patch import SR3_Dataset_patch
from torch.utils.data import DataLoader
import argparse
import glob
from configs import Config

def load_config_and_model(checkpoint_path, config_path):
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default=config_path)
    parser.add_argument("-gpu", "--gpu_ids", type=str, default=None)
    args = parser.parse_args()
    
    configs = Config(args)
    
    diffusion = create_model(
        in_channel=configs.in_channel,
        out_channel=configs.out_channel,
        norm_groups=configs.norm_groups,
        inner_channel=configs.inner_channel,
        channel_multiplier=configs.channel_multiplier,
        attn_res=configs.attn_res,
        res_blocks=configs.res_blocks,
        dropout=configs.dropout,
        diffusion_loss=configs.diffusion_loss,
        conditional=configs.conditional,
        gpu_ids=configs.gpu_ids,
        distributed=configs.distributed,
        init_method=configs.init_method,
        train_schedule=configs.train_schedule,
        train_n_timestep=configs.train_n_timestep,
        train_linear_start=configs.train_linear_start,
        train_linear_end=configs.train_linear_end,
        val_schedule=configs.val_schedule,
        val_n_timestep=configs.val_n_timestep,
        val_linear_start=configs.val_linear_start,
        val_linear_end=configs.val_linear_end,
        finetune_norm=configs.finetune_norm,
        optimizer=None,
        amsgrad=False,
        learning_rate=configs.lr,
        checkpoint=checkpoint_path,
        resume_state=None,
        phase="val",
        height=configs.height
    )
    
    return configs, diffusion

def visualize_results(diffusion, val_loader, output_dir, variable_name, num_samples=5):
    os.makedirs(output_dir, exist_ok=True)
    
    # Use the same number of timesteps as defined in your config (250)
    ddim_steps = 249  # This should match your config's n_timestep
    
    val_data = next(iter(val_loader))
    diffusion.feed_data(val_data)
    diffusion.test(continuous=False, use_ddim=True, ddim_steps=ddim_steps, use_dpm_solver=False)
    visuals = diffusion.get_current_visuals()
    
    sr_candidates = diffusion.generate_multiple_candidates(n=10, ddim_steps=ddim_steps, use_dpm_solver=False)
    
    mean_candidate = sr_candidates.mean(dim=0)
    std_candidate = sr_candidates.std(dim=0)
    bias = mean_candidate - visuals["HR"]
    
    plot_config = {
        "default": {"vmin": -2, "vmax": 2, "cmap": "RdBu_r"},
        # "tp": {"vmin": 0, "vmax": 2, "cmap": "BrBG"},
        # "u": {"vmin": -1, "vmax": 1, "cmap": "RdBu_r"},
        # "v": {"vmin": -1, "vmax": 1, "cmap": "RdBu_r"},
        # "t2m": {"vmin": 250, "vmax": 310, "cmap": "RdBu_r"},
        # "sp": {"vmin": 900, "vmax": 1100, "cmap": "RdBu_r"}
    }
    params = plot_config.get(variable_name, plot_config["default"])
    
    random_idx = np.random.randint(0, val_data["HR"].shape[0], num_samples)
    
    for i, idx in enumerate(random_idx):
        fig, axs = plt.subplots(2, 4, figsize=(20, 10))
        
        axs[0,0].imshow(visuals["HR"][idx,0].cpu().numpy(), **params)
        axs[0,0].set_title("HR")
        
        axs[0,1].imshow(visuals["SR"][idx,0].cpu().numpy(), **params)
        axs[0,1].set_title("SR")
        
        axs[0,2].imshow(visuals["INTERPOLATED"][idx,0].cpu().numpy(), **params)
        axs[0,2].set_title("INTERPOLATED")
        
        axs[0,3].imshow(mean_candidate[idx,0].cpu().numpy(), **params)
        axs[0,3].set_title("MEAN (10 samples)")

        axs[1,0].imshow(np.abs(visuals["HR"][idx,0].cpu().numpy() - visuals["SR"][idx,0].cpu().numpy()), 
                       vmin=0, vmax=2, cmap="Reds")
        axs[1,0].set_title("SR Error")
        
        axs[1,1].imshow(np.abs(visuals["HR"][idx,0].cpu().numpy() - visuals["INTERPOLATED"][idx,0].cpu().numpy()), 
                       vmin=0, vmax=2, cmap="Reds")
        axs[1,1].set_title("INTERP Error")
        
        axs[1,2].imshow(np.abs(bias[idx,0].cpu().numpy()), 
                       vmin=0, vmax=2, cmap="Reds")
        axs[1,2].set_title("MEAN Bias")
        
        axs[1,3].imshow(std_candidate[idx,0].cpu().numpy(), 
                        vmin=0, vmax=2, cmap="Reds")
        axs[1,3].set_title("STD (10 samples)")
        
        sr_mae = np.abs(visuals["HR"][idx,0].cpu().numpy() - visuals["SR"][idx,0].cpu().numpy()).mean()
        interp_mae = np.abs(visuals["HR"][idx,0].cpu().numpy() - visuals["INTERPOLATED"][idx,0].cpu().numpy()).mean()
        mean_mae = np.abs(bias[idx,0].cpu().numpy()).mean()
        
        plt.suptitle(f"Sample {i+1} - SR MAE: {sr_mae:.3f}, INTERP MAE: {interp_mae:.3f}, MEAN MAE: {mean_mae:.3f}", y=1.02)
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, f"sample_{i+1}.png"), bbox_inches="tight", dpi=150)
        plt.close()
        
        np.savez(
            os.path.join(output_dir, f"sample_{i+1}_data.npz"),
            HR=visuals["HR"][idx,0].cpu().numpy(),
            SR=visuals["SR"][idx,0].cpu().numpy(),
            INTERPOLATED=visuals["INTERPOLATED"][idx,0].cpu().numpy(),
            mean_candidate=mean_candidate[idx,0].cpu().numpy(),
            std_candidate=std_candidate[idx,0].cpu().numpy()
        )

if __name__ == "__main__":
    checkpoint_path = "/workspace3/suwen/ddpm/experiments/0514_250515_225832/checkpoints/I50_E1_gen.pth"
    config_path = "/workspace3/suwen/ddpm/configs/sample_ddpm_128.json" 
    output_dir = "/workspace3/suwen/ddpm/experiments/0514_250515_225832/visualizations"
    variable_name = "default"  
    
    target_paths = sorted(glob.glob("/workspace3/suwen/ddpm/dataset/hr/*npy"))
    lr_paths = sorted(glob.glob("/workspace3/suwen/ddpm/dataset/lr/*npy"))
    val_data = SR3_Dataset_patch(target_paths[:2], lr_paths=lr_paths[:2], var=None, patch_size=128)
    val_loader = DataLoader(val_data, batch_size=4, shuffle=False, num_workers=4)
    
    configs, diffusion = load_config_and_model(checkpoint_path, config_path)
    
    visualize_results(diffusion, val_loader, output_dir, variable_name, num_samples=5)
    
    print(f"Plots saved: {output_dir}")