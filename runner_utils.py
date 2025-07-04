import torch
import pcl
from config_data import DatasetConfig

from utils import denorm_human_joints, batch_normalize_canon_human_joints

def pcl_postprocess(batch_size, num_joints, output, R_virt2orig, device, dataset_cfg: DatasetConfig):
    #MEan,std dataset specific handled in config_data.py and no need to call .to(device) again and again
    mean = dataset_cfg.get_joint_mean().to(device)
    std = dataset_cfg.get_joint_std().to(device)

    R_virt2orig = R_virt2orig.to(device)
    new_pre_transform = output.view(batch_size, num_joints, -1)
    new_pre_transform = denorm_human_joints(new_pre_transform, mean, std).unsqueeze(3)

    new_output = pcl.virtPose2CameraPose(new_pre_transform, R_virt2orig, batch_size, num_joints)

    normalized_output = batch_normalize_canon_human_joints(new_output, mean=mean, std=std)
    new_pre_transform = new_pre_transform.squeeze(-1).view(batch_size, num_joints, -1)

    new_pre_transform = batch_normalize_canon_human_joints(new_pre_transform, mean=mean, std=std)
    return {"pre_transform":new_pre_transform, "output":normalized_output, 'output_no_norm':new_output}


def calculate_loss(loss_fn, theta, output, label, device, scale, location):
    pred_location = theta[:,2:]
    pred_scale = theta[:,0:2]

    loc_loss = loss_fn(pred_location, location).to(device)
    scale_loss = loss_fn(pred_scale, scale).to(device)
    regression_loss = loss_fn(output, label).to(device)
    loss = loc_loss + scale_loss + regression_loss
    loss = loc_loss + scale_loss + regression_loss

    return loss, loc_loss, scale_loss, regression_loss

def calculate_batch_mpjpe(output, label):
    difference =  output - label 
    square_difference = torch.square(difference) 
    sum_square_difference_per_point = torch.sum(square_difference, dim=2) 
    euclidean_distance_per_point = torch.sqrt(sum_square_difference_per_point) 
    mpjpe = torch.mean(euclidean_distance_per_point)
    return mpjpe
