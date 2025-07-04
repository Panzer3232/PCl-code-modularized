import torch
import torch.nn as nn
import utils
import pcl
import pcl_util
from runner_utils import pcl_postprocess, calculate_batch_mpjpe, calculate_loss
from config_data import DatasetConfig


def runner_2d3d(epoch, data_loader, model, optimizer, loss_fn, device, use_dataset,
                use_pcl, slant_compensation, denormalize_during_training, training,
                dataset_cfg: DatasetConfig):

    if training:
        model.train()
    else:
        model.eval()
        validation_loss = 0.0

    for i, data in enumerate(data_loader):
        Ks_px_orig = data['camera_original']
        orig_img_shape = data['original_img_shape']
        label = data['normalized_skel_mm'].to(device)
        label_no_norm = data['non_normalized_3d']
        pelvis_location = data['pelvis_location_mm']
        P_px = data['perspective_matrix'] 
        location_px = data['crop_location'].float()
        scale_px = data['crop_scale'].float()
        label_2d_px = data['pose2d_original'] #no hip location removal, 32x2

        square_scale = torch.tensor([torch.max(scale_px.squeeze(0)), torch.max(scale_px.squeeze(0))])
        square_scale_py = square_scale / data['original_img_shape'].squeeze(0)

        scale_py = square_scale_py.unsqueeze(0)
        location_py = utils.pixel_2_pytorch_locations(location_px.cpu(), orig_img_shape[:, 0], orig_img_shape[:, 1]).to(device)

        if use_dataset == "H36m":
            hips = label_2d_px[:,0,:].unsqueeze(1).repeat(1,label_2d_px.shape[1],1)
        else:
            hips = hips = label_2d_px[:,14,:].unsqueeze(1).repeat(1,label_2d_px.shape[1],1)
        label_2d_no_hip = label_2d_px - hips
        #Canonical handld in config_data.py
        canon_label_2d = dataset_cfg.to_canonical(label_2d_no_hip).to(device)
        canon_label_3d = dataset_cfg.to_canonical(label).to(device)
        label_no_norm = dataset_cfg.to_canonical(label_no_norm)

        bs = canon_label_2d.shape[0]
        num_joints = canon_label_2d.shape[1]

        if use_pcl:
            model_input = data['preprocess-model_input'].to(device)
            canon_virt_2d = data['preprocess-canon_virt_2d'].to(device)
            R_virt2orig = data['preprocess-R_virt2orig']
        else:
            model_input = canon_label_2d.detach().clone()
            model_input = model_input / scale_py.unsqueeze(1).to(device)
            #Mean,std calculated in config_data.py with dataset configuration
            mean, std = dataset_cfg.get_mean_std_normalized(device, slant=slant_compensation, use_pcl=False)
            model_input = utils.batch_normalize_canon_human_joints(model_input, mean, std)
            model_input = model_input.view(bs, -1)

        if training:
            optimizer.zero_grad()

        output = model(model_input.to(device))

        if use_pcl:
            postprocess = pcl_postprocess(bs, num_joints, output, R_virt2orig, device, use_dataset)
            output = postprocess['output_no_norm'] if denormalize_during_training else postprocess['output']
            normalized_output = postprocess['output']
            pre_transform = postprocess['pre_transform']
        else:
            output = output.view(canon_label_3d.shape[0], -1, 3)
            if denormalize_during_training:
                #Handled in config_data.py
                mean_3d = dataset_cfg.get_joint_mean().to(device)
                std_3d = dataset_cfg.get_joint_std().to(device)
                output = utils.denorm_human_joints(output, mean_3d, std_3d)

        if training:
            loss = loss_fn(output, label_no_norm.to(device)) if denormalize_during_training else loss_fn(output, canon_label_3d)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()

            if i % 1000 == 0:
                print(f"Epoch: {epoch} | Iteration: {i} | Loss: {loss.item():.4f}")

        else:
            if not denormalize_during_training:
                mean_3d = dataset_cfg.get_joint_mean().to(device)
                std_3d = dataset_cfg.get_joint_std().to(device)
                output = utils.denorm_human_joints(output, mean_3d, std_3d)
            loss = calculate_batch_mpjpe(output.detach(), label_no_norm.to(device))
            validation_loss += loss.item()

    return (model, optimizer) if training else validation_loss / len(data_loader)


def runner_3dFromImage(epoch, data_loader, model, optimizer, loss_fn, device, use_dataset,
                       use_pcl, use_canonical, training, dataset_cfg: DatasetConfig):

    if training:
        model.train()
    else:
        model.eval()
        validation_loss = 0.0

    for i, data in enumerate(data_loader):
        input_small = data['input'].to(device)
        input_big = data['input_big'].to(device)
        img_big = data['input_big_img']
        Ks_px_orig = data['camera_original'].to(device)
        orig_img_shape = data['original_img_shape']
        label = data['normalized_skel_mm']
        label_no_norm = dataset_cfg.to_canonical(data['non_normalized_3d']).to(device)
        pelvis_location = data['pelvis_location_mm']
        location_px = data['crop_location'].float().to(device)
        scale_px = data['crop_scale'].float().to(device)

        location_py = utils.pixel_2_pytorch_locations(location_px.cpu(), orig_img_shape[:, 0], orig_img_shape[:, 1]).to(device)
        scale_py = scale_px / orig_img_shape if use_pcl else data['stn_square_scale_py'].to(device)

        img_w_h_shape = torch.tensor([input_big.shape[3], input_big.shape[2]]).to(device)
        label_2d = dataset_cfg.to_canonical(data['pose2d_original'])
        P_px = data['perspective_matrix'] 
        
        if training:
            optimizer.zero_grad()

        if use_pcl:
            Ks = pcl_util.K_new_resolution_px(Ks_px_orig, orig_img_shape.to(device), img_w_h_shape).to(device)
            output_dict = model(input_big, input_small, Ks, position_gt=location_py, scale_gt=scale_py,
                                rectangular_images=(use_dataset != 'H36m'))
            output = output_dict["output"]
            output_no_norm = output_dict["output_no_norm"]
            theta = output_dict["theta"]
        else:
            output_dict = model(input_big, input_small, position_gt=location_py, scale_gt=scale_py)
            output = output_dict["output"]
            theta = output_dict["theta"]

            mean_3d = dataset_cfg.get_joint_mean().to(device)
            std_3d = dataset_cfg.get_joint_std().to(device)
            output_no_norm = utils.denorm_human_joints(output, mean_3d, std_3d)

        if training:
            loss, loc_loss, scale_loss, regression_loss = calculate_loss(loss_fn, theta, output_no_norm, label_no_norm,
                                                                          device, scale_py, location_py)
            loss.backward()
            optimizer.step()
            if i % 1000 == 0:
                print(f"Epoch: {epoch} | Iteration: {i} | Loss: {loss.item():.4f}")
            return model, optimizer

        else:
            loss = calculate_batch_mpjpe(output.detach(), label_no_norm.to(device))
            validation_loss += loss.item()
            return validation_loss / len(data_loader)
