from __future__ import print_function
import sys, os, time, torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset_h36m import H36MDataset
from dataset_3dhp import MpiInf3dDataset
from model import Resnet_H36m, LinearModel, weight_init
from runner import runner_2d3d, runner_3dFromImage
from config_data import get_dataset_config
import configargparse

def config_parser():

    parser = configargparse.ArgumentParser()

    parser.add_argument('--use_pcl', action='store_true')
    parser.add_argument('--total_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--exp_type', type=str, default='2d3d')
    parser.add_argument('--use_dataset', type=str, default='H36m')
    parser.add_argument('--use_canonical', action='store_true')

    parser.add_argument('--run_name', type=str, default='test')
    parser.add_argument('--model_path', type=str, default='model')
    parser.add_argument('--dataset_root', type=str)
    parser.add_argument('--denormalize_during_training', action='store_true')
    parser.add_argument('--use_slant_compensation', action='store_true')
    parser.add_argument('--use_2d_scale', action='store_true')

    parser.add_argument('--use_mpi_aug', action='store_true')
    parser.add_argument('--use_resnet50', action='store_true')
    parser.add_argument('--use_pretrain', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    return parser

if __name__ == "__main__":
    args = config_parser().parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create dataset config
    dataset_cfg = get_dataset_config(args.use_dataset)
    num_joints = dataset_cfg.get_num_joints(use_canonical=args.use_canonical)

    # Create model folder
    model_folder = os.path.join(args.model_path, args.run_name)
    os.makedirs(model_folder, exist_ok=True)

    # Load datasets
    if args.exp_type == '2d3d':
         # 2D to 3D keypoint lifting using GT 2D pose
        if args.use_dataset == 'H36m':
            train_dataset = H36MDataset(args.dataset_root, subset='trainval', without_image=True,
                                        use_pcl=args.use_pcl, calculate_scale_from_2d=args.use_2d_scale,
                                        use_slant_compensation=args.use_slant_compensation)
            validation_dataset = H36MDataset(args.dataset_root, subset='test', without_image=True,
                                             use_pcl=args.use_pcl, calculate_scale_from_2d=args.use_2d_scale,
                                             use_slant_compensation=args.use_slant_compensation)
        else:
            train_path = os.path.join(args.dataset_root, 'train')
            val_path = os.path.join(args.dataset_root, 'val')
            train_dataset = MpiInf3dDataset(train_path, without_image=True, use_pcl=args.use_pcl,
                                            calculate_scale_from_2d=args.use_2d_scale,
                                            use_slant_compensation=args.use_slant_compensation)
            validation_dataset = MpiInf3dDataset(val_path, without_image=True, use_pcl=args.use_pcl,
                                                 calculate_scale_from_2d=args.use_2d_scale,
                                                 use_slant_compensation=args.use_slant_compensation)
        model = LinearModel()
        model.to(device)
        model.apply(weight_init)

    else:
        if args.use_dataset == 'H36m':
            train_dataset = H36MDataset(args.dataset_root, subset='trainval', without_image=False, use_pcl=args.use_pcl,
                                        calculate_scale_from_2d=args.use_2d_scale,
                                        use_slant_compensation=args.use_slant_compensation)
            validation_dataset = H36MDataset(args.dataset_root, subset='test', without_image=False, use_pcl=args.use_pcl,
                                             calculate_scale_from_2d=args.use_2d_scale,
                                             use_slant_compensation=args.use_slant_compensation)
        else:
            train_path = os.path.join(args.dataset_root, 'train')
            val_path = os.path.join(args.dataset_root, 'val')
            train_dataset = MpiInf3dDataset(train_path, without_image=False, use_pcl=args.use_pcl,
                                            calculate_scale_from_2d=args.use_2d_scale,
                                            use_slant_compensation=True, use_aug=args.use_mpi_aug)
            validation_dataset = MpiInf3dDataset(val_path, without_image=False, use_pcl=args.use_pcl,
                                                 calculate_scale_from_2d=args.use_2d_scale,
                                                 use_slant_compensation=True, use_aug=args.use_mpi_aug)

        model = Resnet_H36m(device, num_joints=num_joints, use_pcl=args.use_pcl,
                            use_resnet50=args.use_resnet50, use_pretrain=args.use_pretrain,
                            dataset=args.use_dataset).to(device)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    loss_fn = nn.MSELoss()
    lowest_validation_loss = 1e7

    print("Starting Training")
    for epoch in range(args.total_epochs):
        time.sleep(2)  # solution found so training doesn't randomly crash
         # Experiments for 2D to 3D Keypoint Lifting
        if args.exp_type == '2d3d':
            model, optimizer = runner_2d3d(epoch, train_loader, model, optimizer, loss_fn, device,
                                           args.use_dataset, args.use_pcl, args.use_slant_compensation,
                                           args.denormalize_during_training, training=True,
                                           dataset_cfg=dataset_cfg)

            time.sleep(2)# solution found so training doesn't randomly crash
            validation_loss = runner_2d3d(epoch, validation_loader, model, optimizer, None, device,
                                          args.use_dataset, args.use_pcl, args.use_slant_compensation,
                                          args.denormalize_during_training, training=False,
                                          dataset_cfg=dataset_cfg)
        # Experiments for 3D Pose from Image Regression    
        else:
            model, optimizer = runner_3dFromImage(epoch, train_loader, model, optimizer, loss_fn, device,
                                                  args.use_dataset, args.use_pcl, args.use_canonical, training=True,
                                                  dataset_cfg=dataset_cfg)

            time.sleep(2)
            validation_loss = runner_3dFromImage(epoch, validation_loader, model, optimizer, None, device,
                                                 args.use_dataset, args.use_pcl, args.use_canonical, training=False,
                                                 dataset_cfg=dataset_cfg)

        print(f"MPJPE on Validation Dataset after Epoch {epoch} = {validation_loss:.4f}")

        torch.save({
            'epoch': epoch,
            'dataset': args.use_dataset,
            'exp_type': args.exp_type,
            'batch_size': args.batch_size,
            'validation_loss': validation_loss,
            'lowest_validation_loss': lowest_validation_loss,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, os.path.join(model_folder, "latest_validation.tar"))

        if validation_loss < lowest_validation_loss:
            lowest_validation_loss = validation_loss
            torch.save({
                'epoch': epoch,
                'dataset': args.use_dataset,
                'exp_type': args.exp_type,
                'batch_size': args.batch_size,
                'validation_loss': lowest_validation_loss,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, os.path.join(model_folder, "lowest_validation_model.tar"))

    print("Finished Training")
