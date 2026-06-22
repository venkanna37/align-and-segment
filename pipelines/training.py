import os
import torch
import wandb
import random
random.seed(42)
import numpy as np
from datetime import datetime
from tqdm import tqdm
from torchmetrics import JaccardIndex
from torch.utils.data import DataLoader
from segmentation_models_pytorch.losses import JaccardLoss
from segmentation_models_pytorch.losses.constants import BINARY_MODE

# custom imports
from .model import load_model, spatial_transformer_network
from .datagen import AlignDatagen
from .utils import transform_mask_with_random_affine, inverse_affine_matrix, add_third_row
from .loss_functions import loss_for_align, loss_for_seg


class AlignTraining:
    def __init__(self, **kwargs):

        # basic parameters
        self.keyword = kwargs.get('keyword', 'test')

        # data parameters
        self.patch_size = kwargs.get('patch_size', None)
        self.data_dir = kwargs.get('data_dir', None)
        self.batch_size = kwargs.get('batch_size', 2)
        self.checkpoints_dir = kwargs.get('checkpoints_dir', './checkpoints')
        self.log_dir = kwargs.get('log_dir', os.path.join(self.checkpoints_dir, 'logs'))
        self.checkpoints_dir = os.path.join(self.checkpoints_dir, self.keyword)
        if not os.path.exists(self.checkpoints_dir):
            os.makedirs(self.checkpoints_dir)
        self.sample_size = kwargs.get('sample_size', None)
        self.city = kwargs.get('city', None)
        self.hold_city = kwargs.get('hold_city', None)
        self.single_index = kwargs.get('single_index', None)
        self.synth_method = kwargs.get('synth_method', 1)  # 1: Uniform
        self.aug_shift = kwargs.get('aug_shift', None)
        self.max_shift = kwargs.get('max_shift', 50)       # this is for reg_loss
        self.do_augh = kwargs.get('do_augh', False)
        self.use_snet_aug = kwargs.get('use_snet_aug', False)
        self.use_tnet_aug = kwargs.get('use_tnet_aug', False)
        self.noise_type = kwargs.get('noise_type', 'u')

        # model parameters
        self.model_name = kwargs.get('model_name', 'ce')
        self.tnet_backbone = kwargs.get('tnet_backbone', 'vitsmall')
        self.use_tnet_weights = kwargs.get('use_tnet_weights', False)

        # train parameters
        self.do_val = kwargs.get('do_val', False)
        self.learning_rate = kwargs.get('learning_rate', 0.0001)
        self.epochs = kwargs.get('epochs', 300)
        self.lr_drop = kwargs.get('lr_drop', self.epochs)
        self.reg_loss_type = kwargs.get('reg_loss_type', 'mse')
        self.seg_loss_type = kwargs.get('seg_loss_type', 'cross_entropy')
        self.reg_loss_wt = kwargs.get('reg_loss_wt', 1)
        self.use_reg = kwargs.get('use_reg', False)
        self.pre_weights = kwargs.get('pre_weights', None)

        # visualization parameters
        self.use_wb = kwargs.get('use_wb', False)
        self.wb_project_name = kwargs.get('wb_project_name', 'Align')
        self.write_images = kwargs.get('write_images', False)


        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.kwargs = kwargs
        # print the parameters as a dictionary
        print("\n -----Training parameters-----")
        for key, value in self.kwargs.items():
            print(f"{key}: {value}")
        print("----------------------------- \n")

    def train(self):

        train_set = AlignDatagen(self.data_dir, sample_size=self.sample_size, set_name="train", city=self.city,
                                 single_index=self.single_index, synth_method=self.synth_method,
                                 aug_shift=self.aug_shift, patch_size=self.patch_size, hold_city=self.hold_city,
                                 noise_type= self.noise_type)
        data_loader_train = DataLoader(train_set, self.batch_size, drop_last=True, num_workers=4, shuffle=True)

        val_set = AlignDatagen(self.data_dir, sample_size=self.sample_size, set_name="val", city=self.city,
                                 single_index=self.single_index, synth_method=self.synth_method,
                                 patch_size=self.patch_size, hold_city=self.hold_city, noise_type= self.noise_type)
        data_loader_val = DataLoader(val_set, self.batch_size, drop_last=False, num_workers=4, shuffle=False)

        # get the UNet model and initialize weights
        snet, tnet = load_model()
        torch.nn.init.zeros_(tnet.fc.weight)

        model = torch.nn.Sequential(snet, tnet)
        model.to(self.device)
        optimizer = torch.optim.AdamW([
            {"params": model[0].parameters(), "lr": self.learning_rate},
            {"params": model[1].parameters(), "lr": self.learning_rate}
        ])
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, self.lr_drop, gamma=0.8)
        best_iou = 0
        best_iou_gold = 0
        start_epoch = 0

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('Trainable params:', n_params)
        n_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        print('Non-trainable params:', n_params)

        iou_creterion = JaccardLoss(mode=BINARY_MODE, from_logits=False)

        if self.use_wb:
            if self.keyword != "test":  # to avoid multiple W&B projects for test runs
                wandb_project = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{self.keyword}"
            else:
                wandb_project = self.keyword
            print(f"Project name in weights and biases : {wandb_project}")
            writer = wandb.init(project=self.wb_project_name, name=wandb_project, dir=self.log_dir, config=self.kwargs)

        iou_seg_m = JaccardIndex(task="binary").to(self.device)
        iou_learn_m = JaccardIndex(task="binary").to(self.device)
        iou_align_m = JaccardIndex(task="binary").to(self.device)
        iou_org_m = JaccardIndex(task="binary").to(self.device)

        # freeze_epoch = self.epochs + 1
        print("Start training")
        for epoch in range(start_epoch, self.epochs):
            iou_seg_m.reset()
            iou_learn_m.reset()
            iou_align_m.reset()
            iou_org_m.reset()

            model.train()
            dict_for_postfix = {}
            tot_loss_avg, ce_loss_avg, aff_loss_avg, iou_loss_avg = 0, 0, 0, 0
            affin_params = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=self.device)

            pbar_train = tqdm(enumerate(data_loader_train), total=len(data_loader_train), desc="train")
            for i, train_batch in pbar_train:
                # Load data
                image = train_batch[0].to(self.device)
                true_mask = train_batch[1].to(self.device)           # true mask only for visualization
                true_affine = train_batch[3].to(self.device)         # ground truth affine matrix for visualization
                mask = train_batch[2].to(self.device)
                weight_mask = torch.ones_like(mask)

                # forward pass
                if self.use_snet_aug:
                    image, mask, true_mask, true_affine = train_set.aug_for_unet(image, mask, true_mask, self.device, true_affine)

                pred_mask = model[0](image)
                input_tnet = torch.cat((mask, (pred_mask > 0).float()), dim=1)
                params, _ = model[1](input_tnet)

                aligned_label = spatial_transformer_network(mask, params)
                weight_mask = (spatial_transformer_network(weight_mask, params.detach()) > 0).float()

                loss = loss_for_seg(pred_mask, aligned_label, weight_mask, loss_type=self.seg_loss_type)
                ce_loss_avg += loss.item()
                dict_for_postfix["seg_ls"] = f'{ce_loss_avg / (i + 1):.4f}'


                if self.use_reg:
                    aug_mask, aug_affine = transform_mask_with_random_affine(mask, self.device, max_shift=self.max_shift)
                    input_tnet2 = torch.cat((mask, aug_mask), dim=1)
                    if self.use_tnet_aug:
                        input_tnet2, aug_affine = train_set.create_aug_data(input_tnet2, aug_affine, self.device)

                    pred_affine, _ = model[1](input_tnet2)
                    inv_aug_aff = inverse_affine_matrix(aug_affine)
                    aff_loss = loss_for_align(pred_affine, inv_aug_aff, device=self.device, loss_type=self.reg_loss_type) * self.reg_loss_wt
                    loss += aff_loss

                    aff_loss_avg += aff_loss.item()
                    dict_for_postfix["reg_ls"] = f'{aff_loss_avg / (i + 1):.4f}'

                    # iou loss
                    reg_weight_mask = torch.ones_like(aug_mask, device=self.device)
                    reg_weight_mask = (spatial_transformer_network(reg_weight_mask, pred_affine.detach()) > 0).float()
                    reg_aligned_mask = spatial_transformer_network(input_tnet2[:, [1]], pred_affine)
                    iou_loss = iou_creterion(reg_aligned_mask, input_tnet2[:, [0]] * reg_weight_mask)
                    iou_loss_avg += iou_loss.item()
                    dict_for_postfix["iou_ls"] = f'{iou_loss_avg / (i + 1):.4f}'
                    loss += iou_loss

                    tot_loss_avg += loss.item()
                    dict_for_postfix["tot_ls"] = f'{tot_loss_avg / (i + 1):.4f}'

                # backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # get metrics
                pred_mask, binary_weight_mask = (pred_mask > 0).to(torch.uint8), weight_mask.to(torch.uint8)
                aligned_label = aligned_label.to(torch.uint8)
                iou_seg_m.update(pred_mask, true_mask)
                iou_learn_m.update(aligned_label, pred_mask * binary_weight_mask)
                iou_org_m.update(pred_mask, mask)

                err_affine = add_third_row(true_affine) @ add_third_row(params.detach())
                err_true_mask = spatial_transformer_network(true_mask, err_affine[:, :2, :])
                err_true_mask = (err_true_mask > 0).float()
                iou_align_m.update(err_true_mask, true_mask)

                iou_seg = iou_seg_m.compute().item()
                iou_learn = iou_learn_m.compute().item()
                iou_align = iou_align_m.compute().item()
                iou_org = iou_org_m.compute().item()
                dict_for_postfix["iou_seg"] = f'{iou_seg:.4f}'
                dict_for_postfix["iou_lea"] = f'{iou_learn:.4f}'
                dict_for_postfix["iou_evl"] = f'{iou_align:.4f}'
                dict_for_postfix["iou_org"] = f'{iou_org:.4f}'

                # get difference between predicted affine matrix and ground truth, flatten it and add to the list
                inv_true_affine = inverse_affine_matrix(true_affine)
                mean_difference = torch.flatten(torch.abs(params.detach() - inv_true_affine), start_dim=1).mean(dim=0)
                affin_params += mean_difference

                if torch.cuda.is_available():
                    peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
                    dict_for_postfix["pk_mem"] = f'{peak_memory:.1f} MB'
                    pbar_train.set_postfix(dict_for_postfix, epoch=f'{epoch + 1}/{self.epochs}')
                else:
                    pbar_train.set_postfix(dict_for_postfix, epoch=f'{epoch + 1}/{self.epochs}')
            pbar_train.reset()

            # training summary
            epoch_loss = ce_loss_avg/len(data_loader_train)
            if self.use_wb:
                writer.log({"train/seg_loss": epoch_loss,
                            "train/iou_learn": iou_learn,
                            "train/iou_seg": iou_seg,
                            "train/iou_align": iou_align,
                            "train/iou_org": iou_org,
                            "epoch": epoch})
                # visualize affine parameters
                affine_params = (affin_params / len(data_loader_train)).cpu().numpy().tolist()
                writer.log({
                        "train/scale_x": affine_params[0],
                        "train/scale_y": affine_params[4],
                        "train/rotate_x": affine_params[1],
                        "train/rotate_y": affine_params[3],
                        "train/translation_x": affine_params[2],
                        "train/translation_y": affine_params[5],
                        "epoch": epoch})
                if self.use_reg:
                    writer.log({"train/reg_loss": aff_loss_avg/len(data_loader_train),
                                "train/iou_loss": iou_loss_avg/len(data_loader_train),
                                "train/tot_loss": tot_loss_avg / len(data_loader_train),
                                "epoch": epoch})
                if self.write_images:
                    b_, c_, h_, w_ = image.shape
                    if c_ > 3:  # first three channels considered as rgb
                        image = image[:, :3]  # if image channels are more than three
                    indices = np.random.choice(self.batch_size, min(self.batch_size, 4), replace=False)
                    cls_labels = {0: "background", 1: "building"}
                    weight_labels = {0: "background", 1: "data_weight"}

                    mask_list = []
                    for idx in indices:
                        wb_masks = {
                            "GT_Mask": {"mask_data": mask[idx][0].cpu().numpy(),
                                                  "class_labels": cls_labels},
                            "GT_True": {"mask_data": true_mask[idx][0].cpu().numpy(),
                                                  "class_labels": cls_labels},
                            "PD_Predict": {"mask_data": pred_mask[idx][0].cpu().numpy(),
                                                "class_labels": cls_labels},
                            "PD_Aligned": {"mask_data": aligned_label[idx][0].detach().cpu().numpy(),
                                                "class_labels": cls_labels},
                            "PD_Weight": {"mask_data": weight_mask[idx][0].detach().cpu().numpy(),
                                                "class_labels": weight_labels}
                        }
                        image_ = (image[idx].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                        mask_list.append(wandb.Image(image_, masks=wb_masks))
                    writer.log({
                        f"train/predictions":mask_list, "epoch": epoch})

            # Validation
            if self.do_val:
                iou_learn_m.reset()
                iou_seg_m.reset()
                iou_align_m.reset()
                iou_org_m.reset()

                model.eval()
                dict_for_postfix = {}
                tot_loss_avg, ce_loss_avg, aff_loss_avg, iou_loss_avg = 0, 0, 0, 0
                affin_params = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=self.device)

                pbar_val = tqdm(enumerate(data_loader_val), total=len(data_loader_val), desc="val")
                for i, val_batch in pbar_val:
                    # Load data
                    image = val_batch[0].to(self.device)
                    true_mask = val_batch[1].to(self.device)
                    true_affine = val_batch[3].to(self.device)
                    mask = val_batch[2].to(self.device)
                    weight_mask = torch.ones_like(mask)

                    # forward pass
                    with torch.no_grad():
                        pred_mask = model[0](image)
                        input_tnet = torch.cat((mask, (pred_mask > 0).float()), dim=1)
                        params, _ = model[1](input_tnet)
                        aligned_label = spatial_transformer_network(mask, params)
                        weight_mask = spatial_transformer_network(weight_mask, params.detach())
                        loss = loss_for_seg(pred_mask, aligned_label, weight_mask, loss_type=self.seg_loss_type)
                        ce_loss_avg += loss.item()
                        dict_for_postfix["seg_ls"] = f'{ce_loss_avg / (i + 1):.4f}'


                        if self.use_reg:
                            aug_mask, aug_affine = transform_mask_with_random_affine(mask, self.device,
                                                                                     max_shift=self.max_shift)
                            input_tnet2 = torch.cat((mask, aug_mask), dim=1)
                            input_tnet2, aug_affine = train_set.create_aug_data(input_tnet2, aug_affine, self.device)
                            pred_affine, _ = model[1](input_tnet2)
                            inv_aug_aff = inverse_affine_matrix(aug_affine)
                            aff_loss = loss_for_align(pred_affine, inv_aug_aff, device=self.device,
                                                      loss_type=self.reg_loss_type)
                            aff_loss_avg += aff_loss.item()
                            dict_for_postfix["reg_ls"] = f'{aff_loss_avg / (i + 1):.4f}'
                            loss += aff_loss

                            # iou loss
                            reg_weight_mask = torch.ones_like(aug_mask, device=self.device)
                            reg_weight_mask = (
                                        spatial_transformer_network(reg_weight_mask, pred_affine.detach()) > 0).float()
                            reg_aligned_mask = spatial_transformer_network(input_tnet2[:, [1]], pred_affine)
                            iou_loss = iou_creterion(reg_aligned_mask, input_tnet2[:, [0]] * reg_weight_mask)
                            iou_loss_avg += iou_loss.item()
                            dict_for_postfix["iou_ls"] = f'{iou_loss_avg / (i + 1):.4f}'
                            loss += iou_loss

                            tot_loss_avg += loss.item()
                            dict_for_postfix["tot_ls"] = f'{tot_loss_avg / (i + 1):.4f}'

                        # get val metrics
                        pred_mask, weight_mask = (pred_mask > 0).to(torch.uint8), weight_mask.to(torch.uint8)
                        aligned_label = aligned_label.to(torch.uint8)
                        iou_seg_m.update(pred_mask, true_mask)
                        iou_learn_m.update(aligned_label, pred_mask * weight_mask)
                        new_affine = add_third_row(true_affine) @ add_third_row(params.detach())
                        new_true_mask = spatial_transformer_network(true_mask, new_affine[:, :2, :])
                        new_true_mask = new_true_mask > 0
                        iou_align_m.update(true_mask, new_true_mask)
                        iou_org_m.update(pred_mask, mask)

                        iou_seg = iou_seg_m.compute().item()
                        iou_learn = iou_learn_m.compute().item()
                        iou_align = iou_align_m.compute().item()
                        iou_org = iou_org_m.compute().item()
                        dict_for_postfix["iou_seg"] = f'{iou_seg:.4f}'
                        dict_for_postfix["iou_lea"] = f'{iou_learn:.4f}'
                        dict_for_postfix["iou_evl"] = f'{iou_align:.4f}'
                        dict_for_postfix["iou_org"] = f'{iou_org:.4f}'

                        # get difference between predicted affine matrix and ground truth, flatten it and add to the list
                        inv_true_affine = inverse_affine_matrix(true_affine)
                        mean_difference = torch.flatten(torch.abs(params.detach() - inv_true_affine), start_dim=1).mean(
                            dim=0)
                        affin_params += mean_difference

                        if torch.cuda.is_available():
                            peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
                            dict_for_postfix["pk_mem"] = f'{peak_memory:.1f} MB'
                            pbar_val.set_postfix(dict_for_postfix, epoch=f'{epoch + 1}/{self.epochs}')
                        else:
                            pbar_val.set_postfix(dict_for_postfix, epoch=f'{epoch + 1}/{self.epochs}')

                    # validation summary
                    if self.use_wb:
                        writer.log({"val/seg_loss": ce_loss_avg / len(data_loader_val),
                                    "val/iou_learn": iou_learn,
                                    "val/iou_seg": iou_seg,
                                    "val/iou_align": iou_align,
                                    "val/iou_org": iou_org,
                                    "epoch": epoch})
                        # visualize affine parameters
                        affine_params = (affin_params / len(data_loader_val)).cpu().numpy().tolist()
                        writer.log({
                            "val/scale_x": affine_params[0],
                            "val/scale_y": affine_params[4],
                            "val/rotate_x": affine_params[1],
                            "val/rotate_y": affine_params[3],
                            "val/translation_x": affine_params[2],
                            "val/translation_y": affine_params[5],
                            "epoch": epoch})
                        if self.use_reg:
                            writer.log({"val/reg_loss": aff_loss_avg / len(data_loader_val),
                                        "val/iou_loss": iou_loss_avg / len(data_loader_val),
                                        "val/total_loss": tot_loss_avg / len(data_loader_val),
                                        "epoch": epoch})
                pbar_val.reset()

            # change lr according to the scheduler
            lr_scheduler.step()

            # save the latest weights
            latest_filename = 'latest.pth'
            checkpoint_latest_path = os.path.join(self.checkpoints_dir, latest_filename)
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'metrics': dict_for_postfix,
                'params': self.kwargs
            }, checkpoint_latest_path)

            # save best model with score
            if iou_learn > best_iou:
                best_iou = iou_learn
                print(f"Best model found at epoch {epoch} with with IOU_learn score: {round(best_iou, 5)}")
                best_file = 'best.pth' if epoch < 200 else 'best300.pth'
                checkpoint_best_path = os.path.join(self.checkpoints_dir, best_file)
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'metrics': dict_for_postfix,
                    'params': self.kwargs
                }, checkpoint_best_path)

            # save best model with based on golden label on validation set
            if iou_align > best_iou_gold:
                best_iou_gold = iou_align
                print(f"Best model found at epoch {epoch} with with IOU_Align score: {round(best_iou_gold, 5)}")
                checkpoint_best_path = os.path.join(self.checkpoints_dir, 'best_gold.pth')
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'metrics': dict_for_postfix,
                    'params': self.kwargs
                }, checkpoint_best_path)
