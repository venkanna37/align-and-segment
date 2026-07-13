"""
Training AnS with synthetic dataset
"""
import os
import torch
import wandb
from tqdm import tqdm
from datetime import datetime
from torchmetrics import JaccardIndex
from torch.utils.data import DataLoader
from segmentation_models_pytorch.losses import JaccardLoss
from segmentation_models_pytorch.losses.constants import BINARY_MODE

# custom imports
from pipelines.models.load_models import load_model
from pipelines.datagen.spacenet2 import AlignDatagen
from pipelines.models.spatial_transformer_network import spatial_transformer_network
from pipelines.utils.process_batch import transform_mask_with_random_affine
from pipelines.utils.matrices import inverse_affine_matrix, add_third_row
from .loss_functions import loss_for_seg


class AlignTraining:
    def __init__(self, **kwargs):

        # basic parameters
        self.keyword = kwargs.get('keyword', 'test')

        # data parameters
        self.patch_size = kwargs.get('patch_size', None)
        self.data_dir = kwargs.get('data_dir', None)
        self.batch_size = kwargs.get('batch_size', 2)
        self.checkpoints_dir = kwargs.get('checkpoints_dir', './checkpoints')
        self.checkpoints_org = self.checkpoints_dir
        self.log_dir = kwargs.get('log_dir', os.path.join(self.checkpoints_dir, 'logs'))
        self.checkpoints_dir = os.path.join(self.checkpoints_dir, self.keyword)
        if not os.path.exists(self.checkpoints_dir):
            os.makedirs(self.checkpoints_dir)
        self.sample_size = kwargs.get('sample_size', None)
        self.city = kwargs.get('city', None)
        self.single_index = kwargs.get('single_index', None)
        self.synth_method = kwargs.get('synth_method', 1)  # 1: Uniform
        self.aug_shift = kwargs.get('aug_shift', None)
        self.max_shift = kwargs.get('max_shift', 100)       # this is for reg_loss
        self.do_augh = kwargs.get('do_augh', False)
        self.use_snet_aug = kwargs.get('use_snet_aug', False)
        self.use_tnet_aug = kwargs.get('use_tnet_aug', False)
        self.noise_type = kwargs.get('noise_type', 'u')

        # model parameters
        self.model_name = kwargs.get('model_name', None)
        self.tnet_backbone = kwargs.get('tnet_backbone', 'vitsmall')
        self.use_tnet_weights = kwargs.get('use_tnet_weights', False)

        # train parameters
        self.snet_lr = kwargs.get('snet_lr', 0.00001)
        self.tnet_lr = kwargs.get('tnet_lr', 0.00001)
        self.epochs = kwargs.get('epochs', 300)
        self.lr_drop = kwargs.get('lr_drop', self.epochs)
        self.seg_loss_type = kwargs.get('seg_loss_type', 'cross_entropy')
        self.reg_loss_wt = kwargs.get('reg_loss_wt', 1)
        self.loss_setting = kwargs.get('loss_setting', None)
        self.warmup_epochs = kwargs.get('warmup_epochs', 0)
        self.num_workers = kwargs.get('num_workers', 4)
        self.use_reg_loss = kwargs.get('use_reg_loss', False)

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
            print(f"  {key:20s}: {value}")
        print("----------------------------- \n")

    def train(self):

        train_set = AlignDatagen(self.data_dir,
                                 sample_size=self.sample_size,
                                 set_name="train",
                                 city=self.city,
                                 single_index=self.single_index,
                                 synth_method=self.synth_method,
                                 aug_shift=self.aug_shift,
                                 patch_size=self.patch_size,
                                 noise_type= self.noise_type)
        data_loader_train = DataLoader(train_set,
                                       self.batch_size,
                                       drop_last=True,
                                       num_workers=self.num_workers,
                                       shuffle=True)

        val_set = AlignDatagen(self.data_dir,
                               sample_size=self.sample_size,
                               set_name="val",
                               city=self.city,
                               single_index=self.single_index,
                               synth_method=self.synth_method,
                               patch_size=self.patch_size,
                               noise_type= self.noise_type)
        data_loader_val = DataLoader(val_set,
                                     self.batch_size,
                                     drop_last=False,
                                     num_workers=self.num_workers,
                                     shuffle=False)

        # get the UNet model and initialize weights
        snet, tnet = load_model(self.model_name, self.tnet_backbone, self.device)
        model = torch.nn.Sequential(snet, tnet)
        model.to(self.device)

        # load pretrained TNet weights
        if self.use_tnet_weights:
            if self.noise_type == 'b':
                weights_path = os.path.join(self.checkpoints_org, 'tnet_b_3e1', 'best_val.pth')
            elif self.noise_type == 'u':
                weights_path = os.path.join(self.checkpoints_org, 'tnet_u_3e1', 'best_val.pth')
            else:
                raise NotImplementedError
            checkpoint = torch.load(weights_path, map_location=self.device)
            model[1].load_state_dict(checkpoint['model'], strict=True)
            for param in model[1].parameters():
                param.requires_grad = False
        else:
            torch.nn.init.zeros_(tnet.fc.weight)

        # trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        # optimizer = torch.optim.AdamW(trainable_params, lr=self.learning_rate)
        optimizer = torch.optim.AdamW([
            {"params": model[0].parameters(), "lr": self.snet_lr},
            {"params": model[1].parameters(), "lr": self.tnet_lr}
        ])
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, self.lr_drop, gamma=0.8)
        best_iou = 0
        best_iou_gold = 0
        start_epoch = 0

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('Trainable params:', n_params)
        n_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        print('Non-trainable params:', n_params)
        n_params = sum(p.numel() for p in model[1].parameters())
        print('TNet params:', n_params)
        n_params = sum(p.numel() for p in model[0].parameters())
        print('SNet params:', n_params)

        iou_criterion = JaccardLoss(mode=BINARY_MODE, from_logits=False)

        # model, training and data details, can be modified here or using command line arguments
        params = {
            'wb_project_name': 'AnS'
        }
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

                seg_loss = loss_for_seg(pred_mask, aligned_label, weight_mask, loss_type=self.seg_loss_type)
                ce_loss_avg += seg_loss.item()
                dict_for_postfix["seg_ls"] = f'{ce_loss_avg / (i + 1):.4f}'

                if self.use_reg_loss:
                    aug_mask1, g1 = transform_mask_with_random_affine(mask, self.device, max_shift=self.max_shift)
                    input_tnet1 = torch.cat((mask, aug_mask1), dim=1)
                    # input_tnet1, g1 = train_set.create_aug_data(input_tnet1, g1, self.device)
                    g1_inv = inverse_affine_matrix(g1)
                    t1, _ = model[1](input_tnet1)
                    loss1_squares = (g1_inv - t1) ** 2

                    if self.loss_setting == 1:
                        aff_loss = loss1_squares.mean() * self.reg_loss_wt
                    else:
                        aug_mask2, g2 = transform_mask_with_random_affine(mask, self.device, max_shift=self.max_shift)
                        input_tnet2 = torch.cat((mask, aug_mask2), dim=1)
                        # input_tnet2, g2 = train_set.create_aug_data(input_tnet2, g2, self.device)
                        t2, _ = model[1](input_tnet2)
                        first_part = add_third_row(t1) @ add_third_row(g1)
                        second_part = add_third_row(t2) @ add_third_row(g2)
                        loss2_squares = (first_part - second_part) ** 2

                        if self.loss_setting == 2:
                            aff_loss = loss2_squares.mean() * self.reg_loss_wt

                        elif self.loss_setting == 3:
                            aff_loss = (loss1_squares.mean() + loss2_squares.mean()) * self.reg_loss_wt

                        elif self.loss_setting in [4, 5, 6]:
                            reg_weight_mask = torch.ones_like(aug_mask1, device=self.device)
                            reg_weight_mask = (spatial_transformer_network(reg_weight_mask, t1.detach()) > 0).float()
                            reg_aligned_mask = spatial_transformer_network(input_tnet1[:, [1]], t1)
                            iou_loss = iou_criterion(reg_aligned_mask, input_tnet1[:, [0]] * reg_weight_mask)
                            iou_loss_avg += iou_loss.item()
                            dict_for_postfix["iou_ls"] = f'{iou_loss_avg / (i + 1):.4f}'

                            if self.loss_setting == 4:
                                aff_loss = loss1_squares.mean() * self.reg_loss_wt
                            elif self.loss_setting == 5:
                                aff_loss = loss2_squares.mean() * self.reg_loss_wt
                            elif self.loss_setting == 6:
                                aff_loss = torch.tensor(0.0, device=self.device)
                        else:
                            raise NotImplementedError

                    if self.loss_setting == 6:
                        loss = iou_loss
                    elif self.loss_setting in [4, 5]:
                        loss = iou_loss
                        loss += aff_loss
                    else:
                        loss = aff_loss

                else:
                    aff_loss = torch.tensor(0.0, device=self.device)
                    loss = aff_loss

                aff_loss_avg += aff_loss.item()
                dict_for_postfix["reg_ls"] = f'{aff_loss_avg / (i + 1):.4f}'

                if epoch >= self.warmup_epochs:
                    loss += seg_loss

                tot_loss_avg += loss.item()
                dict_for_postfix["tot_ls"] = f'{tot_loss_avg / (i + 1):.4f}'


                # backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # get metrics
                pred_mask, binary_weight_mask = (pred_mask > 0).to(torch.uint8), (weight_mask > 0).to(torch.uint8)
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
            pbar_train.close()

            # training summary
            if self.use_wb:
                e_seg_loss = ce_loss_avg / len(data_loader_train)
                e_aff_loss = aff_loss_avg / len(data_loader_train)
                e_iou_loss = iou_loss_avg / len(data_loader_train)

                writer.log({"train/seg_loss": e_seg_loss,
                            "train/aff_loss": e_aff_loss,
                            "train/iou_loss": e_iou_loss,
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

            # Validation
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
                    weight_mask = (spatial_transformer_network(weight_mask, params.detach()) > 0).float()

                    seg_loss = loss_for_seg(pred_mask, aligned_label, weight_mask, loss_type=self.seg_loss_type)
                    ce_loss_avg += seg_loss.item()
                    dict_for_postfix["seg_ls"] = f'{ce_loss_avg / (i + 1):.4f}'

                    aug_mask1, g1 = transform_mask_with_random_affine(mask, self.device, max_shift=self.max_shift)
                    g1_inv = inverse_affine_matrix(g1)
                    input_tnet1 = torch.cat((mask, aug_mask1), dim=1)
                    t1, _ = model[1](input_tnet1)
                    loss1_squares = (g1_inv - t1) ** 2

                    if self.loss_setting == 1:
                        aff_loss = loss1_squares.mean() * self.reg_loss_wt

                    else:
                        aug_mask2, g2 = transform_mask_with_random_affine(mask, self.device,
                                                                          max_shift=self.max_shift)
                        input_tnet2 = torch.cat((mask, aug_mask2), dim=1)
                        t2, _ = model[1](input_tnet2)
                        first_part = add_third_row(t1) @ add_third_row(g1)
                        second_part = add_third_row(t2) @ add_third_row(g2)
                        loss2_squares = (first_part - second_part) ** 2

                        if self.loss_setting == 2:
                            aff_loss = loss2_squares.mean() * self.reg_loss_wt

                        elif self.loss_setting == 3:
                            aff_loss = (loss1_squares.mean() + loss2_squares.mean()) * self.reg_loss_wt

                        elif self.loss_setting in [4, 5, 6]:
                            reg_weight_mask = torch.ones_like(aug_mask1, device=self.device)
                            reg_weight_mask = (
                                        spatial_transformer_network(reg_weight_mask, t1.detach()) > 0).float()
                            reg_aligned_mask = spatial_transformer_network(input_tnet1[:, [1]], t1)
                            iou_loss = iou_criterion(reg_aligned_mask, input_tnet1[:, [0]] * reg_weight_mask)
                            iou_loss_avg += iou_loss.item()
                            dict_for_postfix["iou_ls"] = f'{iou_loss_avg / (i + 1):.4f}'

                            if self.loss_setting == 4:
                                aff_loss = loss1_squares.mean() * self.reg_loss_wt
                            elif self.loss_setting == 5:
                                aff_loss = loss2_squares.mean() * self.reg_loss_wt
                            elif self.loss_setting == 6:
                                aff_loss = torch.tensor(0.0, device=self.device)

                        elif self.loss_setting == 7:
                            aff_loss = torch.tensor(0.0, device=self.device)
                        else:
                            raise NotImplementedError


                    if self.loss_setting == 6:
                        loss = iou_loss
                    elif self.loss_setting in [4, 5]:
                        loss = iou_loss
                        loss += aff_loss
                    else:
                        loss = aff_loss

                    aff_loss_avg += aff_loss.item()
                    dict_for_postfix["reg_ls"] = f'{aff_loss_avg / (i + 1):.4f}'

                    if epoch >= self.warmup_epochs:
                        loss += seg_loss

                    tot_loss_avg += loss.item()
                    dict_for_postfix["tot_ls"] = f'{tot_loss_avg / (i + 1):.4f}'

                    # get val metrics
                    pred_mask, weight_mask = (pred_mask > 0).to(torch.uint8), (weight_mask > 0).to(torch.uint8)
                    aligned_label = aligned_label.to(torch.uint8)
                    iou_seg_m.update(pred_mask, true_mask)
                    iou_learn_m.update(aligned_label, pred_mask * weight_mask)
                    new_affine = add_third_row(true_affine) @ add_third_row(params.detach())
                    new_true_mask = spatial_transformer_network(true_mask, new_affine[:, :2, :])
                    new_true_mask = new_true_mask > 0
                    iou_align_m.update(new_true_mask, true_mask)
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
                e_seg_loss = ce_loss_avg / len(data_loader_val)
                e_aff_loss = aff_loss_avg / len(data_loader_val)
                e_iou_loss = iou_loss_avg / len(data_loader_val)

                writer.log({"val/seg_loss": e_seg_loss,
                            "val/aff_loss": e_aff_loss,
                            "val/iou_loss": e_iou_loss,
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
            pbar_val.close()

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
