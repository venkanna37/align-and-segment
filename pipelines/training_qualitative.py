import os
import torch
import wandb
import random
random.seed(42)
from datetime import datetime
from tqdm import tqdm
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
        self.use_unet_aug = kwargs.get('use_unet_aug', False)
        self.noise_type = kwargs.get('noise_type', 'u')
        self.rescale_value = kwargs.get('rescale_value', 255)

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
        self.border_width = kwargs.get('border_width', 5)
        self.kernel_size = int(self.border_width * 2 + 1)

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
                                 noise_type= self.noise_type, rescale_value=self.rescale_value)
        data_loader_train = DataLoader(train_set, self.batch_size, drop_last=True, num_workers=4, shuffle=True)

        val_set = AlignDatagen(self.data_dir, sample_size=self.sample_size, set_name="val", city=self.city,
                                 single_index=self.single_index, synth_method=self.synth_method,
                                 patch_size=self.patch_size, hold_city=self.hold_city, noise_type= self.noise_type,
                               rescale_value=self.rescale_value)
        data_loader_val = DataLoader(val_set, self.batch_size, drop_last=False, num_workers=4, shuffle=False)

        # get the UNet model and initialize weights
        snet, tnet = load_model(self.model_name, self.tnet_backbone, self.device)

        # load weights to tnet
        if self.use_tnet_weights:
            if self.model_name == 'method1' or self.model_name == 'method2' or self.model_name == 'method3':
                if self.tnet_backbone == 'resnet34':
                    weights_path = "../../runs/challenge/ReS342/best.pth"
                elif self.tnet_backbone == 'vitsmall':
                    weights_path = "../../runs/challenge/ViTsmall/best.pth"
                model_weights = torch.load(weights_path, map_location=self.device)
                tnet.load_state_dict(model_weights['model'], strict=False)
            elif self.model_name == 'Method2' or self.model_name == 'Method3' or self.model_name == 'Method4':
                weights_path = "/home/mwv506/projects/ImageAlign/runs/challenge/ViT320_1k2/best.pth"
                model_weights = torch.load(weights_path, map_location=self.device)
                tnet.load_state_dict(model_weights['model'], strict=False)
            print(f"Loaded TNet weights from {weights_path} for the backbone {self.tnet_backbone}")
        else:
            # if weights not initialized, the transformation metric should be identity
            torch.nn.init.zeros_(tnet.fc.weight)

        model = torch.nn.Sequential(snet, tnet)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('Trainable params:', n_params)
        n_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        print('Non-trainable params:', n_params)

        model.to(self.device)
        optimizer = torch.optim.AdamW([
            {"params": model[0].parameters(), "lr": self.learning_rate},
            {"params": model[1].parameters(), "lr": self.learning_rate}
        ])

        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, self.lr_drop, gamma=0.8)
        iou_creterion = JaccardLoss(mode=BINARY_MODE, from_logits=False)

        if self.use_wb:
            if self.keyword != "test":  # to avoid multiple W&B projects for test runs
                wandb_project = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{self.keyword}"
            else:
                wandb_project = self.keyword
            print(f"Project name in weights and biases : {wandb_project}")
            writer = wandb.init(project=self.wb_project_name, name=wandb_project, dir=self.log_dir, config=self.kwargs)

        iou_learn_m = JaccardIndex(task="binary").to(self.device)
        iou_org_m = JaccardIndex(task="binary").to(self.device)

        best_iou = 0
        print("Start training")
        for epoch in range(self.epochs):
            iou_learn_m.reset()
            iou_org_m.reset()

            model.train()
            dict_for_postfix = {}
            tot_loss_avg, ce_loss_avg, aff_loss_avg, iou_loss_avg = 0, 0, 0, 0
            affin_params = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=self.device)

            pbar_train = tqdm(enumerate(data_loader_train), total=len(data_loader_train), desc="train")
            for i, train_batch in pbar_train:
                # Load data
                image = train_batch[0].to(self.device)
                mask = train_batch[1].to(self.device)           # true mask only for visualization
                weight_mask = torch.ones_like(mask)

                # forward pass
                if self.use_unet_aug:
                    image, mask = train_set.baseline_aug(image, mask, self.device)

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
                iou_learn_m.update(aligned_label, pred_mask * binary_weight_mask)
                iou_org_m.update(pred_mask, mask)

                iou_learn = iou_learn_m.compute().item()
                iou_org = iou_org_m.compute().item()
                dict_for_postfix["iou_lea"] = f'{iou_learn:.4f}'
                dict_for_postfix["iou_org"] = f'{iou_org:.4f}'

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
                            "train/iou_org": iou_org,
                            "epoch": epoch})
                if self.use_reg:
                    writer.log({"train/reg_loss": aff_loss_avg/len(data_loader_train),
                                "train/iou_loss": iou_loss_avg/len(data_loader_train),
                                "train/tot_loss": tot_loss_avg / len(data_loader_train),
                                "epoch": epoch})

            # Validation
            if self.do_val:
                iou_learn_m.reset()
                iou_org_m.reset()

                model.eval()
                dict_for_postfix = {}
                tot_loss_avg, ce_loss_avg, aff_loss_avg, iou_loss_avg = 0, 0, 0, 0
                affin_params = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=self.device)

                pbar_val = tqdm(enumerate(data_loader_val), total=len(data_loader_val), desc="val")
                for i, val_batch in pbar_val:
                    # Load data
                    image = val_batch[0].to(self.device)
                    mask = val_batch[1].to(self.device)
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
                        iou_learn_m.update(aligned_label, pred_mask * weight_mask)
                        iou_org_m.update(pred_mask, mask)

                        iou_learn = iou_learn_m.compute().item()
                        iou_org = iou_org_m.compute().item()
                        dict_for_postfix["iou_lea"] = f'{iou_learn:.4f}'
                        dict_for_postfix["iou_org"] = f'{iou_org:.4f}'

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
                                    "val/iou_org": iou_org,
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
                checkpoint_best_path = os.path.join(self.checkpoints_dir, 'best.pth')
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'metrics': dict_for_postfix,
                    'params': self.kwargs
                }, checkpoint_best_path)
