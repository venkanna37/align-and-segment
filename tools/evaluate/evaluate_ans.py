import os
import json
import torch
import wandb
from tqdm import tqdm
from torchmetrics import JaccardIndex
from torch.utils.data import DataLoader
from huggingface_hub import snapshot_download
from segmentation_models_pytorch.losses import JaccardLoss
from segmentation_models_pytorch.losses.constants import BINARY_MODE

# custom imports
from tools.models.load_models import load_model
from tools.datagen.rebo_datagen import AlignDatagen
from tools.datagen.spacenet2 import AlignDatagen as AlignDatagen_Spacenet
from tools.models.stn import spatial_transformer_network
from tools.training.loss_functions import loss_for_seg


class AlignPrediction():
    def __init__(self, **kwargs):
        # basic parameters
        self.keyword = kwargs.get('keyword', 'test')

        # data parameters
        self.patch_size = kwargs.get('patch_size', 320)
        self.data_dir = kwargs.get('data_dir', './datasets')
        self.dataset_name = kwargs.get('dataset_name', 'sample_data')
        self.batch_size = kwargs.get('batch_size', 2)
        self.checkpoints_dir = kwargs.get('checkpoints_dir', './runs/')
        self.synth_method = kwargs.get('misalign_magnitude', 50)  # 1: Uniform
        self.set_name = kwargs.get('set_name', 'test')
        self.noise_type = kwargs.get('noise_type', 'u')
        self.num_workers = kwargs.get('num_workers', 4)

        # model parameters
        self.model_name = kwargs.get('model_name', 'method1')
        self.tnet_backbone = kwargs.get('tnet_backbone', 'vitsmall')
        self.weights_path = kwargs.get('weights_path', None)
        self.reg_loss_type = kwargs.get('reg_loss_type', 'mse')
        self.seg_loss_type = kwargs.get('seg_loss_type', 'cross_entropy')

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

    def evaluate(self):
        if not os.path.exists(self.data_dir):
            os.mkdir(self.data_dir)
        # fixme currently downloading entire city data here, downloading test set enough here
        if self.dataset_name == 'rebo':
            # downloading both training and test sets
            rebo_data_dir = os.path.join(self.data_dir, 'rebo')
            if not os.path.exists(rebo_data_dir):
                os.makedirs(rebo_data_dir, exist_ok=True)

            snapshot_download(repo_id='kevinlikai/ReBO', repo_type='dataset', local_dir=rebo_data_dir)
            test_set = AlignDatagen(rebo_data_dir,
                                    set_name=self.set_name,
                                    patch_size=512,
                                    )
        else:
            snapshot_download(repo_id='venkanna37/align-and-segment', repo_type='dataset',
                              allow_patterns=[f'{self.dataset_name}/**'], local_dir=self.data_dir)
            test_set = AlignDatagen_Spacenet(self.data_dir,
                                    set_name=self.set_name,
                                    dataset_name=self.dataset_name,
                                    synth_method=self.synth_method,
                                    patch_size=self.patch_size,
                                    noise_type=self.noise_type)
        data_loader_val = DataLoader(test_set,
                                     self.batch_size,
                                     drop_last=False,
                                     num_workers=self.num_workers,
                                     shuffle=False)

        # get model
        snet, tnet = load_model(self.model_name, self.tnet_backbone)
        model = torch.nn.Sequential(snet, tnet)

        # load weights, load complete model weights if exists, esle load decoder and tnet
        complete_weights = os.path.join(self.checkpoints_dir, self.keyword, 'best.pth')
        decoder_weights = os.path.join(self.checkpoints_dir, self.keyword, 'decoder.pth')
        tnet_weights = os.path.join(self.checkpoints_dir, self.keyword, 'tnet.pth')
        if os.path.exists(complete_weights):
            model_weights = torch.load(complete_weights, map_location=self.device)
            model.load_state_dict(model_weights['model'], strict=True)
        elif os.path.exists(decoder_weights) and os.path.exists(tnet_weights):
            decoder_weights = torch.load(decoder_weights, map_location=self.device)
            tnet_weights = torch.load(tnet_weights, map_location=self.device)
            model[0].load_decoder_state_dict(decoder_weights)
            model[1].load_state_dict(tnet_weights, strict=True)
        else:
            raise FileNotFoundError

        model.to(self.device)

        iou_pred_m = JaccardIndex(task="binary").to(self.device)
        iou_learn_m = JaccardIndex(task="binary").to(self.device)
        iou_align_m = JaccardIndex(task="binary").to(self.device)
        iou_org_m = JaccardIndex(task="binary").to(self.device)

        print("Start Evaluating")
        iou_pred_m.reset()
        iou_learn_m.reset()
        iou_align_m.reset()
        iou_org_m.reset()

        model.eval()
        dict_for_postfix = {}
        ce_loss_avg = 0

        pbar_val = tqdm(enumerate(data_loader_val), total=len(data_loader_val), desc="val")
        for i, val_batch in pbar_val:
            # Load data
            image = val_batch[0].to(self.device)
            mask = val_batch[2].to(self.device) if self.dataset_name != 'sanjuan' else val_batch[1].to(self.device)
            true_mask = val_batch[1].to(self.device)
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

                # get val metrics
                pred_mask, weight_mask = (pred_mask > 0).to(torch.uint8), weight_mask.to(torch.uint8)
                aligned_label = aligned_label.to(torch.uint8)
                iou_pred_m.update(pred_mask, true_mask)
                iou_learn_m.update(aligned_label, pred_mask * weight_mask)
                iou_align_m.update(aligned_label, true_mask * weight_mask)

                iou_org_m.update(mask, true_mask)

                iou_pred = iou_pred_m.compute().item()
                iou_learn = iou_learn_m.compute().item()
                iou_align = iou_align_m.compute().item()
                iou_org = iou_org_m.compute().item()
                dict_for_postfix["iou_pre"] = f'{iou_pred:.4f}'
                dict_for_postfix["iou_lea"] = f'{iou_learn:.4f}'
                dict_for_postfix["iou_aln"] = f'{iou_align:.4f}'
                dict_for_postfix["iou_org"] = f'{iou_org:.4f}'

                if torch.cuda.is_available():
                    peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
                    dict_for_postfix["pk_mem"] = f'{peak_memory:.1f} MB'
                    pbar_val.set_postfix(dict_for_postfix)
                else:
                    pbar_val.set_postfix(dict_for_postfix)

        # validation summary
        metrics = {
            "iou_seg": iou_pred,
            "iou_org": iou_org,
            "iou_learn": iou_learn,
            "iou_align": iou_align
        } if self.dataset_name != 'sanjuan' else {"iou_org": iou_org, "iou_learn": iou_learn}

        print("\n -----Validation summary-----")
        print(json.dumps(metrics, indent=4))
        print("-------------------------------")

        return metrics





