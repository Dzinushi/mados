# -*- coding: utf-8 -*-
import os
import sys
import random
import logging
import rasterio
from rasterio.enums import Resampling
import argparse
from glob import glob
import numpy as np
from tqdm import tqdm
from os.path import dirname as up

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.nn import functional as F

sys.path.append(up(os.path.abspath(__file__)))
from marinext_wrapper import MariNext

sys.path.append(os.path.join(up(up(os.path.abspath(__file__))), 'utils'))
from dataset import MADOS, bands_mean, bands_std
from test_time_aug import TTA
from metrics import Evaluation, confusion_matrix
from assets import labels, bool_flag

root_path = up(up(os.path.abspath(__file__)))
logging.basicConfig(filename=os.path.join(root_path, 'logs', 'evaluating_pruned.log'), filemode='a', level=logging.INFO,
                    format='%(name)s - %(levelname)s - %(message)s')
logging.info('*' * 10)


def seed_all(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_band(path):
    return int(path.split('_')[-2])


class ZeroLayer(nn.Module):
    def forward(self, x):
        return torch.zeros_like(x)


def apply_dynamic_pruning(model: nn.Module, threshold: float):
    print(f"Applying dynamic pruning with threshold = {threshold}...")
    pruned_count = 0
    for name, module in model.named_modules():
        if hasattr(module, 'mlp') and hasattr(module.mlp, 'act') and hasattr(module.mlp.act, 'r1'):
            r1 = abs(module.mlp.act.r1.item())
            r2 = abs(module.mlp.act.r2.item())

            if max(r1, r2) < threshold:
                module.mlp = ZeroLayer()
                pruned_count += 1
                print(f"  -> Pruned: {name}.mlp (r1={r1:.2e}, r2={r2:.2e})")

    print(f"Total pruned MLP modules: {pruned_count}")
    return model


def main(options):
    seed_all(0)
    transform_test = transforms.Compose([transforms.ToTensor()])
    standardization = transforms.Normalize(bands_mean, bands_std)

    splits_path = os.path.join(options['path'], 'splits')
    dataset_test = MADOS(options['path'], splits_path, options['split'])

    test_loader = DataLoader(dataset_test, batch_size=options['batch'],
                             num_workers=options['num_workers'], pin_memory=options['pin_memory'],
                             prefetch_factor=options['prefetch_factor'],
                             persistent_workers=options['persistent_workers'], shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models_list = []
    models_files = glob(os.path.join(options['model_path'], '*.pth'))

    for model_file in models_files:
        model = MariNext(options['input_channels'], options['output_channels'], options['config'])
        model.to(device)

        logging.info('Loading model files from folder: %s' % model_file)
        checkpoint = torch.load(model_file, map_location=device)
        checkpoint = {k.replace('decoder', 'decode_head'): v for k, v in checkpoint.items() if
                      ('proj1' not in k) and ('proj2' not in k)}

        model.load_state_dict(checkpoint, strict=False)
        apply_dynamic_pruning(model, options['prune_threshold'])

        del checkpoint
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        model.eval()
        models_list.append(model)

        if options['save_pruned_model']:
            pruned_path = model_file.replace('.pth', '_pruned.pth')
            torch.save(model.state_dict(), pruned_path)
            print(f"Saved pruned model to: {pruned_path}")

    y_true = []
    y_predicted = []

    with torch.no_grad():
        for (image, target) in tqdm(test_loader, desc="testing"):
            if options['test_time_augmentations'] and options['batch'] == 1:
                image = TTA(image)

            image = image.to(device)
            target = target.to(device)
            seed_all(0)

            all_predictions = []
            for model in models_list:
                logits = model(image)
                logits = F.upsample(input=logits, size=(target.shape[-2], target.shape[-1]), mode='bilinear')
                probs = torch.nn.functional.softmax(logits, dim=1)
                predictions = probs.argmax(1)

                if options['test_time_augmentations'] and options['batch'] == 1:
                    predictions = TTA(predictions, reverse_aggregation=True)
                all_predictions.append(predictions)

            all_predictions = torch.cat(all_predictions)
            all_predictions = torch.mode(all_predictions, dim=0, keepdim=True)[0]

            predictions = predictions.reshape(-1)
            target = target.reshape(-1)
            mask = target != -1

            predictions = predictions[mask].cpu().numpy()
            target = target[mask].cpu().numpy()

            y_predicted += predictions.tolist()
            y_true += target.tolist()

        acc = Evaluation(y_predicted, y_true)
        logging.info("\nSTATISTICS:\nEvaluation: " + str(acc))
        print("Evaluation: " + str(acc))
        conf_mat = confusion_matrix(y_true, y_predicted, labels, options['results_percentage'])
        logging.info("Confusion Matrix:\n" + str(conf_mat.to_string()))
        print("Confusion Matrix:\n" + str(conf_mat.to_string()))

        seed_all(0)

        if options['predict_masks']:
            path = options['path']
            tiles = glob(os.path.join(path, '*'))
            ROIs_split = np.genfromtxt(os.path.join(splits_path, options['split'] + '_X.txt'), dtype='str')

            impute_nan = np.tile(bands_mean, (240, 240, 1))

            for tile in tqdm(tiles, desc='testing'):
                splits = [f.split('_cl_')[-1] for f in glob(os.path.join(tile, '10', '*_cl_*'))]

                for crop in splits:
                    crop_name = os.path.basename(tile) + '_' + crop.split('.tif')[0]

                    if crop_name in ROIs_split:
                        all_bands = glob(os.path.join(tile, '*', '*L2R_rhorc*_' + crop))
                        all_bands = sorted(all_bands, key=get_band)

                        current_image = []
                        for c, band in enumerate(all_bands, 1):
                            upscale_factor = int(os.path.basename(os.path.dirname(band))) // 10

                            with rasterio.open(band, mode='r') as src:
                                tags = src.tags().copy()
                                meta = src.meta
                                dtype = src.read(1).dtype
                                current_image.append(src.read(1,
                                                              out_shape=(
                                                                  int(src.height * upscale_factor),
                                                                  int(src.width * upscale_factor)
                                                              ),
                                                              resampling=Resampling.nearest
                                                              ).copy()
                                                     )

                        image = np.stack(current_image)
                        image = np.moveaxis(image, (0, 1, 2), (2, 0, 1))

                        os.makedirs(options['gen_masks_path'], exist_ok=True)

                        output_image = os.path.join(options['gen_masks_path'],
                                                    os.path.basename(crop_name).split('.tif')[0] + '_marinext.tif')

                        meta.update(count=1)

                        with rasterio.open(output_image, 'w',
                                           driver='GTiff',
                                           height=image.shape[0],
                                           width=image.shape[1],
                                           count=1,
                                           dtype=image.dtype,
                                           crs='+proj=latlong') as dst:
                            nan_mask = np.isnan(image)
                            image[nan_mask] = impute_nan[nan_mask]

                            image = transform_test(image)
                            image = standardization(image)
                            image = image.unsqueeze(0)

                            if options['test_time_augmentations']:
                                image = TTA(image)

                            image = image.to(device)

                            all_predictions = []
                            for model in models_list:
                                logits = model(image)
                                logits = F.upsample(input=logits, size=(240, 240), mode='bilinear')

                                predictions = torch.nn.functional.softmax(logits.detach(), dim=1)
                                predictions = predictions.argmax(1) + 1

                                if options['test_time_augmentations']:
                                    predictions = TTA(predictions, reverse_aggregation=True)
                                all_predictions.append(predictions)

                            all_predictions = torch.cat(all_predictions)
                            all_predictions = torch.mode(all_predictions, dim=0, keepdim=True)[0]

                            predictions = predictions.squeeze().cpu().numpy()
                            dst.write_band(1, predictions.astype(dtype).copy())
                            dst.update_tags(**tags)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default="/data/datasets/MADOS", help='Path of the images')
    parser.add_argument("--config", default='marinext.tiny.240x240.mados.py', type=str)
    parser.add_argument('--split', default='test', type=str)
    parser.add_argument('--test_time_augmentations', default=True, type=bool_flag)
    parser.add_argument('--batch', default=1, type=int)
    parser.add_argument('--input_channels', default=11, type=int)
    parser.add_argument('--output_channels', default=15, type=int)
    parser.add_argument('--model_path',
                        default="/data/Development/My/mados/marinext/trained_models/marinext_trelu_full_funny_params/105/",
                        help='Path to pytorch model')
    parser.add_argument('--results_percentage', default=True, type=bool_flag)
    parser.add_argument('--predict_masks', default=False, type=bool_flag)
    parser.add_argument('--gen_masks_path', default=os.path.join(root_path, 'data', 'predicted_marinext'))
    parser.add_argument('--num_workers', default=6, type=int)
    parser.add_argument('--pin_memory', default=True, type=bool_flag)
    parser.add_argument('--prefetch_factor', default=2, type=int)
    parser.add_argument('--persistent_workers', default=True, type=bool_flag)
    parser.add_argument('--save_pruned_model', default=True, type=bool_flag, help='Save the pruned state dict?')
    parser.add_argument('--prune_threshold', default=1e-20, type=float,
                        help='Threshold for TReLU r1/r2 to consider MLP dead')

    args = parser.parse_args()
    options = vars(args)
    main(options)