#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim, masked_mean, masked_ssim_loss
from utils.mask_utils import load_valid_mask, interior_mask
from lpipsPyTorch import lpips
import json
import ast
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser

def readImages(renders_dir, gt_dir, names=None):
    renders = []
    gts = []
    image_names = []
    for fname in sorted(os.listdir(renders_dir)) if names is None else names:
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names


def evaluation_manifest(method_dir, scene_dir):
    manifest_path = method_dir / "mask_manifest.json"
    if manifest_path.is_file():
        with open(manifest_path) as file:
            manifest = json.load(file)
        return manifest["masked"], manifest["images"]
    # Legacy exports have no manifest. Do not silently report unmasked metrics
    # for a model trained with masks; rerender to export aligned indexed masks.
    cfg_path = Path(scene_dir) / "cfg_args"
    if cfg_path.is_file():
        expression = ast.parse(cfg_path.read_text(), mode="eval").body
        if isinstance(expression, ast.Call):
            for kw in expression.keywords:
                if kw.arg == "masks" and ast.literal_eval(kw.value):
                    raise ValueError("Masked model needs a fresh render.py export with mask_manifest.json")
    return (method_dir / "masks").is_dir(), None


def evaluate_masked_images(renders, gts, image_names, mask_dir):
    per_view = {name: {} for name in ("L1", "SSIM", "PSNR", "LPIPS", "ValidRatio")}
    for render, gt, name in tqdm(zip(renders, gts, image_names), total=len(image_names), desc="Masked metrics"):
        size = (gt.shape[-1], gt.shape[-2])
        mask = load_valid_mask(str(mask_dir / name), size, size).to(gt.device)
        ratio = float(mask.float().mean())
        per_view["ValidRatio"][name] = ratio
        per_view["LPIPS"][name] = None  # Scalar LPIPS has no mask-aware receptive-field handling.
        if ratio == 0:
            for metric in ("L1", "SSIM", "PSNR"):
                per_view[metric][name] = None
            continue
        per_view["L1"][name] = masked_mean((render - gt).abs(), mask).item()
        mse = masked_mean((render - gt).square(), mask)
        per_view["PSNR"][name] = (-10 * torch.log10(mse)).item()
        per_view["SSIM"][name] = ((1 - masked_ssim_loss(render, gt, mask)).item()
                                  if interior_mask(mask, 11).any() else None)
    summary = {"Masked": True}
    for metric, scores in per_view.items():
        values = [v for v in scores.values() if v is not None]
        summary[metric] = sum(values) / len(values) if values else None
    summary["EvaluatedViews"] = sum(v is not None for v in per_view["L1"].values())
    summary["SSIMViews"] = sum(v is not None for v in per_view["SSIM"].values())
    if not summary["EvaluatedViews"]:
        raise ValueError("No valid pixels available for masked evaluation")
    return summary, per_view

def evaluate(model_paths):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"

            for method in os.listdir(test_dir):
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}

                method_dir = test_dir / method
                if not method_dir.is_dir():
                    continue
                gt_dir = method_dir/ "gt"
                renders_dir = method_dir / "renders"
                masked, names = evaluation_manifest(method_dir, scene_dir)
                renders, gts, image_names = readImages(renders_dir, gt_dir, names)

                if masked:
                    summary, per_view = evaluate_masked_images(renders, gts, image_names, method_dir / "masks")
                    print("Masked evaluation (LPIPS not computed):", summary)
                    full_dict[scene_dir][method] = summary
                    per_view_dict[scene_dir][method] = per_view
                    continue

                ssims = []
                psnrs = []
                lpipss = []

                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    ssims.append(ssim(renders[idx], gts[idx]))
                    psnrs.append(psnr(renders[idx], gts[idx]))
                    lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))

                print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
                print("")

                full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                        "PSNR": torch.tensor(psnrs).mean().item(),
                                                        "LPIPS": torch.tensor(lpipss).mean().item()})
                per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                            "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                            "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)}})

            with open(scene_dir + "/results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
        except Exception as error:
            print("Unable to compute metrics for model {}: {}".format(scene_dir, error))
            raise

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    args = parser.parse_args()
    evaluate(args.model_paths)
