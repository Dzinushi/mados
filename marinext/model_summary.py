# -*- coding: utf-8 -*-
import os
import sys
import time
import argparse
from glob import glob
from os.path import dirname as up

import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode

sys.path.append(up(os.path.abspath(__file__)))
from marinext_wrapper import MariNext

root_path = up(up(os.path.abspath(__file__)))


class ZeroLayer(nn.Module):
    def forward(self, x):
        return torch.zeros_like(x)


def apply_dynamic_pruning(model: nn.Module, threshold: float):
    print(f"\nApplying dynamic pruning (on-the-fly) with threshold = {threshold}...")
    pruned_count = 0
    for name, module in model.named_modules():
        if hasattr(module, 'mlp') and hasattr(module.mlp, 'act') and hasattr(module.mlp.act, 'r1'):
            r1 = abs(module.mlp.act.r1.item())
            r2 = abs(module.mlp.act.r2.item())
            if max(r1, r2) < threshold:
                module.mlp = ZeroLayer()
                pruned_count += 1
    print(f"Total pruned MLP modules: {pruned_count}\n")
    return model


def sync_pruned_structure(model: nn.Module, state_dict: dict):
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state_dict.keys())

    missing_in_ckpt = model_keys - ckpt_keys
    if not missing_in_ckpt:
        return model

    print(
        f"\nDetected {len(missing_in_ckpt)} missing keys in checkpoint. Aligning model structure (restoring pruning)...")

    pruned_parents = set()
    for key in missing_in_ckpt:
        if '.mlp.' in key:
            pruned_parents.add(key.split('.mlp.')[0])
        elif '.attn.' in key:
            pruned_parents.add(key.split('.attn.')[0])

    for parent_name in pruned_parents:
        parts = parent_name.split('.')
        parent = model
        for p in parts:
            if p.isdigit():
                parent = parent[int(p)]
            else:
                parent = getattr(parent, p)

        if any(f"{parent_name}.mlp." in k for k in missing_in_ckpt):
            parent.mlp = ZeroLayer()
            print(f"  -> Restored pruned structure for: {parent_name}.mlp")
        elif any(f"{parent_name}.attn." in k for k in missing_in_ckpt):
            parent.attn = ZeroLayer()
            print(f"  -> Restored pruned structure for: {parent_name}.attn")

    return model


def count_parameters(model: nn.Module):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def measure_flops(model: nn.Module, input_size: list, device: torch.device):
    input_data = torch.randn(input_size, device=device)
    with FlopCounterMode(display=False) as flop_counter:
        model(input_data)
    return flop_counter.get_total_flops()


def measure_latency(model: nn.Module, input_size: list, device: torch.device, warmup: int, runs: int):
    model.eval()
    input_data = torch.randn(input_size, device=device)

    print(f"Warming up for {warmup} iterations...")
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_data)

    if device.type == "cuda":
        torch.cuda.synchronize()

    print(f"Measuring latency over {runs} iterations...")
    start_time = time.time()
    with torch.no_grad():
        for _ in range(runs):
            _ = model(input_data)

    if device.type == "cuda":
        torch.cuda.synchronize()  # Ждем пока GPU закончит все вычисления

    total_time = time.time() - start_time
    avg_time_ms = (total_time / runs) * 1000
    fps = runs / total_time

    return avg_time_ms, fps


def measure_peak_memory(model: nn.Module, input_size: list, device: torch.device):
    if device.type != "cuda":
        return 0.0

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    input_data = torch.randn(input_size, device=device)
    with torch.no_grad():
        _ = model(input_data)

    memory_bytes = torch.cuda.max_memory_allocated()
    return memory_bytes / (1024 ** 3)  # В гигабайтах


def main(options):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MariNext(options['input_channels'], options['output_channels'], options['config'])
    models_files = glob(os.path.join(options['model_path'], '*.pth'))
    if models_files:
        print(f"Loading checkpoint: {models_files[0]}")
        checkpoint = torch.load(models_files[0], map_location=device)
        checkpoint = {
            k.replace('decoder', 'decode_head'): v for k, v in checkpoint.items()
            if ('proj1' not in k) and ('proj2' not in k)
        }
        model = sync_pruned_structure(model, checkpoint)
        model.load_state_dict(checkpoint, strict=False)
    else:
        print("Warning: No checkpoint found. Pruning will be applied to random weights.")

    model.to(device)
    if options['apply_pruning']:
        model = apply_dynamic_pruning(model, options['prune_threshold'])
    model.eval()

    input_size = [1, options['input_channels'], 240, 240]

    print("\n" + "=" * 50)
    print("MODEL BENCHMARK SUMMARY")
    print("=" * 50)

    total_params, trainable_params = count_parameters(model)
    print(f"Total Parameters:     {total_params / 1e6:.2f} M")
    print(f"Trainable Parameters: {trainable_params / 1e6:.2f} M")

    print("\nCalculating FLOPs...")
    flops = measure_flops(model, input_size, device)
    print(f"Total FLOPs:          {flops / 1e9:.2f} GFLOPs")

    avg_time, fps = measure_latency(model, input_size, device, options['warmup_runs'], options['benchmark_runs'])
    print(f"\nAvg Latency:          {avg_time:.2f} ms")
    print(f"Throughput (FPS):     {fps:.2f} FPS")

    if device.type == "cuda":
        mem = measure_peak_memory(model, input_size, device)
        print(f"Peak GPU Memory:      {mem:.2f} GB")

    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='marinext.tiny.240x240.mados.py', type=str)
    parser.add_argument('--input_channels', default=11, type=int)
    parser.add_argument('--output_channels', default=15, type=int)
    parser.add_argument('--model_path',
                        default="/data/Development/My/mados/marinext/trained_models/marinext_trelu_full_funny_params/105/",
                        help='Path to pytorch model')
    parser.add_argument('--warmup_runs', default=10, type=int, help='Number of warmup iterations before measuring time')
    parser.add_argument('--benchmark_runs', default=50, type=int, help='Number of iterations to average latency over')
    parser.add_argument('--apply_pruning', default=True, type=lambda x: x.lower() == 'true',
                        help='Apply dynamic TReLU pruning? (Default: True. Pass "False" to benchmark original model)')
    parser.add_argument('--prune_threshold', default=1e-20, type=float, help='Threshold for TReLU r1/r2')
    args = parser.parse_args()
    options = vars(args)

    torch.manual_seed(0)

    main(options)