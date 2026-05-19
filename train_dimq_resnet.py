"""Train DIMQ weight quantization on torchvision image classifiers.

Default behavior is intentionally W-only DIMQ:
- build a torchvision model with ImageNet pretrained weights;
- wrap Conv2d/Linear layers with DIMQ, skipping first and last by default;
- train with CrossEntropy + DIMQ softmin distortion + center separation;
- export both dequantized and compact DIMQ checkpoints.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn

from quant import DIMQConfig, apply_dimq, dimq_regularization_loss, get_tau, set_dimq_tau
from quant.export_dimq import (
    assert_unique_values_within_codebook,
    export_compact_checkpoint,
    export_dequantized_checkpoint,
)


TORCHVISION_WEIGHT_ENUMS = {
    "resnet18": "ResNet18_Weights",
    "resnet34": "ResNet34_Weights",
    "resnet50": "ResNet50_Weights",
    "resnet101": "ResNet101_Weights",
    "resnet152": "ResNet152_Weights",
    "wide_resnet50_2": "Wide_ResNet50_2_Weights",
    "wide_resnet101_2": "Wide_ResNet101_2_Weights",
    "mobilenet_v2": "MobileNet_V2_Weights",
    "swin_s": "Swin_S_Weights",
    "vit_b_16": "ViT_B_16_Weights",
    "vit_b_32": "ViT_B_32_Weights",
}

RESNET_WEIGHT_ENUMS = {
    name: enum_name
    for name, enum_name in TORCHVISION_WEIGHT_ENUMS.items()
    if name.startswith("resnet") or name.startswith("wide_resnet")
}


def parse_args(
    default_arch: str = "resnet50",
    default_output_dir: str = "checkpoints/dimq_resnet",
    default_batch_size: int | None = None,
    default_overrides: Mapping[str, Any] | None = None,
) -> argparse.Namespace:
    default_overrides = default_overrides or {}

    def default(name: str, fallback: Any) -> Any:
        return default_overrides.get(name, fallback)

    parser = argparse.ArgumentParser(description="DIMQ QAT for torchvision image classifiers")
    parser.add_argument("--arch", default=default_arch, choices=sorted(TORCHVISION_WEIGHT_ENUMS))
    parser.add_argument("--dataset-module", default="dataset1k")
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--weights", default="DEFAULT", help="torchvision weights enum member or DEFAULT")
    parser.add_argument("--checkpoint", default=None, help="optional checkpoint loaded after pretrained init")

    parser.add_argument("--epochs", type=int, default=default("epochs", 80))
    parser.add_argument(
        "--batch-size",
        type=int,
        default=default_batch_size,
        help="override train/val DataLoader batch size; default keeps the dataset module setting",
    )
    parser.add_argument("--batch-size-note", default=None, help="dataset1k owns DataLoader batch size; kept as a CLI note")
    parser.add_argument("--lr", type=float, default=default("lr", 0.01))
    parser.add_argument("--optimizer", default=default("optimizer", "sgd"), choices=["sgd", "adamw"])
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=default("weight_decay", 1e-4))
    parser.add_argument("--workers", type=int, default=None, help="dataset1k owns workers; kept for compatibility")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--print-freq", type=int, default=20)
    parser.add_argument("--no-progress", action="store_true", help="disable tqdm progress bars")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)

    parser.add_argument("--w-bits", type=int, default=3)
    parser.add_argument("--a-bits", type=int, default=default("a_bits", None))
    parser.add_argument(
        "--a-center-init",
        default=default("a_center_init", "kmeans"),
        choices=["kmeans", "quantile", "linspace_std", "linspace_minmax"],
    )
    parser.add_argument("--a-cluster-momentum", type=float, default=default("a_cluster_momentum", 0.95))
    parser.add_argument("--a-kmeans-iters", type=int, default=default("a_kmeans_iters", 10))
    parser.add_argument("--a-kmeans-sample-size", type=int, default=default("a_kmeans_sample_size", 32768))
    parser.add_argument("--center-init", default="kmeans", choices=["kmeans", "quantile", "linspace_std", "linspace_minmax"])
    parser.add_argument("--forward-mode", default="hard_ste", choices=["hard_ste", "soft"])
    parser.add_argument("--loss-reduction", default="sum", choices=["sum", "mean_per_layer"])
    parser.add_argument("--lambda-dimq", type=float, default=default("lambda_dimq", 1e-4))
    parser.add_argument("--gamma-sep", type=float, default=default("gamma_sep", 1e-3))
    parser.add_argument("--eta-margin", type=float, default=default("eta_margin", 1.0))
    parser.add_argument("--tau-start", type=float, default=default("tau_start", 1.0))
    parser.add_argument("--tau-end", type=float, default=default("tau_end", 1e-5))
    parser.add_argument("--tau-schedule", default="exponential", choices=["exponential", "linear", "cosine"])
    parser.add_argument("--chunk-size", type=int, default=262144)
    parser.add_argument("--center-lr-scale", type=float, default=default("center_lr_scale", 1.0))
    parser.add_argument("--no-skip-first", action="store_true", default=default("no_skip_first", False))
    parser.add_argument("--no-skip-last", action="store_true", default=default("no_skip_last", False))
    parser.add_argument("--quantize-downsample", action="store_true", help="include ResNet downsample projection layers")
    parser.add_argument(
        "--skip-depthwise",
        action=argparse.BooleanOptionalAction,
        default=default("skip_depthwise", False),
        help="skip depthwise Conv2d layers where groups=in_channels=out_channels",
    )
    parser.add_argument(
        "--sort-centers-after-step",
        action=argparse.BooleanOptionalAction,
        default=default("sort_centers_after_step", False),
    )

    parser.add_argument("--output-dir", default=default_output_dir)
    parser.add_argument("--save-every", type=int, default=0, help="save resumable training state every N epochs; 0 disables")
    parser.add_argument(
        "--export-best",
        action=argparse.BooleanOptionalAction,
        default=default("export_best", True),
        help="export quantized checkpoints whenever validation reaches a new best",
    )
    parser.add_argument(
        "--save-quantized-every",
        type=int,
        default=default("save_quantized_every", 10),
        help="export quantized dequantized/compact checkpoints every N epochs; 0 disables",
    )
    parser.add_argument(
        "--save-analysis-every",
        type=int,
        default=default("save_analysis_every", 1),
        help="save per-layer quantization analysis every N epochs; 0 disables detailed analysis",
    )
    return parser.parse_args()


def main(
    default_arch: str = "resnet50",
    default_output_dir: str = "checkpoints/dimq_resnet",
    default_batch_size: int | None = None,
    default_overrides: Mapping[str, Any] | None = None,
) -> None:
    args = parse_args(
        default_arch=default_arch,
        default_output_dir=default_output_dir,
        default_batch_size=default_batch_size,
        default_overrides=default_overrides,
    )
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = load_dataloaders(args.dataset_module)
    if args.batch_size is not None:
        train_loader = rebuild_dataloader_with_batch_size(train_loader, args.batch_size)
        val_loader = rebuild_dataloader_with_batch_size(val_loader, args.batch_size)
        print(f"Using overridden train/val batch size: {args.batch_size}")
    model = build_torchvision_model(
        arch=args.arch,
        pretrained=args.pretrained,
        weights_name=args.weights,
        num_classes=args.num_classes,
    )
    if args.checkpoint:
        load_model_checkpoint(model, args.checkpoint)
    model.to(device)

    cfg = DIMQConfig(
        w_bits=args.w_bits,
        a_bits=args.a_bits,
        a_center_init=args.a_center_init,
        a_cluster_momentum=args.a_cluster_momentum,
        a_kmeans_iters=args.a_kmeans_iters,
        a_kmeans_sample_size=args.a_kmeans_sample_size,
        skip_first=not args.no_skip_first,
        skip_last=not args.no_skip_last,
        skip_downsample=not args.quantize_downsample,
        skip_depthwise=args.skip_depthwise,
        tau_start=args.tau_start,
        tau_end=args.tau_end,
        tau_schedule=args.tau_schedule,
        total_epochs=args.epochs,
        lambda_dimq=args.lambda_dimq,
        gamma_sep=args.gamma_sep,
        eta_margin=args.eta_margin,
        center_init=args.center_init,
        center_lr_scale=args.center_lr_scale,
        loss_reduction=args.loss_reduction,
        chunk_size=args.chunk_size,
        forward_mode=args.forward_mode,
        sort_centers_after_step=args.sort_centers_after_step,
    )
    dimq_modules = apply_dimq(model, cfg)
    if not dimq_modules:
        raise RuntimeError("No Conv2d/Linear layers were selected for DIMQ wrapping")
    patched_attn = maybe_enable_swin_attention_fake_quant(model, args.arch)
    msg = f"Applied DIMQ to {len(dimq_modules)} layers; first wrapped layer: {dimq_modules[0].name}"
    if patched_attn is not None:
        msg += f"; patched {patched_attn} Swin attention modules for fake-quant weights"
    print(msg)
    save_run_config(args, cfg, dimq_modules, output_dir)

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, dimq_modules, args, cfg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    steps_per_epoch = infer_steps_per_epoch(train_loader, args.max_train_batches)
    max_steps = max(1, args.epochs * steps_per_epoch)
    global_step = 0
    best_acc1 = float("-inf")
    best_epoch = 0
    last_val_acc1 = 0.0
    runtime_start = time.perf_counter()
    runtime_peak_allocated_mib = 0.0
    runtime_peak_reserved_mib = 0.0

    for epoch in range(args.epochs):
        epoch_num = epoch + 1
        reset_cuda_peak_memory(device)
        synchronize_cuda(device)
        epoch_start = time.perf_counter()
        train_start = time.perf_counter()
        train_metrics, global_step = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            dimq_modules=dimq_modules,
            cfg=cfg,
            device=device,
            epoch=epoch,
            global_step=global_step,
            max_steps=max_steps,
            print_freq=args.print_freq,
            use_progress=not args.no_progress,
            max_batches=args.max_train_batches,
        )
        synchronize_cuda(device)
        train_seconds = time.perf_counter() - train_start
        scheduler.step()

        synchronize_cuda(device)
        val_start = time.perf_counter()
        val_metrics = evaluate(model, val_loader, criterion, device, args.max_val_batches, not args.no_progress)
        synchronize_cuda(device)
        val_seconds = time.perf_counter() - val_start
        last_val_acc1 = val_metrics["acc1"]
        is_best = val_metrics["acc1"] > best_acc1
        memory_stats = get_memory_stats(device)
        runtime_peak_allocated_mib = max(runtime_peak_allocated_mib, float(memory_stats.get("max_allocated_mib", 0.0)))
        runtime_peak_reserved_mib = max(runtime_peak_reserved_mib, float(memory_stats.get("max_reserved_mib", 0.0)))
        runtime_metrics = build_runtime_record(
            train_seconds=train_seconds,
            val_seconds=val_seconds,
            epoch_seconds=time.perf_counter() - epoch_start,
            total_elapsed_seconds=time.perf_counter() - runtime_start,
            memory=memory_stats,
        )
        if is_best:
            best_acc1 = val_metrics["acc1"]
            best_epoch = epoch_num
        if is_best and args.export_best:
            epoch_extra = build_export_extra(
                args,
                epoch_num=epoch_num,
                val_acc1=val_metrics["acc1"],
                best_acc1=best_acc1,
                best_epoch=best_epoch,
            )
            export_quantized_checkpoints(
                model,
                output_dir / "best_dimq",
                extra=epoch_extra,
            )
        print(
            "Epoch {epoch}: lr={lr:.6g} tau={tau} "
            "train_loss={loss:.4f} train_acc1={train_acc1:.3f} task={task:.4f} dimq={dimq:.4f} sep={sep:.4f} "
            "val_loss={val_loss:.4f} val_acc1={acc1:.3f} best_acc1={best:.3f} "
            "time={time:.1f}s gpu={gpu}".format(
                epoch=epoch_num,
                lr=optimizer.param_groups[0]["lr"],
                tau=format_tau_for_log(train_metrics),
                loss=train_metrics["loss"],
                train_acc1=train_metrics["acc1"],
                task=train_metrics["task_loss"],
                dimq=train_metrics["dimq_loss"],
                sep=train_metrics["sep_loss"],
                val_loss=val_metrics["loss"],
                acc1=val_metrics["acc1"],
                best=best_acc1,
                time=runtime_metrics["epoch_seconds"],
                gpu=format_memory_for_log(memory_stats),
            )
        )
        epoch_record = build_epoch_record(
            epoch_num=epoch_num,
            lr=optimizer.param_groups[0]["lr"],
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            best_acc1=best_acc1,
            best_epoch=best_epoch,
            is_best=is_best,
            runtime=runtime_metrics,
        )
        if args.save_analysis_every > 0 and epoch_num % args.save_analysis_every == 0:
            analysis_path, quant_summary = save_quantization_analysis(
                dimq_modules,
                output_dir,
                epoch_record,
            )
            epoch_record["analysis_path"] = str(analysis_path.relative_to(output_dir))
            epoch_record["quant_summary"] = quant_summary
        append_jsonl(output_dir / "training_log.jsonl", epoch_record)

        if args.save_every > 0 and epoch_num % args.save_every == 0:
            save_training_checkpoint(
                model,
                optimizer,
                scheduler,
                cfg,
                output_dir / f"epoch_{epoch_num}.pth",
                epoch,
                best_acc1,
            )
        if args.save_quantized_every > 0 and epoch_num % args.save_quantized_every == 0:
            epoch_extra = build_export_extra(
                args,
                epoch_num=epoch_num,
                val_acc1=val_metrics["acc1"],
                best_acc1=best_acc1,
                best_epoch=best_epoch,
            )
            export_quantized_checkpoints(
                model,
                output_dir / f"epoch_{epoch_num:04d}_dimq",
                extra=epoch_extra,
            )

    assert_unique_values_within_codebook(model)
    extra = build_export_extra(
        args,
        epoch_num=args.epochs,
        val_acc1=last_val_acc1,
        best_acc1=best_acc1,
        best_epoch=best_epoch,
    )
    save_training_checkpoint(
        model,
        optimizer,
        scheduler,
        cfg,
        output_dir / "last_train_state.pth",
        args.epochs - 1,
        best_acc1,
    )
    export_quantized_checkpoints(model, output_dir / "dimq", extra=extra)
    write_json(
        output_dir / "runtime_summary.json",
        build_runtime_summary(
            total_elapsed_seconds=time.perf_counter() - runtime_start,
            completed_epochs=args.epochs,
            best_acc1=best_acc1,
            best_epoch=best_epoch,
            final_memory=get_memory_stats(device),
            peak_allocated_mib=runtime_peak_allocated_mib,
            peak_reserved_mib=runtime_peak_reserved_mib,
        ),
    )
    print(f"Saved DIMQ outputs to {output_dir}")


def build_torchvision_model(arch: str, pretrained: bool, weights_name: str, num_classes: int) -> nn.Module:
    import torchvision.models as models

    if not hasattr(models, arch):
        raise ValueError(f"torchvision.models has no architecture named {arch}")

    builder = getattr(models, arch)
    weights = resolve_torchvision_weights(models, arch, weights_name) if pretrained else None
    try:
        model = builder(weights=weights)
    except TypeError:
        model = builder(pretrained=pretrained)

    replace_classifier_head(model, arch, num_classes)
    return model


def build_resnet(arch: str, pretrained: bool, weights_name: str, num_classes: int) -> nn.Module:
    return build_torchvision_model(arch, pretrained, weights_name, num_classes)


def maybe_enable_swin_attention_fake_quant(model: nn.Module, arch: str) -> int | None:
    if not arch.startswith("swin"):
        return None

    from models import enable_dimq_swin_attention_fake_quant

    return enable_dimq_swin_attention_fake_quant(model)


def replace_classifier_head(model: nn.Module, arch: str, num_classes: int) -> None:
    if num_classes == 1000:
        return

    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        print(f"Replaced pretrained classifier with a new {num_classes}-class head")
        return

    for attr_name in ("classifier", "heads", "head"):
        if replace_linear_attr_or_tail(model, attr_name, num_classes):
            print(f"Replaced pretrained classifier with a new {num_classes}-class head")
            return

    raise ValueError(f"{arch} does not expose a supported Linear classifier head")


def replace_linear_attr_or_tail(model: nn.Module, attr_name: str, num_classes: int) -> bool:
    module = getattr(model, attr_name, None)
    if isinstance(module, nn.Linear):
        setattr(model, attr_name, nn.Linear(module.in_features, num_classes))
        return True

    if isinstance(module, nn.Sequential):
        for index in range(len(module) - 1, -1, -1):
            if isinstance(module[index], nn.Linear):
                module[index] = nn.Linear(module[index].in_features, num_classes)
                return True
    return False


def resolve_torchvision_weights(models: Any, arch: str, weights_name: str) -> Any:
    enum_name = TORCHVISION_WEIGHT_ENUMS[arch]
    weights_enum = getattr(models, enum_name, None)
    if weights_enum is None:
        return None
    if weights_name.upper() == "DEFAULT":
        return weights_enum.DEFAULT
    if hasattr(weights_enum, "__members__") and weights_name in weights_enum.__members__:
        return weights_enum[weights_name]
    raise ValueError(
        f"Unknown weights {weights_name!r} for {arch}; "
        f"use DEFAULT or one of {list(weights_enum.__members__)}"
    )


def load_dataloaders(module_name: str):
    module = importlib.import_module(module_name)
    train_loader = getattr(module, "train_loader")
    val_loader = getattr(module, "val_loader", None)
    if val_loader is None:
        val_loader = getattr(module, "test_loader")
    return train_loader, val_loader


def rebuild_dataloader_with_batch_size(loader, batch_size: int):
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    from torch.utils.data import DataLoader

    kwargs = {
        "dataset": loader.dataset,
        "batch_size": int(batch_size),
        "sampler": getattr(loader, "sampler", None),
        "num_workers": getattr(loader, "num_workers", 0),
        "collate_fn": getattr(loader, "collate_fn", None),
        "pin_memory": getattr(loader, "pin_memory", False),
        "drop_last": getattr(loader, "drop_last", False),
        "timeout": getattr(loader, "timeout", 0),
        "worker_init_fn": getattr(loader, "worker_init_fn", None),
        "persistent_workers": getattr(loader, "persistent_workers", False),
    }
    multiprocessing_context = getattr(loader, "multiprocessing_context", None)
    if multiprocessing_context is not None:
        kwargs["multiprocessing_context"] = multiprocessing_context
    generator = getattr(loader, "generator", None)
    if generator is not None:
        kwargs["generator"] = generator
    prefetch_factor = getattr(loader, "prefetch_factor", None)
    if kwargs["num_workers"] > 0 and prefetch_factor is not None:
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(**kwargs)


def load_model_checkpoint(model: nn.Module, checkpoint_path: str) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded checkpoint {checkpoint_path}; missing={len(missing)} unexpected={len(unexpected)}")


def build_optimizer(model: nn.Module, dimq_modules: list[Any], args: argparse.Namespace, cfg: DIMQConfig):
    center_ids = {id(module.centers) for module in dimq_modules}
    base_params = []
    center_params = []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        if id(param) in center_ids:
            center_params.append(param)
        else:
            base_params.append(param)

    param_groups = [{"params": base_params, "lr": args.lr, "weight_decay": args.weight_decay}]
    if center_params:
        param_groups.append({"params": center_params, "lr": args.lr * cfg.center_lr_scale, "weight_decay": 0.0})
    if args.optimizer == "adamw":
        return torch.optim.AdamW(param_groups, lr=args.lr)
    return torch.optim.SGD(param_groups, lr=args.lr, momentum=args.momentum)


def should_update_tau(cfg: DIMQConfig) -> bool:
    return cfg.lambda_dimq != 0.0 or cfg.forward_mode == "soft"


def train_one_epoch(
    *,
    model: nn.Module,
    train_loader,
    criterion,
    optimizer,
    dimq_modules,
    cfg: DIMQConfig,
    device: torch.device,
    epoch: int,
    global_step: int,
    max_steps: int,
    print_freq: int,
    use_progress: bool,
    max_batches: int | None,
) -> tuple[dict[str, Any], int]:
    model.train()
    meters = RunningAverages()
    update_tau = should_update_tau(cfg)
    last_tau = cfg.tau_start if update_tau else 0.0
    total_batches = infer_steps_per_epoch(train_loader, max_batches)
    progress_iter = make_progress_bar(
        enumerate(train_loader),
        total=total_batches,
        desc=f"Train {epoch + 1}/{cfg.total_epochs}",
        enabled=use_progress,
    )

    for batch_idx, batch in progress_iter:
        if max_batches is not None and batch_idx >= max_batches:
            break
        images, target = move_batch(batch, device)
        if update_tau:
            progress = global_step / max(1, max_steps - 1)
            last_tau = get_tau(progress, cfg)
            set_dimq_tau(dimq_modules, last_tau)

        output = model(images)
        task_loss = criterion(output, target)
        batch_acc1 = accuracy_top1(output, target)
        compute_dimq_loss = cfg.lambda_dimq != 0.0
        compute_sep_loss = cfg.gamma_sep != 0.0
        if not compute_dimq_loss and not compute_sep_loss:
            dimq_loss = task_loss.new_zeros(())
            sep_loss = task_loss.new_zeros(())
        else:
            dimq_loss, sep_loss = dimq_regularization_loss(
                dimq_modules,
                compute_dimq=compute_dimq_loss,
                compute_sep=compute_sep_loss,
            )
        total_loss = task_loss + cfg.lambda_dimq * dimq_loss + cfg.gamma_sep * sep_loss
        if not bool(torch.isfinite(total_loss).detach().cpu()):
            raise FloatingPointError(
                f"Non-finite loss at epoch={epoch + 1} batch={batch_idx}: "
                f"task={task_loss.item()} dimq={dimq_loss.item()} sep={sep_loss.item()}"
            )

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()

        if cfg.sort_centers_after_step:
            for module in dimq_modules:
                module.sort_centers_()

        batch_size = images.size(0)
        meters.update("loss", total_loss.item(), batch_size)
        meters.update("task_loss", task_loss.item(), batch_size)
        meters.update("dimq_loss", dimq_loss.item(), batch_size)
        meters.update("sep_loss", sep_loss.item(), batch_size)
        meters.update("acc1", batch_acc1, batch_size)

        if hasattr(progress_iter, "set_postfix"):
            progress_iter.set_postfix(
                loss=f"{meters.avg('loss'):.4f}",
                acc1=f"{meters.avg('acc1'):.3f}",
                tau=f"{last_tau:.2e}" if update_tau else "off",
                dimq=f"{meters.avg('dimq_loss'):.3f}",
                sep=f"{meters.avg('sep_loss'):.3f}",
            )
        elif batch_idx % max(1, print_freq) == 0:
            print(
                "Epoch {epoch} [{batch}/{total}] tau={tau} "
                "loss={loss:.4f} acc1={acc1:.3f} task={task:.4f} dimq={dimq:.4f} sep={sep:.4f}".format(
                    epoch=epoch + 1,
                    batch=batch_idx,
                    total=total_batches,
                    tau=f"{last_tau:.6g}" if update_tau else "off",
                    loss=meters.avg("loss"),
                    acc1=meters.avg("acc1"),
                    task=meters.avg("task_loss"),
                    dimq=meters.avg("dimq_loss"),
                    sep=meters.avg("sep_loss"),
                )
            )

        global_step += 1

    metrics = meters.as_dict()
    for key in ("loss", "task_loss", "dimq_loss", "sep_loss", "acc1"):
        metrics.setdefault(key, 0.0)
    metrics["tau"] = float(last_tau) if update_tau else 0.0
    metrics["tau_active"] = bool(update_tau)
    return metrics, global_step


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion,
    device: torch.device,
    max_batches: int | None,
    use_progress: bool,
) -> dict[str, float]:
    model.eval()
    meters = RunningAverages()
    correct = 0
    total = 0
    total_batches = infer_steps_per_epoch(loader, max_batches)
    progress_iter = make_progress_bar(
        enumerate(loader),
        total=total_batches,
        desc="Val",
        enabled=use_progress,
    )
    for batch_idx, batch in progress_iter:
        if max_batches is not None and batch_idx >= max_batches:
            break
        images, target = move_batch(batch, device)
        output = model(images)
        loss = criterion(output, target)
        pred = output.argmax(dim=1)
        correct += int((pred == target).sum().item())
        total += int(target.numel())
        meters.update("loss", loss.item(), images.size(0))
        meters.update("acc1", accuracy_top1(output, target), images.size(0))
        if hasattr(progress_iter, "set_postfix"):
            progress_iter.set_postfix(
                loss=f"{meters.avg('loss'):.4f}",
                acc1=f"{meters.avg('acc1'):.3f}",
            )
    metrics = meters.as_dict()
    metrics.setdefault("loss", 0.0)
    metrics.setdefault("acc1", 0.0)
    metrics["acc1"] = 100.0 * correct / max(1, total)
    return metrics


def accuracy_top1(output: torch.Tensor, target: torch.Tensor) -> float:
    pred = output.argmax(dim=1)
    correct = (pred == target).sum().item()
    return 100.0 * float(correct) / max(1, target.numel())


def make_progress_bar(iterable, *, total: int, desc: str, enabled: bool):
    if not enabled:
        return iterable
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, total=total, desc=desc, dynamic_ncols=True, leave=False)


def move_batch(batch, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    images, target = batch
    return images.to(device, non_blocking=True), target.to(device, non_blocking=True)


def infer_steps_per_epoch(loader, max_batches: int | None) -> int:
    try:
        length = len(loader)
    except TypeError:
        length = max_batches or 1
    if max_batches is not None:
        return min(length, max_batches)
    return length


def is_cuda_device(device: torch.device) -> bool:
    return device.type == "cuda" and torch.cuda.is_available()


def cuda_device_index(device: torch.device) -> int:
    if device.index is not None:
        return int(device.index)
    return int(torch.cuda.current_device())


def synchronize_cuda(device: torch.device) -> None:
    if is_cuda_device(device):
        torch.cuda.synchronize(cuda_device_index(device))


def reset_cuda_peak_memory(device: torch.device) -> None:
    if is_cuda_device(device):
        torch.cuda.reset_peak_memory_stats(cuda_device_index(device))


def bytes_to_mib(value: int | float) -> float:
    return float(value) / (1024.0 ** 2)


def get_memory_stats(device: torch.device) -> dict[str, Any]:
    if not is_cuda_device(device):
        return {
            "cuda": False,
            "device": str(device),
            "allocated_bytes": 0,
            "reserved_bytes": 0,
            "max_allocated_bytes": 0,
            "max_reserved_bytes": 0,
            "allocated_mib": 0.0,
            "reserved_mib": 0.0,
            "max_allocated_mib": 0.0,
            "max_reserved_mib": 0.0,
        }

    device_index = cuda_device_index(device)
    props = torch.cuda.get_device_properties(device_index)
    total_bytes = int(props.total_memory)
    allocated_bytes = int(torch.cuda.memory_allocated(device_index))
    reserved_bytes = int(torch.cuda.memory_reserved(device_index))
    max_allocated_bytes = int(torch.cuda.max_memory_allocated(device_index))
    max_reserved_bytes = int(torch.cuda.max_memory_reserved(device_index))
    return {
        "cuda": True,
        "device": str(device),
        "device_index": device_index,
        "device_name": props.name,
        "total_bytes": total_bytes,
        "allocated_bytes": allocated_bytes,
        "reserved_bytes": reserved_bytes,
        "max_allocated_bytes": max_allocated_bytes,
        "max_reserved_bytes": max_reserved_bytes,
        "total_mib": bytes_to_mib(total_bytes),
        "allocated_mib": bytes_to_mib(allocated_bytes),
        "reserved_mib": bytes_to_mib(reserved_bytes),
        "max_allocated_mib": bytes_to_mib(max_allocated_bytes),
        "max_reserved_mib": bytes_to_mib(max_reserved_bytes),
        "allocated_fraction": allocated_bytes / max(1, total_bytes),
        "reserved_fraction": reserved_bytes / max(1, total_bytes),
        "max_allocated_fraction": max_allocated_bytes / max(1, total_bytes),
        "max_reserved_fraction": max_reserved_bytes / max(1, total_bytes),
    }


def format_tau_for_log(metrics: dict[str, Any]) -> str:
    if not bool(metrics.get("tau_active", True)):
        return "off"
    return f"{float(metrics.get('tau', 0.0)):.6g}"


def format_memory_for_log(memory: dict[str, Any]) -> str:
    if not memory.get("cuda"):
        return "n/a"
    return (
        f"alloc={memory['allocated_mib']:.0f}MiB "
        f"reserved={memory['reserved_mib']:.0f}MiB "
        f"peak={memory['max_allocated_mib']:.0f}MiB"
    )


def seconds_to_hms(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def build_runtime_record(
    *,
    train_seconds: float,
    val_seconds: float,
    epoch_seconds: float,
    total_elapsed_seconds: float,
    memory: dict[str, Any],
) -> dict[str, Any]:
    return {
        "train_seconds": float(train_seconds),
        "val_seconds": float(val_seconds),
        "epoch_seconds": float(epoch_seconds),
        "total_elapsed_seconds": float(total_elapsed_seconds),
        "train_hms": seconds_to_hms(train_seconds),
        "val_hms": seconds_to_hms(val_seconds),
        "epoch_hms": seconds_to_hms(epoch_seconds),
        "total_elapsed_hms": seconds_to_hms(total_elapsed_seconds),
        "memory": memory,
    }


def build_runtime_summary(
    *,
    total_elapsed_seconds: float,
    completed_epochs: int,
    best_acc1: float,
    best_epoch: int,
    final_memory: dict[str, Any],
    peak_allocated_mib: float,
    peak_reserved_mib: float,
) -> dict[str, Any]:
    return {
        "format": "dimq_runtime_summary",
        "completed_epochs": int(completed_epochs),
        "total_elapsed_seconds": float(total_elapsed_seconds),
        "total_elapsed_hms": seconds_to_hms(total_elapsed_seconds),
        "seconds_per_epoch": float(total_elapsed_seconds) / max(1, int(completed_epochs)),
        "best_acc1": float(best_acc1),
        "best_epoch": int(best_epoch),
        "peak_allocated_mib": float(peak_allocated_mib),
        "peak_reserved_mib": float(peak_reserved_mib),
        "final_memory": final_memory,
    }


def save_run_config(
    args: argparse.Namespace,
    cfg: DIMQConfig,
    dimq_modules: list[Any],
    output_dir: Path,
) -> None:
    """Persist the static run setup for later hyperparameter analysis."""

    config = {
        "format": "dimq_run_config",
        "args": vars(args),
        "quant_cfg": cfg.to_dict(),
        "dimq_layers": [
            {
                "name": module.name,
                "module_type": module.__class__.__name__,
                "weight_shape": list(module.weight.shape),
                "num_parameters": int(module.weight.numel()),
                "bits": int(module.cfg.w_bits),
                "codebook_size": int(module.K),
                "activation_bits": None if module.cfg.a_bits is None else int(module.cfg.a_bits),
                "activation_codebook_size": int(getattr(module, "aK", 0)),
            }
            for module in dimq_modules
        ],
    }
    write_json(output_dir / "run_config.json", config)
    (output_dir / "training_log.jsonl").write_text("", encoding="utf-8")


def build_epoch_record(
    *,
    epoch_num: int,
    lr: float,
    train_metrics: dict[str, Any],
    val_metrics: dict[str, float],
    best_acc1: float,
    best_epoch: int,
    is_best: bool,
    runtime: dict[str, Any],
) -> dict[str, Any]:
    return {
        "format": "dimq_epoch_record",
        "epoch": epoch_num,
        "lr": float(lr),
        "tau": float(train_metrics.get("tau", 0.0)),
        "tau_active": bool(train_metrics.get("tau_active", True)),
        "train": dict(train_metrics),
        "val": dict(val_metrics),
        "best_acc1": float(best_acc1),
        "best_epoch": int(best_epoch),
        "is_best": bool(is_best),
        "runtime": runtime,
    }


@torch.no_grad()
def save_quantization_analysis(
    dimq_modules: list[Any],
    output_dir: Path,
    epoch_record: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    layers, summary = collect_quantization_analysis(dimq_modules)
    analysis_dir = output_dir / "analysis"
    analysis_path = analysis_dir / f"epoch_{int(epoch_record['epoch']):04d}_quant_stats.pth"
    analysis_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "dimq_epoch_quantization_analysis",
            "epoch": int(epoch_record["epoch"]),
            "metrics": epoch_record,
            "summary": summary,
            "layers": layers,
        },
        analysis_path,
    )
    return analysis_path, summary


@torch.no_grad()
def collect_quantization_analysis(dimq_modules: list[Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    layers: dict[str, Any] = {}
    total_params = 0
    weighted_entropy = 0.0
    weighted_mse = 0.0
    quant_mses: list[float] = []
    quant_maes: list[float] = []
    max_abs_errors: list[float] = []
    sep_violation_ratios: list[float] = []
    unique_counts: list[int] = []

    for module in dimq_modules:
        stats = module.quantization_stats()
        weight = module.weight.detach().float()
        q_weight, _ = module.hard_quantized_weight()
        error = q_weight.detach().float() - weight
        num_params = int(weight.numel())
        total_params += num_params

        quant_mse = float(error.pow(2).mean().item())
        quant_mae = float(error.abs().mean().item())
        max_abs_error = float(error.abs().max().item())
        assignment_entropy = tensor_to_number(stats["avg_assignment_entropy"])
        sep_violation_ratio = tensor_to_number(stats["sep_violation_ratio"])
        unique_count = int(tensor_to_number(stats["unique_quant_values_after_hard"]))

        weighted_entropy += assignment_entropy * num_params
        weighted_mse += quant_mse * num_params
        quant_mses.append(quant_mse)
        quant_maes.append(quant_mae)
        max_abs_errors.append(max_abs_error)
        sep_violation_ratios.append(sep_violation_ratio)
        unique_counts.append(unique_count)

        centers = module.centers.detach().float().cpu()
        center_grad = module.centers.grad
        weight_grad = module.weight.grad
        layers[module.name] = {
            "module_type": module.__class__.__name__,
            "weight_shape": list(module.weight.shape),
            "num_parameters": num_params,
            "bits": int(module.cfg.w_bits),
            "codebook_size": int(module.K),
            "tau": float(module.tau),
            "centers": centers,
            "centers_sorted": centers.sort().values,
            "hard_codebook_usage": stats["hard_codebook_usage"].detach().float().cpu(),
            "stats": {
                "avg_assignment_entropy": assignment_entropy,
                "center_min": tensor_to_number(stats["center_min"]),
                "center_max": tensor_to_number(stats["center_max"]),
                "center_pair_min_distance": tensor_to_number(stats["center_pair_min_distance"]),
                "margin_delta": tensor_to_number(stats["margin_delta"]),
                "sep_violation_ratio": sep_violation_ratio,
                "unique_quant_values_after_hard": unique_count,
                "weight_mean": float(weight.mean().item()),
                "weight_std": float(weight.std(unbiased=False).item()),
                "weight_min": float(weight.min().item()),
                "weight_max": float(weight.max().item()),
                "quant_mse": quant_mse,
                "quant_mae": quant_mae,
                "quant_max_abs_error": max_abs_error,
                "last_center_grad_norm": tensor_norm_or_none(center_grad),
                "last_weight_grad_norm": tensor_norm_or_none(weight_grad),
            },
        }

    summary = {
        "num_layers": len(dimq_modules),
        "total_quantized_params": total_params,
        "weighted_assignment_entropy": weighted_entropy / max(1, total_params),
        "weighted_quant_mse": weighted_mse / max(1, total_params),
        "mean_quant_mse": mean_or_zero(quant_mses),
        "max_quant_mse": max(quant_mses, default=0.0),
        "mean_quant_mae": mean_or_zero(quant_maes),
        "max_quant_max_abs_error": max(max_abs_errors, default=0.0),
        "mean_sep_violation_ratio": mean_or_zero(sep_violation_ratios),
        "max_sep_violation_ratio": max(sep_violation_ratios, default=0.0),
        "mean_unique_quant_values_after_hard": mean_or_zero([float(v) for v in unique_counts]),
        "min_unique_quant_values_after_hard": min(unique_counts, default=0),
    }
    return layers, summary


def tensor_to_number(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def tensor_norm_or_none(value: torch.Tensor | None) -> float | None:
    if value is None:
        return None
    return float(value.detach().float().norm().cpu().item())


def mean_or_zero(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(to_jsonable(payload), sort_keys=True))
        handle.write("\n")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        detached = value.detach().cpu()
        if detached.numel() == 1:
            return detached.item()
        return detached.tolist()
    return value


def build_export_extra(
    args: argparse.Namespace,
    *,
    epoch_num: int,
    val_acc1: float,
    best_acc1: float,
    best_epoch: int,
) -> dict[str, Any]:
    return {
        "arch": args.arch,
        "num_classes": args.num_classes,
        "epoch": epoch_num,
        "val_acc1": val_acc1,
        "best_acc1": best_acc1,
        "best_epoch": best_epoch,
    }


def export_quantized_checkpoints(
    model: nn.Module,
    prefix: Path,
    *,
    extra: dict[str, Any],
) -> tuple[Path, Path]:
    dequantized_path = prefix.parent / f"{prefix.name}_dequantized.pth"
    compact_path = prefix.parent / f"{prefix.name}_compact.pth"
    export_dequantized_checkpoint(model, dequantized_path, extra=extra)
    export_compact_checkpoint(model, compact_path, extra=extra)
    print(f"Saved quantized checkpoints: {dequantized_path.name}, {compact_path.name}")
    return dequantized_path, compact_path


def save_training_checkpoint(
    model: nn.Module,
    optimizer,
    scheduler,
    cfg: DIMQConfig,
    path: Path,
    epoch: int,
    best_acc1: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "dimq_train_state",
            "epoch": epoch,
            "best_acc1": best_acc1,
            "quant_cfg": cfg.to_dict(),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        },
        path,
    )


class RunningAverages:
    def __init__(self) -> None:
        self.totals: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    def update(self, key: str, value: float, n: int = 1) -> None:
        if not math.isfinite(float(value)):
            raise FloatingPointError(f"Non-finite metric {key}: {value}")
        self.totals[key] = self.totals.get(key, 0.0) + float(value) * int(n)
        self.counts[key] = self.counts.get(key, 0) + int(n)

    def avg(self, key: str) -> float:
        return self.totals.get(key, 0.0) / max(1, self.counts.get(key, 0))

    def as_dict(self) -> dict[str, float]:
        return {key: self.avg(key) for key in self.totals}


if __name__ == "__main__":
    main()
