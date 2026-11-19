"""Shared pilot runner: original NST objective (weight 0) and Sobel variant."""

import argparse
import copy
import csv
import json
import math
import os
import time
from pathlib import Path

import torch
from PIL import Image, ImageOps, ImageDraw
from torchvision import models, transforms

from main import ContentLoss, StyleLoss, Normalization
from edge_loss import SobelEdgeLoss

ROOT = Path(__file__).resolve().parent


def build_model(cnn, mean, std, content, style):
    # Compute targets using feature layers only. The old builder forwards style
    # through an already inserted ContentLoss, which fails for different sizes.
    features = torch.nn.Sequential(Normalization(mean, std, content.device))
    model = torch.nn.Sequential(Normalization(mean, std, content.device))
    style_losses, content_losses = [], []
    conv = 0
    for layer in copy.deepcopy(cnn).children():
        if isinstance(layer, torch.nn.Conv2d):
            conv += 1
            name = f"conv_{conv}"
        elif isinstance(layer, torch.nn.ReLU):
            layer = torch.nn.ReLU(inplace=False)
            name = f"relu_{conv}"
        elif isinstance(layer, torch.nn.MaxPool2d):
            name = f"pool_{conv}"
        elif isinstance(layer, torch.nn.BatchNorm2d):
            name = f"bn_{conv}"
        else:
            raise ValueError(f"Unsupported VGG layer: {type(layer)}")
        features.add_module(name, layer)
        model.add_module(name, layer)
        with torch.no_grad():
            if name == "conv_4":
                loss = ContentLoss(features(content))
                model.add_module("content_loss", loss)
                content_losses.append(loss)
            if name in {"conv_1", "conv_2", "conv_3", "conv_4", "conv_5"}:
                loss = StyleLoss(features(style))
                model.add_module(f"style_loss_{conv}", loss)
                style_losses.append(loss)
        if name == "conv_5":
            break
    if not content_losses or len(style_losses) != 5:
        raise ValueError("Expected a VGG-style backbone with at least five convolutions")
    return model, style_losses, content_losses


def load_image(path, size, device):
    # Preserve the full photograph, including lettering near the image border.
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    scale = size / max(image.size)
    shape = tuple(max(16, round(d * scale)) for d in image.size)
    image = image.resize(shape, Image.Resampling.LANCZOS)
    return transforms.ToTensor()(image).unsqueeze(0).to(device)


def save_image(tensor, path):
    transforms.ToPILImage()(tensor.detach().cpu().squeeze(0).clamp(0, 1)).save(path)


def transfer(cnn, content, style, steps, style_weight, content_weight, edge_weight):
    mean = content.new_tensor([0.485, 0.456, 0.406])
    std = content.new_tensor([0.229, 0.224, 0.225])
    model, style_losses, content_losses = build_model(cnn, mean, std, content, style)
    model.eval().requires_grad_(False)
    edge_loss = SobelEdgeLoss(content)
    output = content.detach().clone().requires_grad_(True)
    # One closure per step gives every weight exactly the same evaluation budget.
    optimizer = torch.optim.LBFGS([output], max_iter=1, max_eval=1)
    history = []

    def scores():
        model(output)
        return (sum(item.loss for item in style_losses),
                sum(item.loss for item in content_losses), edge_loss(output))

    for step in range(1, steps + 1):
        def closure():
            with torch.no_grad():
                output.clamp_(0, 1)
            optimizer.zero_grad()
            style_score, content_score, edge_score = scores()
            total = style_weight * style_score + content_weight * content_score + edge_weight * edge_score
            total.backward()
            history.append(dict(step=step, style=style_score.item(), content=content_score.item(),
                                edge=edge_score.item(), total=total.item()))
            return total
        optimizer.step(closure)
        if step == 1 or step % 50 == 0 or step == steps:
            print(f"  step {step}/{steps}: {history[-1]}", flush=True)
    with torch.no_grad():
        output.clamp_(0, 1)
        s, c, e = scores()
    # Final scores are recomputed after the last update and clamp.
    return output.detach(), history, dict(style=s.item(), content=c.item(), edge=e.item())


def main(default_weights=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--content", type=Path, nargs="+", help="Photos; default: all images in imgs/")
    parser.add_argument("--style", type=Path, default=ROOT / "style.jpg")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "pilot_output")
    parser.add_argument("--edge-weights", type=float, nargs="+", default=default_weights or [0, 1000, 10000, 100000])
    parser.add_argument("--image-size", type=int, default=512, help="Longest side, without cropping")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--style-weight", type=float, default=1e6)
    parser.add_argument("--content-weight", type=float, default=1)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--threads", type=int, default=4, help="CPU worker threads")
    args = parser.parse_args()
    if args.steps < 1 or args.image_size < 32 or args.threads < 1:
        parser.error("steps/threads must be positive and image-size must be at least 32")
    weights = [args.style_weight, args.content_weight, *args.edge_weights]
    if any(not math.isfinite(w) or w < 0 for w in weights):
        parser.error("All loss weights must be finite and nonnegative")
    photos = args.content or sorted(p for p in (ROOT / "imgs").iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    if not photos:
        parser.error("No content images found")
    for path in [args.style, *photos]:
        if not path.is_file():
            parser.error(f"Image not found: {path}")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    torch.set_num_threads(args.threads)
    if "TORCH_HOME" not in os.environ:
        torch.hub.set_dir(str(ROOT / ".torch_cache" / "hub"))
    print(f"Device: {device}; photos: {len(photos)}; edge weights: {args.edge_weights}", flush=True)
    # Reuse the original feature/loss definitions, freezing VGG throughout.
    cnn = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device).eval().requires_grad_(False)
    style = load_image(args.style, args.image_size, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index, photo in enumerate(photos, 1):
        folder = args.output_dir / f"{index:02d}_{photo.stem}"
        folder.mkdir(parents=True, exist_ok=True)
        content = load_image(photo, args.image_size, device)
        save_image(content, folder / "original.png")
        save_image(style, folder / "style.png")
        config = vars(args).copy()
        config.update(content=str(photo.resolve()), device=str(device), size=list(content.shape[-2:]))
        (folder / "config.json").write_text(json.dumps(config, default=str, indent=2), encoding="utf-8")
        panels = [("Original", folder / "original.png")]
        results = []
        for weight in dict.fromkeys(args.edge_weights):
            label = "baseline" if weight == 0 else f"edge_{weight:g}"
            print(f"{photo.name}: {label}", flush=True)
            start = time.perf_counter()
            output, history, metrics = transfer(cnn, content, style, args.steps, args.style_weight, args.content_weight, weight)
            save_image(output, folder / f"{label}.png")
            with (folder / f"{label}_loss.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(history[0]))
                writer.writeheader()
                writer.writerows(history)
            results.append(dict(variant=label, edge_weight=weight, seconds=time.perf_counter()-start, **metrics))
            panels.append((label, folder / f"{label}.png"))
        with (folder / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
        width, height = content.shape[-1], content.shape[-2]
        sheet = Image.new("RGB", (width * len(panels), height + 32), "white")
        draw = ImageDraw.Draw(sheet)
        for i, (label, path) in enumerate(panels):
            with Image.open(path) as panel:
                sheet.paste(panel, (i * width, 32))
            draw.text((i * width + 5, 8), label, fill="black")
        sheet.save(folder / "comparison.png")
        print(f"Saved to {folder}", flush=True)


if __name__ == "__main__":
    main()
