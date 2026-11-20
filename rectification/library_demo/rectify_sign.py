"""Four-corner sign rectification; RGB arrays and EXIF-oriented pixel coordinates."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


def load_rgb(path):
    with Image.open(path) as image:
        return np.asarray(ImageOps.exif_transpose(image).convert("RGB"))


def validate_corners(corners, image_shape):
    """Require TL, TR, BR, BL order; reject crossed, tiny or off-image regions."""
    points = np.asarray(corners, dtype=np.float64)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise ValueError("corners must be four finite [x, y] pairs")
    height, width = image_shape[:2]
    if ((points < 0).any() or (points[:, 0] > width - 1).any()
            or (points[:, 1] > height - 1).any()):
        raise ValueError("corners must lie inside the EXIF-oriented image")
    edges = np.roll(points, -1, axis=0) - points
    following = np.roll(edges, -1, axis=0)
    crosses = edges[:, 0] * following[:, 1] - edges[:, 1] * following[:, 0]
    if not np.all(crosses > 0) or cv2.contourArea(points.astype(np.float32)) < 100:
        raise ValueError("corners must form a clockwise convex quadrilateral: TL, TR, BR, BL")
    if (points[:2, 1].mean() >= points[2:, 1].mean()
            or points[[0, 3], 0].mean() >= points[[1, 2], 0].mean()):
        raise ValueError("check top/bottom and left/right corner order")
    return points


def rectify_image(rgb, corners, target_size):
    """Return (rectified RGB, source-to-destination H). Size is [width, height].

    Uses only geometric resampling, with no text reconstruction or enhancement.
    Corners must be in the coordinate system of the supplied RGB array.
    """
    points = validate_corners(corners, rgb.shape)
    if len(target_size) != 2 or any(int(v) != v or v < 2 for v in target_size):
        raise ValueError("target_size must contain integer width and height >= 2")
    width, height = map(int, target_size)
    destination = np.array([[0, 0], [width - 1, 0],
                            [width - 1, height - 1], [0, height - 1]], np.float32)
    homography = cv2.getPerspectiveTransform(points.astype(np.float32), destination)
    if not np.isfinite(homography).all() or abs(np.linalg.det(homography)) < 1e-12:
        raise ValueError("degenerate homography")
    rectified = cv2.warpPerspective(rgb, homography, (width, height),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    return rectified, homography


def geometric_checks(rgb, corners, target_size, homography):
    """Check forward/inverse geometry and sampling coverage, not OCR accuracy."""
    width, height = target_size
    source = np.asarray(corners, np.float64).reshape(-1, 1, 2)
    destination = np.array([[0, 0], [width - 1, 0],
                            [width - 1, height - 1], [0, height - 1]], np.float64)
    projected = cv2.perspectiveTransform(source, homography).reshape(-1, 2)
    corner_error = float(np.linalg.norm(projected - destination, axis=1).max())
    gx, gy = np.meshgrid(np.linspace(0, width - 1, 9), np.linspace(0, height - 1, 5))
    grid = np.stack([gx.ravel(), gy.ravel()], axis=1).reshape(-1, 1, 2)
    source_grid = cv2.perspectiveTransform(grid, np.linalg.inv(homography))
    recovered = cv2.perspectiveTransform(source_grid, homography)
    inverse_error = float(np.linalg.norm(recovered - grid, axis=2).max())
    valid = cv2.warpPerspective(np.ones(rgb.shape[:2], np.uint8) * 255,
                               homography, (width, height), flags=cv2.INTER_NEAREST)
    coverage = float(np.mean(valid == 255))
    if corner_error > 1e-4 or inverse_error > 1e-4 or coverage < 1:
        raise RuntimeError("geometry validation failed")
    return dict(max_corner_reprojection_error_px=corner_error,
                max_grid_roundtrip_error_px=inverse_error,
                valid_source_coverage_fraction=coverage,
                output_size_wh=[width, height], checks_passed=True,
                scope="Geometry only; physical aspect ratio and OCR improvement are not established.")


def font(size):
    for path in ["/System/Library/Fonts/Supplemental/Arial.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def annotated_image(rgb, corners):
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    points = [tuple(map(int, point)) for point in corners]
    thickness = max(3, round(max(image.size) / 480))
    radius = thickness * 2
    draw.line(points + [points[0]], fill="#32ff94", width=thickness)
    for index, (x, y) in enumerate(points):
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill="#ff5555")
        draw.text((x + radius, y - 7 * thickness), f"{index + 1}",
                  font=font(6 * thickness), fill="#ffff66",
                  stroke_width=max(1, thickness // 3), stroke_fill="black")
    return image


def comparison_image(original, annotated, rectified, crop_box, checks):
    """Two context thumbnails plus before/after panels at comparable widths."""
    canvas = Image.new("RGB", (1800, 1480), "#f1f5f9")
    draw = ImageDraw.Draw(canvas)
    draw.text((50, 26), "Campus sign | four-corner perspective rectification", font=font(40), fill="#102337")
    draw.text((50, 82), "Photo: Library signboard. Selected plane: upper University of Illinois panel.",
              font=font(23), fill="#526477")

    def panel(image, box):
        x, y, width, height = box
        thumbnail = ImageOps.contain(image, (width, height), Image.Resampling.LANCZOS)
        canvas.paste(thumbnail, (x + (width - thumbnail.width) // 2,
                                 y + (height - thumbnail.height) // 2))

    draw.text((80, 137), "1  Original photograph", font=font(27), fill="#102337")
    draw.text((940, 137), "2  Corners: TL, TR, BR, BL", font=font(27), fill="#102337")
    panel(original, (80, 186, 780, 480))
    panel(annotated, (940, 186, 780, 480))
    draw.text((60, 691), "3  Before: annotated source crop", font=font(27), fill="#102337")
    panel(annotated.crop(crop_box), (60, 737, 1680, 340))
    draw.text((60, 1095), f"4  After: rectified sign ({rectified.width} x {rectified.height} px)", font=font(27), fill="#102337")
    panel(rectified, (60, 1140, 1680, 220))
    draw.text((60, 1382), "Original pixels resampled; no text synthesis. Aspect ratio is an estimate, not a physical measurement.",
              font=font(23), fill="#526477")
    draw.text((60, 1420), f"Geometry checks passed | corner error {checks['max_corner_reprojection_error_px']:.2g} px | valid source coverage 100%",
              font=font(21), fill="#526477")
    return canvas


def run(annotation_path, output_dir):
    annotation_path, output_dir = Path(annotation_path), Path(output_dir)
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    source_path = annotation_path.parent / annotation["image"]
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if annotation.get("source_sha256") and source_hash != annotation["source_sha256"]:
        raise ValueError("source image hash mismatch; verify coordinates against this image")
    rgb = load_rgb(source_path)
    if list(rgb.shape[1::-1]) != annotation["oriented_image_size_wh"]:
        raise ValueError("image dimensions differ from corner-annotation coordinates")
    corners, size = annotation["corners_tl_tr_br_bl"], annotation["target_size_wh"]
    rectified, homography = rectify_image(rgb, corners, size)
    checks = geometric_checks(rgb, corners, size, homography)
    original, annotated = Image.fromarray(rgb), annotated_image(rgb, corners)
    points = np.asarray(corners)
    lower = np.maximum(points.min(axis=0).astype(int) - 100, 0)
    upper = np.minimum(points.max(axis=0).astype(int) + 100, rgb.shape[1::-1])
    crop_box = tuple(map(int, [*lower, *upper]))
    output_dir.mkdir(parents=True, exist_ok=True)
    annotated.save(output_dir / "corners_full.png")
    annotated.crop(crop_box).save(output_dir / "corners_detail.png")
    original.crop(crop_box).save(output_dir / "source_crop.png")
    Image.fromarray(rectified).save(output_dir / "rectified.png")
    checks.update(source_sha256=source_hash, opencv_version=cv2.__version__,
                  homography_source_to_rectified=homography.tolist(),
                  homography_rectified_to_source=np.linalg.inv(homography).tolist(),
                  source_crop_box_xyxy=list(crop_box),
                  interpolation="OpenCV INTER_LINEAR", photometric_processing="none")
    (output_dir / "validation.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    comparison_image(original, annotated, Image.fromarray(rectified), crop_box, checks).save(output_dir / "comparison.png")
    print(json.dumps(checks, indent=2))
    return checks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation", type=Path, default=Path(__file__).with_name("annotations.json"))
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()
    run(args.annotation, args.output_dir)
