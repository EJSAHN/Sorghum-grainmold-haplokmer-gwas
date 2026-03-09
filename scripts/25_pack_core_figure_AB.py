#!/usr/bin/env python3
"""
25_pack_core_figure_AB.py

Combine:
  A) 23_zoom_*_zoom_core_hotspot (PNG)
  B) 24_core_*_core_decomposition_stackedbar (PNG)

into a single publication figure (PNG 300 dpi + PDF vector).

Requires:
  pip/conda install pillow matplotlib
"""

from pathlib import Path
import argparse

from PIL import Image
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panelA", required=True, help="Panel A PNG (zoom)")
    ap.add_argument("--panelB", required=True, help="Panel B PNG (decomposition)")
    ap.add_argument("--out_prefix", required=True, help="Output prefix (no extension)")
    ap.add_argument("--labelA", default="A", help="Panel label A")
    ap.add_argument("--labelB", default="B", help="Panel label B")
    args = ap.parse_args()

    a_path = Path(args.panelA)
    b_path = Path(args.panelB)
    if not a_path.exists():
        raise SystemExit(f"[ERROR] Missing: {a_path}")
    if not b_path.exists():
        raise SystemExit(f"[ERROR] Missing: {b_path}")

    imgA = Image.open(a_path).convert("RGB")
    imgB = Image.open(b_path).convert("RGB")

    # Make them the same width (use max width)
    W = max(imgA.size[0], imgB.size[0])

    def resize_to_width(img, w):
        if img.size[0] == w:
            return img
        h = int(round(img.size[1] * (w / img.size[0])))
        return img.resize((w, h), resample=Image.Resampling.LANCZOS)

    imgA = resize_to_width(imgA, W)
    imgB = resize_to_width(imgB, W)

    # Stack vertically with padding
    pad = 40
    outH = imgA.size[1] + pad + imgB.size[1]
    canvas = Image.new("RGB", (W, outH), (255, 255, 255))
    canvas.paste(imgA, (0, 0))
    canvas.paste(imgB, (0, imgA.size[1] + pad))

    # Add panel letters using matplotlib for consistent text rendering
    fig = plt.figure(figsize=(W / 200, outH / 200), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(canvas)
    ax.axis("off")

    # Panel labels
    ax.text(10, 30, args.labelA, fontsize=24, fontweight="bold")
    ax.text(10, imgA.size[1] + pad + 30, args.labelB, fontsize=24, fontweight="bold")

    out_prefix = Path(args.out_prefix)
    out_png = str(out_prefix) + ".png"
    out_pdf = str(out_prefix) + ".pdf"

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Wrote: {out_png}")
    print(f"[OK] Wrote: {out_pdf}")


if __name__ == "__main__":
    main()
