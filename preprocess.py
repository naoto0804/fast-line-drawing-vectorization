import argparse
from pathlib import Path

import cairosvg
import re

parser = argparse.ArgumentParser()
parser.add_argument("--root", default="../data")
parser.add_argument("--dataset", nargs="*")
args = parser.parse_args()

root = Path(args.root)
for dataset in args.dataset:
    data_root = root / dataset
    svg_root = data_root / "svg"
    png_root = data_root / "png"

    for subset in ["train", "test"]:
        id_file = data_root / "{:s}.txt".format(subset)
        if not id_file.exists():
            ids = []
            for f in (data_root / subset).glob("*"):
                ids.append(f.with_suffix("").name)
            with id_file.open("w") as id_f:
                id_f.write("\n".join(ids))

    png_root.mkdir(parents=True, exist_ok=True)

    if dataset in ["cat", "baseball"]:
        for subset in ["train", "test"]:
            for svg_name in (data_root / subset).glob("*"):
                png_name = png_root / svg_name.with_suffix(".png").name
                if not png_name.exists():
                    print(png_name)
                    with svg_name.open("r") as svg_f:
                        # "#XXXXXX" -> "#000000" (black)
                        pattern = r'#[a-z0-9]*'
                        repl = "#000000"
                        content = re.sub(pattern, repl, svg_f.read())
                        cairosvg.svg2png(content, write_to=str(png_name))
    else:
        for svg_name in svg_root.glob("*"):
            png_name = png_root / svg_name.with_suffix(".png").name
            if not png_name.exists():
                print(png_name)
                if dataset == "kanji":
                    cairosvg.svg2png(url=str(svg_name), write_to=str(png_name))
                elif dataset == "ch":
                    cairosvg.svg2png(url=str(svg_name), write_to=str(png_name))
                elif dataset == "line":
                    with svg_name.open("r") as svg_f:
                        # set RGB value (X, X, X) to (0, 0, 0)
                        pattern = r'([0-9]*,[0-9]*,[0-9]*)'
                        repl = "0,0,0"
                        content = re.sub(pattern, repl, string=svg_f.read())
                    cairosvg.svg2png(content, write_to=str(png_name))
