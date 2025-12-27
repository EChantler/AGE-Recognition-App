import os
from pathlib import Path

try:
	from PIL import Image
except ImportError as e:
	raise RuntimeError(
		"Pillow (PIL) is required. Install via 'pip install Pillow'."
	) from e


def make_grid(
	input_dir: Path,
	output_path: Path,
	cols: int = 4,
	rows: int = 5,
	cell_size: tuple[int, int] = (224, 224),
	background: tuple[int, int, int] = (255, 255, 255),
	padding: int = 0,
):
	"""Compose images from a folder into a grid and save.

	Args:
		input_dir: Directory containing source images.
		output_path: Path to save the composed grid image.
		cols: Number of columns in the grid.
		rows: Number of rows in the grid.
		cell_size: (width, height) for each grid cell.
		background: RGB background color.
		padding: Padding around and between cells, in pixels.
	"""
	exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
	image_paths = [
		p for p in sorted(Path(input_dir).iterdir()) if p.is_file() and p.suffix.lower() in exts
	]

	if not image_paths:
		raise FileNotFoundError(f"No images found in {input_dir}")

	total_cells = cols * rows
	image_paths = image_paths[:total_cells]

	W, H = cell_size
	grid_w = cols * W + padding * (cols + 1)
	grid_h = rows * H + padding * (rows + 1)
	canvas = Image.new("RGB", (grid_w, grid_h), background)

	for idx, img_path in enumerate(image_paths):
		with Image.open(img_path) as im:
			im = im.convert("RGB")
			# Always resize to 224x224
			tile = im.resize((W, H), Image.LANCZOS)

			col = idx % cols
			row = idx // cols
			x0 = padding + col * (W + padding)
			y0 = padding + row * (H + padding)
			canvas.paste(tile, (x0, y0))

	output_path.parent.mkdir(parents=True, exist_ok=True)
	canvas.save(output_path)
	return output_path


def main():
	here = Path(__file__).resolve()
	# Samples folder is model/samples; output in model/combined_grid.png
	samples_dir = here.parent / "samples"
	output_path = here.parent / "combined_grid.png"

	result = make_grid(
		input_dir=samples_dir,
		output_path=output_path,
		cols=5,
		rows=4,
		cell_size=(224, 224),
		padding=0,
	)
	print(f"Saved grid image to: {result}")


if __name__ == "__main__":
	main()

