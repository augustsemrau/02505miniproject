lint:
	flake8 src

format:
	black src
	isort src

install:
	pip install -r requirements.txt

dataset:
	python src/data/make_data.py
	python src/data/make_image_patches.py --patch_size 128
	python src/data/make_image_patches.py --patch_size 64
	python src/data/make_image_patches.py --patch_size 32
	python src/data/make_image_patches.py --patch_size 16

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete