lint:
	flake8 src

format:
	black src
	isort src

install:
	pip3 install -r requirements.txt

dataset:
	python3 src/data/make_data.py
	python3 src/data/make_image_patches.py --patch_size 256
	python3 src/data/make_image_patches.py --patch_size 128
	python3 src/data/make_image_patches.py --patch_size 64
	python3 src/data/make_image_patches.py --patch_size 32

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete