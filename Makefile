.PHONY: install prepare train eval push test

install:
	python -m pip install -U pip
	python -m pip install -e .

prepare:
	python -m ghana_asr.cli.prepare --config configs/whisper_akan_ewe.yaml

train:
	python -m ghana_asr.cli.train --config configs/whisper_akan_ewe.yaml

eval:
	python -m ghana_asr.cli.evaluate --config configs/whisper_akan_ewe.yaml --split test

push:
	python -m ghana_asr.cli.push --config configs/whisper_akan_ewe.yaml

test:
	pytest -q
