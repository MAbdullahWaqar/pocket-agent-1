.PHONY: all install data train quantize eval demo

all: install data train quantize eval

install:
	pip install -r requirements.txt

data:
	python data/generate_data.py

train:
	python train/finetune.py

quantize:
	python quantize/quantize.py

eval:
	python eval/evaluate.py

demo:
	streamlit run demo.py
