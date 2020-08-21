Source Repo: [renatoviolin/Bart_T5-summarization](https://github.com/renatoviolin/Bart_T5-summarization)  
Credits: [renatoviolin](https://github.com/renatoviolin)

# Text Summarization
Summarization Task using various models available in [HugginFace Transformers](https://github.com/huggingface/transformers)

## Install requirements
```
pip install -U transformers
pip install -U torch
pip install flask
```

## Run
```
python app.py -models 'BART,T5,PEGASUS-MED,PEGASUS-CNN'
```