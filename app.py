# import config
import torch
import flask
from flask import Flask, request, render_template
from flask_ngrok import run_with_ngrok
import json
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import PegasusTokenizer, PegasusForConditionalGeneration


BART_PATH = 'facebook/bart-large-cnn'
T5_PATH = 't5-large'
PEGASUS_PATH = 'google/pegasus-cnn_dailymail'

app = Flask(__name__)
run_with_ngrok(app)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def bart_summarize(input_text, num_beams=4, num_words=50):
    input_text = str(input_text)
    input_text = ' '.join(input_text.split())
    input_tokenized = bart_tokenizer.encode(input_text, return_tensors='pt').to(device)
    summary_ids = bart_model.generate(input_tokenized,
                                      num_beams=int(num_beams),
                                      no_repeat_ngram_size=3,
                                      length_penalty=2.0,
                                      min_length=30,
                                      max_length=int(num_words),
                                      early_stopping=True)
    output = [bart_tokenizer.decode(g, skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False) for g in summary_ids]
    return output[0]


def t5_summarize(input_text, num_beams=4, num_words=50):
    input_text = str(input_text).replace('\n', '')
    input_text = ' '.join(input_text.split())
    input_tokenized = t5_tokenizer.encode(input_text, return_tensors="pt").to(device)
    summary_task = torch.tensor([[21603, 10]]).to(device)
    input_tokenized = torch.cat([summary_task, input_tokenized], dim=-1).to(device)
    summary_ids = t5_model.generate(input_tokenized,
                                    num_beams=int(num_beams),
                                    no_repeat_ngram_size=3,
                                    length_penalty=2.0,
                                    min_length=30,
                                    max_length=int(num_words),
                                    early_stopping=True)
    output = [t5_tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]
    return output[0]


def pegasus_cnn_summarize(input_text, num_beams=4, num_words=50):
    input_text = str(input_text)
    input_text = ' '.join(input_text.split())
    input_tokenized = pegasus_cnn_tokenizer.encode(input_text, return_tensors='pt').to(device)
    summary_ids = pegasus_cnn_model.generate(input_tokenized,
                                             num_beams=int(num_beams),
                                             no_repeat_ngram_size=3,
                                             length_penalty=2.0,
                                             min_length=30,
                                             max_length=int(num_words),
                                             early_stopping=True)
    output = [pegasus_cnn_tokenizer.decode(g, skip_special_tokens=True,
                                           clean_up_tokenization_spaces=False) for g in summary_ids]
    return output[0]


def pegasus_med_summarize(input_text, num_beams=4, num_words=50):
    input_text = str(input_text)
    input_text = ' '.join(input_text.split())
    input_tokenized = pegasus_med_tokenizer.encode(input_text, return_tensors='pt').to(device)
    summary_ids = pegasus_med_model.generate(input_tokenized,
                                             num_beams=int(num_beams),
                                             no_repeat_ngram_size=3,
                                             length_penalty=2.0,
                                             min_length=30,
                                             max_length=int(num_words),
                                             early_stopping=True)
    output = [pegasus_med_tokenizer.decode(g, skip_special_tokens=True,
                                           clean_up_tokenization_spaces=False) for g in summary_ids]
    return output[0]


@app.route('/')
def index():
    return render_template('index.html', models=models, models_str=','.join(models))


@app.route('/predict', methods=['POST'])
def predict():
    try:
        sentence = request.json['input_text']
        num_words = request.json['num_words']
        num_beams = request.json['num_beams']
        model = request.json['model']
        if sentence != '':
            if model.lower() == 'bart':
                output = bart_summarize(sentence, num_beams, num_words)
            elif model.lower() == 't5':
                output = t5_summarize(sentence, num_beams, num_words)
            elif model.lower() == 'pegasus-cnn':
                output = pegasus_cnn_summarize(sentence, num_beams, num_words)
            elif model.lower() == 'pegasus-med':
                output = pegasus_med_summarize(sentence, num_beams, num_words)
            else:
                output = None
            response = {'response': {
                'summary': str(output),
                'model': model.lower()
            }}
            return flask.jsonify(response)
        else:
            res = dict({'message': 'Empty input'})
            return app.response_class(response=json.dumps(res), status=500, mimetype='application/json')
    except Exception as ex:
        res = dict({'message': str(ex)})
        print(res)
        return app.response_class(response=json.dumps(res), status=500, mimetype='application/json')


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("-models", type=str)
    args = parser.parse_args()
    models = args.models
    print("Models names passed in Arguments:", str(models))
    # models = 'BART,T5,PEGASUS-CNN,PEGASUS-MED'
    models = models.split(',')
    if 'BART' in models:
        print('Model files downloading for BART')
        bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
        bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
        bart_model.to(device)
        bart_model.eval()
    if 'T5' in models:
        print('Model files downloading for T5')
        t5_model = T5ForConditionalGeneration.from_pretrained('t5-large')
        t5_tokenizer = T5Tokenizer.from_pretrained('t5-large')
        t5_model.to(device)
        t5_model.eval()
    if 'PEGASUS-CNN' in models:
        print('Model files downloading for PEGASUS-CNN')
        pegasus_cnn_model = PegasusForConditionalGeneration.from_pretrained('google/pegasus-cnn_dailymail')
        pegasus_cnn_tokenizer = PegasusTokenizer.from_pretrained('google/pegasus-cnn_dailymail')
        pegasus_cnn_model.to(device)
        pegasus_cnn_model.eval()
    if 'PEGASUS-MED' in models:
        print('Model files downloading for PEGASUS-MED')
        pegasus_med_model = PegasusForConditionalGeneration.from_pretrained('google/pegasus-pubmed')
        pegasus_med_tokenizer = PegasusTokenizer.from_pretrained('google/pegasus-pubmed')
        pegasus_med_model.to(device)
        pegasus_med_model.eval()
    # app.run(host='0.0.0.0', debug=True, port=8000, use_reloader=False)
    app.run()
