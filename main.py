from src.data.data_loader import Loader
from src.data.data_preprocessing import PREPROCESS
from src.feature_extraction.roberta import RobertaTrainer, MaskedWordInference

import json
import os



def main(output_dir, data_path, periods, **kwargs):
    print('*'*10, 'Loading data', '*'*10, '\n')
    corpora = {}
    results = {}

    xml_tag = kwargs['xml_tag']
    for period in periods:
        path = data_path.format(period)
        corpora[period] = Loader.from_xml(path, xml_tag).forward(target_words=kwargs['target_words'], max_documents=kwargs['max_documents'], shuffle=kwargs['shuffle']) # Loader.from_txt(path).forward()

        corpora[period] = list(map(lambda x: PREPROCESS().forward(x, **kwargs['preprocessing_options']), corpora[period]))
        path = f'{output_dir}/MLM_roberta_{period}'
        trainor = RobertaTrainer(**kwargs['mlm_options'])
        trainor.train(data=corpora[period], output_dir= path)

        results[period] = {}
        MLM = MaskedWordInference(path)
        for word in kwargs['target_words']:
            results[period][word] = []
            for i, sentence in enumerate(corpora[period]):
                if word in sentence.split()[:100]:
                    t = {}
                    t['sentence'] = ' '.join(sentence.split()[:120])
                    t['top_words'], _ = MLM.get_top_k_words(word= word, sentence= sentence, k=kwargs['inference_options']['top_k'])
                    results[period][word].append(t)

    return results         
        
        




if __name__ == "__main__":
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    periods = [
        1980,
        # 1982,
        # 1985,
        # 1987,
        # 1989,
        # 1990,
        # 1992,
        # 1995,
        # 2000,
        # 2001,
        # 2005,
        # 2008,
        # 2010,
        # 2012,
        # 2015,
        # 2017
    ]

    xml_tag = 'fulltext'
    file_path = "input/xml/TheNewYorkTimes{}.xml"

    preprocessing_options = {
        "remove_stopwords": False, 
        "remove_punctuation": True, 
        "remove_numbers": False, 
        "lowercase": True, 
        "lemmatize": True
        }

    mlm_options = {
        "model_name": "roberta-base", 
        "max_length": 128,
        "mlm_probability": 0.15, 
        "batch_size": 32, 
        "epochs": 30, 
        "split_ratio": 0.8, 
        "learning_rate": 5e-5,
        "truncation": True, 
        "padding": "max_length"
        }

    inference_options = {
        "top_k": 3
        }

    target_words = [
            "office",
            "gay",
            "abuse",
            "king",
            "apple",
            "bank",
            "war",
            "love",
            "money",
            "school",
            "police",
            "family",
            "work"
        ]
    
    r = main(
        output_dir,
        file_path, 
        periods, 
        xml_tag = 'fulltext',
        target_words = target_words,
        max_documents = 10000,
        shuffle = True,
        preprocessing_options = preprocessing_options,
        mlm_options = mlm_options,
        inference_options = inference_options
        )
    
    with open(f"{output_dir}/results.json", 'w') as f:
        json.dump(r, f, indent=4)











