from src.data.data_loader import Loader
from src.data.data_preprocessing import PREPROCESS
from src.feature_extraction.roberta import RobertaTrainer, RobertaInference
from src.feature_extraction.word2vec import Word2VecTrainer, Word2VecAlign, Word2VecInference

import json
import os



def main(output_dir, data_path, periods, **kwargs):
    print('*'*10, 'Loading data', '*'*10, '\n')
    corpora = {}
    target_word =kwargs['target_word'][0]

    graph_inputs = {}

    xml_tag = kwargs['xml_tag']
    word2vec_paths = []
    for period in periods:
        path = data_path.format(period)
        corpora[period] = Loader.from_xml(
            path, 
            xml_tag
            ).forward(
                target_words=[target_word], 
                max_documents=kwargs['max_documents'], 
                shuffle=kwargs['shuffle']
                ) # Loader.from_txt(path).forward()


        corpora[period] = list(map(lambda x: 
                                    PREPROCESS().forward(
                                       x, 
                                       **kwargs['preprocessing_options']
                                    ), 
                                    corpora[period]
                                )
                            )
        
        roberta_path = f'{output_dir}/MLM_roberta_{period}'
        trainor = RobertaTrainer(**kwargs['mlm_options'])
        trainor.train(data=corpora[period], output_dir= roberta_path)

        word2vec_path = f'{output_dir}/word2vec_{period}'
        word2vec_paths.append(word2vec_path)
        trainor = Word2VecTrainer()
        trainor.train(data=corpora[period], output_dir=word2vec_path)
    

    # aligning word2vec
    aligned_word2vec_dir = f'{output_dir}/word2vec_aligned'
    align = Word2VecAlign(model_paths= word2vec_paths).align_models(reference_index=-1, output_dir=aligned_word2vec_dir, method="procrustes")

    # inference
    for i, period in enumerate(periods):
        word2vec = Word2VecInference(f'{output_dir}/word2vec_aligned/word2vec_{period}_aligned.model')
        roberta = RobertaInference(f'{output_dir}/MLM_roberta_{period}')



        context_words = word2vec.get_top_k_words(
            positive=[target_word], 
            k=kwargs['inference_options']['Context_k']
            )

        similar_words = []
        for i, doc in enumerate(corpora[period]):
            top_k_words = roberta.get_top_k_words(
                word=target_word,
                sentence=doc,
                k=kwargs['inference_options']['MLM_k']
                )
            similar_words.extend(top_k_words)

        break
    
    return context_words, similar_words
             
        
        




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
        "MLM_k": 3,
        "Context_k": 10,
        }

    target_word = [
            "office"
        ]
    
    r = main(
        output_dir,
        file_path, 
        periods, 
        xml_tag = 'fulltext',
        target_word = target_word,
        max_documents = 10000,
        shuffle = True,
        preprocessing_options = preprocessing_options,
        mlm_options = mlm_options,
        inference_options = inference_options
        )
    
    with open(f"{output_dir}/results.json", 'w') as f:
        json.dump(r, f, indent=4)











