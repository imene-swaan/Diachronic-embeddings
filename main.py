from semantics.data.data_loader import Loader
from semantics.data.data_loader import split_xml
from semantics.data.data_preprocessing import PREPROCESS
from semantics.feature_extraction.roberta import  RobertaInference
from semantics.feature_extraction.word2vec import Word2VecInference
from semantics.graphs.temporal_graph import TemporalGraph
from pathlib import Path
import os
import numpy as np


def main(output_dir, data_path, periods, **kwargs):
    corpora = {}
    target_word = kwargs['target_word']
    xml_tag = kwargs['xml_tag']
    word2vec_paths = []


    for period in periods:

        # loading the data
        print(f'Loading data from {period} ...', '\n')
        path = data_path.format(period)

        try:
            corpora[period] = Loader.from_xml(
                path, 
                xml_tag
                ).forward(
                    target_words= target_word, 
                    max_documents=kwargs['max_documents'], 
                    shuffle=kwargs['shuffle']
                    ) # Loader.from_txt(path).forward()
        except ValueError:
            data_dir = Path(path).parent
            file_name = Path(path).stem
            i = 0
            corpora[period] = []

            while os.path.exists(f"{data_dir}/{file_name}_{i}.xml"):
                corpora[period].extend(
                    Loader.from_xml(
                        path= f"{data_dir}/{file_name}_{i}.xml", 
                        tag= xml_tag
                        ).forward(
                            target_words=target_word, 
                            max_documents=kwargs['max_documents'], 
                            shuffle=kwargs['shuffle']
                            )
                    )
                i += 1

            # split_paths = split_xml(
            #     path= path,
            #     output_dir= f'{data_dir}',
            #     max_children= 1000
            #     )
               
            # if i < len(split_paths):
            #     corpora[period] = []
            #     for split_path in split_paths:
            #         corpora[period].extend(
            #             Loader.from_xml(
            #                 split_path, 
            #                 xml_tag
            #                 ).forward(
            #                     target_words=[target_word], 
            #                     max_documents=kwargs['max_documents'], 
            #                     shuffle=kwargs['shuffle']
            #                     )
            #             ) # Loader.from_txt(split_path).forward()
            #         os.remove(split_path)
               
            

        # preprocessing
        print(f'Preprocessing data from {period} ...', '\n')
        corpora[period] = list(map(lambda x: 
                                    PREPROCESS().forward(
                                       x, 
                                       **kwargs['preprocessing_options']
                                    ), 
                                    corpora[period]
                                )
                            )
        #training the models
        # print(f'Training Roberta from {period} ...', '\n')
        # roberta_path = f'{output_dir}/MLM_roberta_{period}'
        # trainor = RobertaTrainer(**kwargs['mlm_options'])
        # trainor.train(
        #     data = corpora[period],
        #     output_dir = roberta_path
        #     )


        # print(f'Training Word2Vec from {period} ...', '\n')
        # word2vec_path = f'{output_dir}/word2vec'
        # if not os.path.exists(word2vec_path):
        #     os.mkdir(word2vec_path)

    #     sentences = list(map(lambda x: 
    #                          PREPROCESS().forward(
    #                              x, 
    #                              remove_stopwords =True
    #                             ), 
    #                             corpora[period]
    #                         )
    #                     )
    #     sentences = [sentence.split() for sentence in sentences]
    #     trainor = Word2VecTrainer()
    #     trainor.train(
    #         data=sentences, 
    #         output_dir= f'{word2vec_path}/word2vec_{period}.model',
    #         epochs= 10
    #         )
    #     del sentences
    #     word2vec_paths.append(f'{word2vec_path}/word2vec_{period}.model')
    #     words = list(trainor.model.wv.key_to_index)
    #     if 'office' not in words:
    #         print('\n\n\n office not in vocab \n\n\n')
            
    

    # # aligning word2vec
    # print(f'Aligning Word2Vec models ...', '\n')
    # aligned_word2vec_dir = f'{output_dir}/word2vec_aligned'
    # if not os.path.exists(aligned_word2vec_dir):
    #     os.mkdir(aligned_word2vec_dir)

    # Word2VecAlign(
    #     model_paths= word2vec_paths
    #     ).align_models(
    #         reference_index=-1, 
    #         output_dir=aligned_word2vec_dir, 
    #         method="procrustes"
    #         )


    # Creating the Temporal Graph
    print(f'Creating the Temporal Graph ...', '\n')

    tg = TemporalGraph()
    inference_options = kwargs['inference_options']

    for i, period in enumerate(periods):
        print(f'Creating the Graph of {period} ...', '\n')

        roberta_path = f'{output_dir}/MLM_roberta_{period}'
        word2vec_path = f'{output_dir}/word2vec_aligned/word2vec_{period}_aligned.model'

        roberta = RobertaInference(pretrained_model_path= roberta_path)
        word2vec = Word2VecInference(pretrained_model_path= word2vec_path)

        tg.add_graph(
            target_word= target_word[0],
            level = inference_options['level'],
            k = inference_options['MLM_k'],
            c = inference_options['Context_k'],
            dataset= corpora[period],
            word2vec_model= word2vec,
            mlm_model= roberta,
        )
    

    return tg

    
             
        
        




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
        "epochs": 3, 
        "split_ratio": 0.8, 
        "learning_rate": 5e-5,
        "truncation": True, 
        "padding": "max_length"
        }

    
    inference_options = {
        "MLM_k": 3,
        "Context_k": 2,
        "level": 2,
        }

    target_word = [
            "office"
        ]
    
    tg = main(
        output_dir,
        file_path, 
        periods, 
        xml_tag = 'fulltext',
        target_word = target_word,
        max_documents = 25000,
        shuffle = True,
        preprocessing_options = preprocessing_options,
        mlm_options = mlm_options,
        inference_options = inference_options
        )
    
    i = 0
    index, node_features, edge_indices, edge_features, labels, label_indices = tg[i]

    print(f"""Example of graph snapshot {periods[i]}: \n
      Node index shape: {index.shape} \n
      Node feature shape: {node_features.shape} \n
      Edge index shape: {edge_indices.shape} \n
      Edge feature shape: {edge_features.shape} \n 
      Labels shape: {labels.shape} \n
      Labels mask shape: {label_indices.shape}
      """)


    xs = np.array(tg.xs)
    edge_indices = np.array(tg.edge_indices)
    edge_features = np.array(tg.edge_features)
    ys = np.array(tg.ys)
    y_indices = np.array(tg.y_indices)

    with open(f'{output_dir}/xs.npy', 'wb') as f:
        np.save(f, xs)
    
    with open(f'{output_dir}/edge_indices.npy', 'wb') as f:
        np.save(f, edge_indices)

    with open(f'{output_dir}/edge_features.npy', 'wb') as f:
        np.save(f, edge_features)

    with open(f'{output_dir}/ys.npy', 'wb') as f:
        np.save(f, ys)

    with open(f'{output_dir}/y_indices.npy', 'wb') as f:
        np.save(f, y_indices)










