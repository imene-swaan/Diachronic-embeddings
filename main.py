from semantics.data.data_loader import Loader
# from semantics.data.data_loader import split_xml
from semantics.data.data_preprocessing import PREPROCESS
from semantics.feature_extraction.roberta import  RobertaInference
from semantics.feature_extraction.word2vec import Word2VecInference
from semantics.graphs.temporal_graph import TemporalGraph
from semantics.inference.visualize import WordTraffic, visualize_graph
from semantics.inference.obsedian import ObsedianGraph
from semantics.utils.utils import count_occurence
import os
import numpy as np
import json


def main(output_dir, data_path, periods, **kwargs):
    corpora = {}
    target_word = kwargs.get('target_word', None)
    xml_tag = kwargs.get('xml_tag', None)
    # word2vec_paths = []
    tg = TemporalGraph()
    inference_options = kwargs['inference_options']
    preprocessing_options = kwargs['preprocessing_options']

    obsedian_vault = kwargs.get('obsedian_vault', 'semantics-obsedian')


    for i, period in enumerate(periods[:]):

        # loading the data
        print(f'Loading data from {period} ...', '\n')
        path = data_path.format(period)

        corpora[period] = Loader.from_xml(
                path, 
                xml_tag
                ).sample(
                    target_words= target_word, 
                    max_documents=kwargs.get('max_documents', None),
                    shuffle=kwargs['shuffle']
                    ) # Loader.from_txt(path).forward()

        # preprocessing
        print(f'Preprocessing data from {period} ...', '\n')
        prep = PREPROCESS()
        corpora[period] = list(map(lambda x: prep.forward(text=x, **preprocessing_options), corpora[period]))


        print('Number of clean documents: ', len(corpora[period]), '\n')

        print('Number of sentences with the word: ')
        for word in target_word:
            print(word, ': ', count_occurence(corpora[period], word), '\n')
        
        print(f'Creating the Graph of {period} ...', '\n')

        roberta_path = f'{output_dir}/MLM_roberta_{period}'
        word2vec_path = f'{output_dir}/word2vec_aligned/word2vec_{period}_aligned.model'

        roberta = RobertaInference(pretrained_model_path= roberta_path)
        word2vec = Word2VecInference(pretrained_model_path= word2vec_path)

        nds = tg.add_graph(
            target_word= target_word[0],
            level = inference_options['level'],
            k = inference_options['MLM_k'],
            c = inference_options['Context_k'],
            dataset= corpora[period],
            word2vec_model= word2vec,
            mlm_model= roberta,
        )

        if tg[i].edge_index.shape[1] != tg[i].edge_features.shape[0]:
            print('edge_indices and edge_features do not match')
            print('xs: ', tg[i].node_features.shape)
            print('edge_indices: ', tg[i].edge_index.shape)
            print('edge_features: ', tg[i].edge_features.shape)
            print('period: ', period)
            print('ys: ', tg[i].labels.shape)
            print('y_indices: ', tg[i].label_mask.shape)
            raise ValueError
        

        if not os.path.exists(f'{output_dir}/inference_{period}'):
            os.mkdir(f'{output_dir}/inference_{period}')

        with open(f'{output_dir}/inference_{period}/nds.json', 'w') as f:
            json.dump(nds, f, indent=4)

    tg.align_graphs()
    tg.label_graphs()
    
    for i, period in enumerate(periods):
        print(f'Saving the Graph of {period} ...')
        print('xs: ', tg[i].node_features.shape)
        print('edge_indices: ', tg[i].edge_index.shape)
        print('edge_features: ', tg[i].edge_features.shape)
        print('ys: ', tg[i].labels.shape)
        print('y_indices: ', tg[i].label_mask.shape, '\n')

        with open(f'{output_dir}/inference_{period}/index.json', 'w') as f:
            json.dump(tg[i].index, f, indent=4)
        
        with open(f'{output_dir}/inference_{period}/edge_indices.npy', 'wb') as f:
            np.save(f, tg[i].edge_index)
        
        with open(f'{output_dir}/inference_{period}/edge_features.npy', 'wb') as f:
            np.save(f, tg[i].edge_features)
        
        with open(f'{output_dir}/inference_{period}/xs.npy', 'wb') as f:
            np.save(f, tg[i].node_features)
        
        with open(f'{output_dir}/inference_{period}/ys.npy', 'wb') as f:
            np.save(f, tg[i].labels)
        
        with open(f'{output_dir}/inference_{period}/y_indices.npy', 'wb') as f:
            np.save(f, tg[i].label_mask)

        obsedian_graph = ObsedianGraph(
            vault_path= obsedian_vault,
            graph= tg[i],
        )

        obsedian_graph.generate_markdowns(folder= f'{period}')
        obsedian_graph.style()
   
    return tg

    
             
        
        




if __name__ == "__main__":
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    periods = [
        1980,
        1982,
        1985,
        1987,
        1989,
        1990,
        1992,
        1995,
        2000,
        2001,
        2005,
        2008,
        2010,
        2012,
        2015,
        2017
    ]

    xml_tag = 'fulltext'
    file_path = "input/xml/TheNewYorkTimes{}.xml"

    preprocessing_options = {
        "remove_stopwords": True, 
        "remove_punctuation": True, 
        "remove_numbers": True, 
        "lowercase": True, 
        "lemmatize": True,
        "remove_full_stop": True,
        "remove_short_words": True
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
        "MLM_k": 2,
        "Context_k": 2,
        "level": 2,
        }

    target_word = [
            # "office",
            # "work",
            # "job",
            # "career",
            # "profession",
            # "employment",
            # "occupation",
            # "vocation",
            # "trade",
            # "business",
            # "position",
            # "post",
            "trump",
            # "biden",
            # "obama",
            # "bush",
            # "ford",
            # "nixon",
            # "putin",
            # "merkel",
            # "us",
            # "usa",
            # "america",
            # "russia",
            # "china",
            # "germany",
            # "uk",
            # "france",
            # "italy",
            # "japan"
        ]
    
    visualization_options = {
        "obsedian_vault": "semantics-obsedian"
    }
    
    tg = main(
        output_dir,
        file_path, 
        periods, 
        xml_tag = 'fulltext',
        target_word = target_word,
        max_documents = 200,
        shuffle = True,
        preprocessing_options = preprocessing_options,
        mlm_options = mlm_options,
        inference_options = inference_options,
        visualization_options = visualization_options
        )
    

    # index, xs, edge_indices, edge_features, ys, y_indices = tg[0]

    # fig = visualize_graph(
    #     graph= (index, xs, edge_indices, edge_features, ys, y_indices) ,
    #     title= 'Graph Visualization',
    #     node_label_feature= 0,
    #     edge_label_feature= 1
    #     )
    
    # image_dir = f'{output_dir}/images/graph_{periods[0]}.png'
    # if not os.path.exists(f'{output_dir}/images'):
    #     os.mkdir(f'{output_dir}/images')
    # fig.savefig(image_dir)
    










