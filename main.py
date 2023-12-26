from logging import config
from regex import P
from semantics.data.data_loader import Loader
# from semantics.data.data_loader import split_xml
from semantics.data.data_preprocessing import PREPROCESS
from semantics.feature_extraction.roberta import  RobertaInference
from semantics.feature_extraction.word2vec import Word2VecInference
from semantics.graphs.temporal_graph import TemporalGraph
from semantics.inference.visualize import WordTraffic, visualize_graph
from semantics.inference.obsedian import ObsedianGraph
from semantics.inference.graph_clustering import GraphClusterer
from semantics.utils.utils import count_occurence
from semantics.utils.components import GraphIndex
from semantics.models.tgcn import TemporalGCNTrainer, TGCNInference
import os
import numpy as np
import json
import yaml


def main(output_dir, data_path, periods, **kwargs):
    corpora = {}
    target_word = kwargs.get('target_word', None)
    xml_tag = kwargs.get('xml_tag', None)
  
    tg = TemporalGraph()
    inference_options = kwargs['inference_options']
    preprocessing_options = kwargs['preprocessing_options']
    
    for i, period in enumerate(periods[:]):

        # loading the data
        path = data_path.format(period)
        print(f'Loading data of {period} from {path} ...', '\n')

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

        print('Number of sentences with the word')
        for word in target_word:
            print(word, ': ', count_occurence(corpora[period], word), '\n')
        
        print(f'Creating the Graph of {period} ...', '\n')

        roberta_path = f'{output_dir}/MLM_roberta_{period}'
        word2vec_path = f'{output_dir}/word2vec_aligned/word2vec_{period}_aligned.model'

        roberta = RobertaInference(pretrained_model_path= roberta_path)
        word2vec = Word2VecInference(pretrained_model_path= word2vec_path)

        nodes = tg.add_graph(
            target_word= target_word,
            level = inference_options['level'],
            k = inference_options['MLM_k'],
            c = inference_options['Context_k'],
            dataset= corpora[period],
            word2vec_model= word2vec,
            mlm_model= roberta,
            edge_threshold= inference_options['edge_threshold'],
            accumulate= inference_options['accumulate']
        )

        if tg[i].edge_index.shape[1] != tg[i].edge_features.shape[0]:
            print('edge_indices and edge_features do not match')
            print('xs: ', tg[i].node_features.shape)
            print('edge_indices: ', tg[i].edge_index.shape)
            print('edge_features: ', tg[i].edge_features.shape)
            print('period: ', period)
            print('ys: ', tg[i].labels.shape)
            print('y_indices: ', tg[i].label_mask.shape)
            raise ValueError('edge_indices and edge_features do not match')
        

        if not os.path.exists(f'{output_dir}/inference_{period}'):
            os.mkdir(f'{output_dir}/inference_{period}')

        with open(f'{output_dir}/inference_{period}/nds.json', 'w') as f:
            json.dump(
                {
                    'similar_nodes': nodes.similar_nodes, 
                    'context_nodes': nodes.context_nodes
                }, f, indent=4)

    tg.align_graphs()
    tg.label_graphs()
    
    if not os.path.exists(f'{output_dir}/inference'):
        os.mkdir(f'{output_dir}/inference')  
        
    with open(f'{output_dir}/inference/index.json', 'w') as f:
        json.dump(
            {
                'key_to_index': tg[i].index.key_to_index, 
                'index_to_key': tg[i].index.index_to_key
            }, f, indent=4)
        
    for i, period in enumerate(periods):
        print(f'Saving the Graph of {period} ...')
        print('xs: ', tg[i].node_features.shape)
        print('edge_indices: ', tg[i].edge_index.shape)
        print('edge_features: ', tg[i].edge_features.shape)
        print('ys: ', tg[i].labels.shape)
        print('y_indices: ', tg[i].label_mask.shape, '\n')
        
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
    
    
    
    print('Loading the Graphs ...')
    index = []
    xs = []
    ys = []
    edge_indices = []
    edge_features = []
    y_indices = []

    for period in periods:
        with open(f'{output_dir}/inference/index.json', 'r') as f:
            d = json.load(f)
            i = GraphIndex(index_to_key= d['index_to_key'], key_to_index= d['key_to_index'])
            index.append(i)

        with open(f'{output_dir}/inference_{period}/edge_indices.npy', 'rb') as f:
            edge_indices.append(np.load(f))
        
        with open(f'{output_dir}/inference_{period}/edge_features.npy', 'rb') as f:
            edge_features.append(np.load(f))

        with open(f'{output_dir}/inference_{period}/xs.npy', 'rb') as f:
            xs.append(np.load(f))

        with open(f'{output_dir}/inference_{period}/ys.npy', 'rb') as f:
            ys.append(np.load(f))

        with open(f'{output_dir}/inference_{period}/y_indices.npy', 'rb') as f:
            y_indices.append(np.load(f))


    tg = TemporalGraph(
        index= index,
        xs= xs,
        ys= ys,
        edge_indices= edge_indices,
        edge_features= edge_features,
        y_indices= y_indices,
    )
            
    print('Creating the obsedian graph')
    visualization_options = kwargs['visualization_options']

    obsedian_vault = visualization_options.get('obsedian_vault', 'semantics-obsedian')
    view_period = visualization_options.get('view_period', 0)

    obsedian_graph = ObsedianGraph(
        vault_path= obsedian_vault,
        graph= tg[view_period],
        )
    obsedian_graph.generate_markdowns(folder= f'{periods[view_period]}', add_tag= f'{periods[view_period]}')
    obsedian_graph.JugglStyle()
    obsedian_graph.Filter(by_tag= f'{periods[view_period]}')

    
        
    # print('Modeling the temporal graph ...')

    # tg_upto2015 = TemporalGraph(
    #     index= index[:-1],
    #     xs= xs[:-1],
    #     ys= ys[:-1],
    #     edge_indices= edge_indices[:-1],
    #     edge_features= edge_features[:-1],
    #     y_indices= y_indices[:-1],
    # )

    # tg_2017 = TemporalGraph(
    #     index= index[-1:],
    #     xs= xs[-1:],
    #     ys= ys[-1:],
    #     edge_indices= edge_indices[-1:],
    #     edge_features= edge_features[-1:],
    #     y_indices= y_indices[-1:],
    # )
   
        
    # number_node_features = xs[0].shape[1]
    # number_edge_features = edge_features[0].shape[1]
    
    
    # print('Training the TGCN ...')
    # gnn = TemporalGCNTrainer(node_features= number_node_features, edge_features= number_edge_features, epochs= 100, split_ratio= 0.9, learning_rate= 0.01, device= 'cpu')

    # if not os.path.exists(f'{output_dir}/TGCN'):
    #     os.mkdir(f'{output_dir}/TGCN')
    
    # model_path = f'{output_dir}/TGCN/model'
    # gnn.train(
    #     graph= tg_upto2015,
    #     output_dir= model_path)
    
    
    # print('Loading the TGCN ...')
    # config_path = f'{model_path}.yaml'
    # with open(config_path, 'r') as f:
    #     config = yaml.load(f, Loader=yaml.FullLoader)

    # print('Predicting the last period (2017) using the trained TGCN ...')
    # tgcn = TGCNInference(pretrained_model_path= f'{model_path}.pt', **config)
    # y_hat = tgcn.predict(graph= tg_2017)
    # y = tg_2017[0].edge_features[:, 1].reshape(-1, 1)
    # mse = tgcn.mse_loss(y_hat= y_hat, y= y)
    # print('mse: ', mse, '\n')

    print('Creating the graph visualizations ...')     
    node_types = np.unique(tg[0].node_features[:, 0].tolist())
    node_colors = ['#d84c3e', '#b4f927', '#13ebef', '#4476ff', '#f9f927'][:len(node_types)]
    node_color_map = {int(val): color for val, color in zip(node_types, node_colors)}
    
    for i, period in enumerate(periods):
        print(f'Creating the plot for {period} ...')
        fig = visualize_graph(
            graph= tg[i],
            title= f'Word Graph - {periods[i]}',
            node_color_feature=0,
            node_color_map= node_color_map,
            node_size_feature= 0,
            edge_label_feature= 1,
            color_bar=True,
            color_norm= 'auto',
            target_node= target_word[0],
            )
        
        with open(f'{output_dir}/images/graph_{periods[i]}.png', 'wb') as f:
            fig.savefig(f)
    
    print('Creating the word traffic animation ...')
    wt = WordTraffic(
        temporal_graph= tg,
        title= 'Word Traffic',
        node_color_feature=0,
        node_color_map= node_color_map,
        node_size_feature= 0,
        edge_label_feature= 1,
        target_node= target_word[0],
        radius= 2,
        distance= 1,
        )
    
    wt.animate(save_path= f'{output_dir}/images/word_traffic.gif', interval= 3000, repeat= True)

    # g = GraphClusterer(graph= tg[0])
    print('Done!')

    
             
        
        




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
        "edge_threshold": 0.1,
        "accumulate": False,
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
            # "trump",
            # "biden",
            # "obama",
            # "bush",
            # "ford",
            # "nixon",
            # "putin",
            # "us",
            # "usa",
            # "america",
            # "russia",
            # "china",
            # "germany",
            # "uk",
            # "france",
            # "italy",
            # "japan",
            'abuse',
            'corruption',
            'crime',
            'misuse',
            'offense',
            'exploitation',
            'injustice',
            'misconduct',
            'misdeed'
        ]
    
    visualization_options = {
        "obsedian_vault": "semantics-obsedian",
        "view_period": -1
    }
    
    main(
        output_dir,
        file_path, 
        periods, 
        xml_tag = 'fulltext',
        target_word = target_word,
        max_documents = 10,
        shuffle = True,
        preprocessing_options = preprocessing_options,
        mlm_options = mlm_options,
        inference_options = inference_options,
        visualization_options = visualization_options
        )
    
    










