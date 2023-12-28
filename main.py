from matplotlib import animation
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
from pathlib import Path

def main(**kwargs):
    #load the pipeline of the experiment
    pipeline = kwargs.get('pipeline', None)
    if pipeline is None:
        raise ValueError('Pipeline is not defined')
    
    # create the output directory if it does not exist
    output_dir = kwargs.get('output_dir', None)
    if output_dir is None:
        raise ValueError('Output directory is not defined')
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    # load the periods of the experiment
    periods = kwargs.get('periods', None)
    if periods is None:
        raise ValueError('Periods are not defined')
    
    # load the target words of the experiment
    target_words = kwargs.get('target_words', None)
    if target_words is None:
        raise ValueError('Target words are not defined')
    

    # load the corpora
    if pipeline['load_data']:
        print('Loading the data ... ')
        corpora = {}
    
        data_options = kwargs.get('data_options', None)
        if data_options is None:
            raise ValueError('Data options are not defined')
        
        
        data_path = data_options.get('data_path', None)
        if data_path is None:
            raise ValueError('Data path is not defined')
        
        max_documents = data_options.get('max_documents', None)
        shuffle = data_options.get('shuffle', None)
        seed = data_options.get('seed', None)

        if data_options['input_type'] == 'xml':
            xml_tag = data_options.get('xml_tag', None)
            if xml_tag is None:
                raise ValueError('XML tag is not defined')
        
            
            for i, period in enumerate(periods[:]):
                # loading the data
                path = data_path.format(period)
                print(f'Loading data of {period} from {path} ...', '\n')

                corpora[period] = Loader.from_xml(
                        path, 
                        xml_tag
                        ).sample(
                            target_words= target_words, 
                            max_documents=max_documents,
                            shuffle=shuffle,
                            random_seed=seed
                            )
                
        elif data_options['input_type'] == 'txt':
            for i, period in enumerate(periods[:]):
                # loading the data
                path = data_path.format(period)
                print(f'Loading data of {period} from {path} ...', '\n')

                corpora[period] = Loader.from_txt(path).sample(
                    target_words= target_words, 
                    max_documents=max_documents,
                    shuffle=shuffle,
                    random_seed=seed
                    )

        else:
            raise ValueError('Input type is not Implemented')
        
    # Preprocess the corpora
    if pipeline['preprocess']:
        print('Preprocessing the data ...')
        preprocessing_options = kwargs.get('preprocessing_options', None)
        if preprocessing_options is None:
            raise ValueError('Preprocessing options are not defined')

        prep = PREPROCESS()
        for period in periods:
            print(f'Preprocessing data from {period} ...', '\n')
            corpora[period] = list(map(lambda x: prep.forward(text=x, **preprocessing_options), corpora[period]))

            print('Number of sentences with the word')
            for word in target_words:
                print(word, ': ', count_occurence(corpora[period], word), '\n')


    if pipeline['train_mlm']:
        pass

    if pipeline['train_word2vec']:
        pass
    
    
    # TODO: fix this
    model_paths = kwargs.get('models', None)
    if model_paths is None:
        raise ValueError('Model paths are not defined')
    ############
        
    
    if pipeline['construct_temporal_graph']:
        print('Creating the temporal Graph ...')
        tg = TemporalGraph()
        temporal_graph_options = kwargs.get('temporal_graph_options', None)
        if temporal_graph_options is None:
            raise ValueError('Temporal graph options are not defined')
        
        for i, period in enumerate(periods):
            roberta_path = model_paths['roberta_path'].format(output_dir, period)
            word2vec_path = model_paths['word2vec_path'].format(output_dir, period)
            print(f'Adding the Graph from {period} ...', '\n')
            
            keep_k = dict(map(lambda item: (int(item[0]), tuple(item[1])) , temporal_graph_options['keep_k'].items()))
         
            
            tg.add_graph(
                target_word= target_words,
                level=  temporal_graph_options['level'],
                k= temporal_graph_options['MLM_k'],
                c= temporal_graph_options['Context_k'],
                dataset= corpora[period],
                word2vec_model= Word2VecInference(pretrained_model_path= word2vec_path),
                mlm_model= RobertaInference(pretrained_model_path= roberta_path),
                edge_threshold= temporal_graph_options['edge_threshold'],
                accumulate= temporal_graph_options['accumulate'],
                keep_k= keep_k
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
            

        tg.align_graphs()
        tg.label_graphs()
    
    if pipeline['save_temporal_graph']:
        print('Saving the temporal Graph ...')
        if not os.path.exists(f'{output_dir}/inference'):
            os.mkdir(f'{output_dir}/inference') 

        with open(f'{output_dir}/inference/index.json', 'w') as f:
            json.dump(dict(tg[0].index), f, indent=4)
            
        
        for i, period in enumerate(periods):
            print(f'Saving the Graph of {period} ...')
            print('xs: ', tg[i].node_features.shape)
            print('edge_indices: ', tg[i].edge_index.shape)
            print('edge_features: ', tg[i].edge_features.shape)
            print('ys: ', tg[i].labels.shape)
            print('y_indices: ', tg[i].label_mask.shape, '\n')

            if not os.path.exists(f'{output_dir}/inference_{period}'):
                os.mkdir(f'{output_dir}/inference_{period}')

            nodes = dict(tg.nodes[i])
            del nodes['target_nodes']

            with open(f'{output_dir}/inference_{period}/nodes.json', 'w') as f:
                json.dump(nodes, f, indent=4)

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


        

    if pipeline['load_temporal_graph']:
        print('Loading the Graphs ...')
        xs = []
        ys = []
        edge_indices = []
        edge_features = []
        y_indices = []

        with open(f'{output_dir}/inference/index.json', 'r') as f:
            d = json.load(f)

        index_to_key = dict(map(lambda item: (int(item[0]), item[1]), d['index_to_key'].items()))
        graph_index = GraphIndex(index_to_key= index_to_key, key_to_index= d['key_to_index'])
        index = [graph_index] * len(periods)

        for period in periods:
            print(f'Loading the Graph of {period} ...', '\n')
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
    
    if pipeline['visualize_obsedian']:
        print('Creating the obsedian graph')
        obsedian_options = kwargs.get('obsedian_options', None)
        if obsedian_options is None:
            raise ValueError('Obsedian options are not defined')

        obsedian_vault = obsedian_options.get('obsedian_vault', None)
        if obsedian_vault is None:
            raise ValueError('Obsedian vault is not defined')
        
        view_period = obsedian_options.get('view_period', None)
        if view_period is None:
            raise ValueError('View period (snapshot) is not defined')

        obsedian_graph = ObsedianGraph(
            vault_path= obsedian_vault,
            graph= tg[view_period],
            )
        obsedian_graph.generate_markdowns(folder= f'{periods[view_period]}', add_tag= f'{periods[view_period]}')
        obsedian_graph.JugglStyle()
        obsedian_graph.Filter(by_tag= f'{periods[view_period]}')
    
    if pipeline['train_tgcn']:
        print('Modeling the temporal graph with TGCN ...')

        tgcn_options = kwargs.get('tgcn_options', None)
        if tgcn_options is None:
            raise ValueError('TGCN options are not defined')
        

        tg_upto2015 = TemporalGraph(
            index= index[:-1],
            xs= xs[:-1],
            ys= ys[:-1],
            edge_indices= edge_indices[:-1],
            edge_features= edge_features[:-1],
            y_indices= y_indices[:-1],
        )

        number_node_features = xs[0].shape[1]
        number_edge_features = edge_features[0].shape[1]

        gnn = TemporalGCNTrainer(node_features= number_node_features, edge_features= number_edge_features, **tgcn_options)

        tgcn_path = model_paths['tgcn_path'].format(output_dir)
        tgcn_dir = str(Path(tgcn_path).parent)

        if not os.path.exists(tgcn_dir):
            os.mkdir(tgcn_dir)
        
        gnn.train(
            graph= tg_upto2015,
            output_dir= tgcn_path)
    
    if pipeline['load_tgcn']:
        tgcn_path = model_paths['tgcn_path'].format(output_dir)

        with open(f'{tgcn_path}.yaml', 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        
        tgcn = TGCNInference(
            pretrained_model_path= f'{tgcn_path}.pt',
            **config
        )
    
    if pipeline['Infer_tgcn']:
        tg_2017 = TemporalGraph(
            index= index[-1:],
            xs= xs[-1:],
            ys= ys[-1:],
            edge_indices= edge_indices[-1:],
            edge_features= edge_features[-1:],
            y_indices= y_indices[-1:],
        )

        y_hat = tgcn.predict(graph= tg_2017)
        y = tg_2017[0].edge_features[:, 1].reshape(-1, 1)
        mse = tgcn.mse_loss(y_hat= y_hat, y= y)
        print('mse: ', mse, '\n')
    
    if pipeline['visualize_wordgraph']:
        print('Creating the graph visualizations ...')  
        node_types = np.unique(tg[0].node_features[:, 0].tolist())
        node_colors = ['#d84c3e', '#b4f927', '#13ebef', '#4476ff', '#f9f927'][:len(node_types)]
        node_color_map = {int(val): color for val, color in zip(node_types, node_colors)}

        wordgraph_options = kwargs.get('wordgraph_options', None)
        if wordgraph_options is None:
            raise ValueError

        for i, period in enumerate(periods):
            print(f'Creating the plot for {period} ...')
            fig = visualize_graph(
                graph= tg[i],
                node_color_map= node_color_map,
                target_node= target_words[0],
                **wordgraph_options
                )
            
            if not os.path.exists(f'{output_dir}/images'):
                os.mkdir(f'{output_dir}/images')

            with open(f'{output_dir}/images/graph_{periods[i]}.png', 'wb') as f:
                fig.savefig(f)
    
    if pipeline['animate_wordgraph']:
        print('Creating the word traffic animation ...')
        animation_options = kwargs.get('animation_options', None)
        if animation_options is None:
            raise ValueError
        
        wt = WordTraffic(
            temporal_graph= tg,
            node_color_map= node_color_map,
            target_node= target_words[0],
            **animation_options['wordgraph']
            )
        save_path = f'{output_dir}/images/word_traffic.gif'
        wt.animate(save_path= save_path, **animation_options['gif'])

    if pipeline['cluster_wordgraph']:
        # g = GraphClusterer(graph= tg[0])
        pass
    
    print('Done!')

    
             
        
        




if __name__ == "__main__":
    with open('config.yaml', 'r') as f:
        configuration = yaml.load(f, Loader=yaml.FullLoader)

    main(**configuration)
    
    










