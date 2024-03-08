from matplotlib import animation
from semantics.data.data_loader import Loader
# from semantics.data.data_loader import split_xml
from semantics.data.data_preprocessing import PREPROCESS
from semantics.feature_extraction.roberta import  RobertaInference, RobertaTrainer
from semantics.feature_extraction.word2vec import Word2VecInference, Word2VecTrainer, Word2VecAlign
from semantics.graphs.temporal_graph import TemporalGraph
from semantics.inference.visualize import WordTraffic, visualize_graph, visualize_change, animate, WordFrequency, plotTseries
from semantics.inference.obsedian import ObsedianGraph
from semantics.inference.semantic_shift import SemanticShift
from semantics.inference.graph_clustering import GraphClustering, ClusterMerger
from semantics.utils.utils import count_occurence, generate_colors
from semantics.utils.components import GraphIndex, WordGraph
from semantics.models.tgcn import TemporalGCNTrainer, TGCNInference
from semantics.models.link_prediction import TemporalGCNTrainer, LPInference
import os
import numpy as np
import json
import yaml
from pathlib import Path
from glob import glob
import pandas as pd

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
    
    save_preprocessed = pipeline['save_preprocessed']

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
        
        sample = data_options.get('sample', False)
        if sample:
            target = target_words
        else:
            target = None

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
                print(f'of {period} from {path} ...', '\n')

                corpora[period] = Loader.from_xml(
                        path, 
                        xml_tag
                        ).sample(
                            target_words= target,
                            max_documents=max_documents,
                            shuffle=shuffle,
                            random_seed=seed
                            )
                
                
                print('Length of the corpus: ', len(corpora[period]), '\n')
        elif data_options['input_type'] == 'txt':
            for i, period in enumerate(periods[:]):
                # loading the data
                path = data_path.format(period)
                print(f'of {period} from {path} ...', '\n')

                corpora[period] = Loader.from_txt(path).sample(
                    target_words= target,
                    max_documents=max_documents,
                    shuffle=shuffle,
                    random_seed=seed
                    )
                print('Length of the corpus: ', len(corpora[period]), '\n')

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
            print(f'from {period} ...', '\n')
            corpora[period] = list(map(lambda x: prep.forward(text=x, **preprocessing_options), corpora[period]))

            if save_preprocessed:
                input_dir = Path(data_path.format(period)).parent.parent
                file_name = Path(data_path.format(period)).stem

                if not os.path.exists(f'{input_dir}/txt'):
                    os.mkdir(f'{input_dir}/txt')
                save_path = f'{input_dir}/txt/{file_name}.txt'
                print(f'Saving to {save_path} ...', '\n')
                with open(save_path, 'w') as f:
                    for item in corpora[period]:
                        f.write(item + '\n')
            

            print('Number of sentences with the word ...')
            for word in target_words:
                print(word, ': ', count_occurence(corpora[period], word), '\n')



    if pipeline['train_mlm']:
        print('Training the MLM model ...')
        mlm_options = kwargs.get('mlm_options', None)
        if mlm_options is None:
            raise ValueError('MLM options are not defined')
        mlm_options['learning_rate'] = float(mlm_options['learning_rate'])
        print('MLM options: ', mlm_options, '\n')
       
        trainer = RobertaTrainer(**mlm_options)
    
        for period in periods:
            print(f'Training the MLM model for {period} ...', '\n')
            roberta_path = f'{output_dir}/MLM_roberta_{period}'
            trainer.train(corpora[period], output_dir= roberta_path)

    
    # Train the word2vec models
    if pipeline['train_word2vec']:
        print('Training the word2vec model ...')
        word2vec_options = kwargs.get('word2vec_options', None)
        if word2vec_options is None:
            raise ValueError('Word2Vec options are not defined')
        
        init_options = word2vec_options.get('initialize', None)
        train_options = word2vec_options.get('train', None)
        

        word2vec_dir = f'{output_dir}/word2vec'
        if not os.path.exists(word2vec_dir):
            os.mkdir(word2vec_dir)

        w2v_paths = []
        for period in periods:
            print(f'Training the word2vec model for {period} ...', '\n')
            word2vec_path = f'{word2vec_dir}/w2v_{period}.model'

            word2vec = Word2VecTrainer(**init_options)
            word2vec.train(
                corpora[period], 
                output_path= word2vec_path,
                **train_options)
            
            w2v_paths.append(word2vec_path)
            

    # Align the word2vec models
    if pipeline['align_word2vec']:
        word2vec_options = kwargs.get('word2vec_options', None)
        if word2vec_options is None:
            raise ValueError('Word2Vec options are not defined')

        w2v_paths = sorted(glob(f'{output_dir}/word2vec/w2v_*.model'))
        print('w2v paths: ', w2v_paths, '\n')
        word2vec_a_dir = f'{output_dir}/word2vec_aligned'
        if not os.path.exists(word2vec_a_dir):
            os.mkdir(word2vec_a_dir)

        align_options = word2vec_options.get('align', None)
        if align_options is None:
            raise ValueError('Align options are not defined')
        aligner = Word2VecAlign(model_paths= w2v_paths)
        aligner.align(output_dir= word2vec_a_dir, **align_options)
        
    
    
    # TODO: fix this
    model_paths = kwargs.get('models', None)
    if model_paths is None:
        raise ValueError('Model paths are not defined')
    ############
        
    
    # Construct the temporal graph
    if pipeline['construct_temporal_graph']:
        print('Creating the temporal Graph ...')
        tg = TemporalGraph()
        temporal_graph_options: dict = kwargs.get('temporal_graph_options', None)
        if temporal_graph_options is None:
            raise ValueError('Temporal graph options are not defined')
        
        word2vec_path_template: str = model_paths.get('word2vec_path', None)
        roberta_path_template: str = model_paths.get('roberta_path', None)

        for i, period in enumerate(periods):
            if roberta_path_template is not None:
                roberta_path = roberta_path_template.format(output_dir, period)
                roberta = RobertaInference(pretrained_model_path= roberta_path)
            else:
                raise ValueError('MLM path is not defined. Please add the path to your pretrained model to the config file. Check the RobertaInference class for more information')
            
            if word2vec_path_template is not None:
                word2vec_path = word2vec_path_template.format(output_dir, period)
                word2vec = Word2VecInference(pretrained_model_path= word2vec_path)
            
            else:
                word2vec = None
            
            print('\nMLM path: ', roberta_path, '\n')
            print('Word2Vec path: ', word2vec_path, '\n')
            
            print(f'Adding the Graph from {period} ...', '\n')

            level = temporal_graph_options.get('level', 0)
            k = temporal_graph_options.get('MLM_k', 1)
            c = temporal_graph_options.get('Context_k', 1)
            accumulate = temporal_graph_options.get('accumulate', False)
            edge_threshold = temporal_graph_options.get('edge_threshold', 0.5)
            use_context_only = temporal_graph_options.get('use_context_only', False)
            
            keep_k = temporal_graph_options.get('keep_k', None)
            if keep_k is not None:
                keep_k = dict(map(lambda item: (int(item[0]), tuple(item[1])) , keep_k.items()))
         
            
            tg.add_graph(
                target_word= target_words,
                level=  level,
                k= k,
                c= c,
                dataset= corpora[period],
                word2vec_model= word2vec,
                mlm_model= roberta,
                edge_threshold= edge_threshold,
                accumulate= accumulate,
                keep_k= keep_k,
                use_only_context= use_context_only
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
        label_feature_idx = temporal_graph_options.get('label_feature_idx', None)
        tg.label_graphs(label_feature_idx= label_feature_idx)
    
    
    
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
    
    if pipeline['descriptive_stats']:
        print('Calculating the descriptive statistics ...')
        n = []
        e = []
        for i, period in enumerate(periods):
            print(f'of {period} ...')
            
            # count the number of nodes (node features row is not all zeros)
            number_of_nodes = np.count_nonzero(tg[i].node_features, axis= 0)[0]
            number_of_edges = tg[i].edge_index.shape[1]
            n.append(number_of_nodes)
            e.append(number_of_edges)

            print('Number of nodes: ', number_of_nodes)
            print('Number of edges: ', number_of_edges, '\n')
        
     
        if not os.path.exists(f'{output_dir}/images_desc'):
            os.mkdir(f'{output_dir}/images_desc')

        fig = WordFrequency(
            data= corpora,
            main_word= target_words[0],
            labels= periods,
            type= 'relative'
            )
        
        with open(f'{output_dir}/images_desc/word_frequency.png', 'wb') as f:
            fig.savefig(f)

        fig = plotTseries(
            ts = [n, e],
            legend= ['Number of nodes', 'Number of edges'],
            time = periods,
            title = ' '
        )
        
        
        with open(f'{output_dir}/images_desc/ts.png', 'wb') as f:
            fig.savefig(f)

    

    if pipeline['visualize_wordgraph']:
        print('Creating the graph visualizations ...')  

        wordgraph_options = kwargs.get('wordgraph_options', None)
        if wordgraph_options is None:
            raise ValueError
        
        node_color_feature: int = wordgraph_options.get('node_color_feature', None)
        if node_color_feature is None:
            raise ValueError('Node color feature is not defined')
        

        node_color_map = {
            # 0: '#d84c3e',
            1: "#b4f927",
            2: "#13ebef",
            3: "#4476ff"
        }
        if not os.path.exists(f'{output_dir}/images_l1'):
            os.mkdir(f'{output_dir}/images_l1')

        for i, period in enumerate(periods):
            print(f'Creating the plot for {period}, target {target_words[0]}...')
            fig = visualize_graph(
                graph= tg[i],
                node_color_map= node_color_map,
                target_node= target_words[0],
                title= f'Graph of {period}',
                **wordgraph_options
                )

           
            with open(f'{output_dir}/images_l1/graph_{periods[i]}.png', 'wb') as f:
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
        print('Clustering the word graph ...')
        clustering_options = kwargs.get('clustering_options', None)
        if clustering_options is None:
            raise ValueError('Clustering options are not defined')
        
        k = clustering_options.get('n_clusters', None)
        method = clustering_options.get('method', None)

        clusted_tg = tg.copy()
        file_names = []
      

        clusterings = {}

        for i, period in enumerate(periods):
            print(f'of {period} ...')

            # detach the target word from the graph by removing the edges
            g = tg[i]
            edges = g.edge_index.T
            target = target_words[0]
            target_index = g.index.key_to_index[target]
            
            new_edges = []
            for edge in edges:
                if target_index in edge:
                    continue
                new_edges.append(edge)
            
            new_edges = np.array(new_edges).T
            
            g = WordGraph(
                index= g.index,
                node_features= g.node_features,
                edge_index= new_edges,
                edge_features= g.edge_features,
                labels= g.labels,
                label_mask= g.label_mask
            )


            gc = GraphClustering(graph= g)
            communities = gc.get_clusters(method= method, k = k, label= True, structure= False)
            # print('Communities: ', communities, '\n')
            
            raw_clusters = gc.get_clusters(method= method, k = k, label= False, structure= False)

            # clusterings[period] = communities
            clusterings[period] = raw_clusters


        # print('Clusterings: ', clusterings, '\n')
        # align the clusters
        aligner = ClusterMerger()
        aligner.merge_clusters(clusterings)
        aligned_clusters = aligner.get_merged_clusters()
        print('Aligned clusters: ', aligned_clusters, '\n')

        # with open(f'{output_dir}/aligned_clusters.json', 'w') as f:
        #     json.dump(aligned_clusters, f, indent=4)

        # raise ValueError('Stop here')

        with open(f'{output_dir}/aligned_clusters.json', 'r') as f:
            aligned_clusters = json.load(f)

        aligned_clusters = dict(map(lambda item: (int(item[0]), dict(map(lambda x: (int(x[0]), x[1]), item[1].items()))), aligned_clusters.items()))
        unique_cluster_ids = set(cid for clusters in aligned_clusters.values() for cid in clusters)

        total_nodes_per_period = {period: sum(len(cluster) for cluster in clusters.values()) for period, clusters in aligned_clusters.items()}

        # The sense distribution
        sense_distribution = []
        for i, period in enumerate(periods):
            d = {'period': period}
            for cluster_id in unique_cluster_ids:
                if cluster_id == -1 or cluster_id in [9,8,7,10]:
                    continue
                if cluster_id in aligned_clusters[period]:
                    d[f"cluster_{cluster_id}"] = np.round(len(aligned_clusters[period][cluster_id]) / total_nodes_per_period[period], 2)

                else:
                    d[f"cluster_{cluster_id}"] = 0
            
            sense_distribution.append(d)
        
        df = pd.DataFrame(sense_distribution)
        df.to_csv(f'{output_dir}/sense_distribution.csv', index= False)

        fig = plotTseries(
            ts= [
                df[f'cluster_{cluster_id}'] for cluster_id in [1,2,3,4,5,6]
            ],
            legend = [
                "Web-Animal",
                "Web-Network",
                "Web-Internet",
                "Web-Business",
                "Web-Social-Media",
                "Web-Search-Engine",
                # "Trump-business",
                # "Trump-Reform-Party",
                # "Trump-Election",
                # "Trump-Donald",
                # "Trump-President"
            ],
            time= periods,
            colors = [
                "#a5f20c",
                "#0ca5f2",
                "#a50cf2",
                "#f20c0c",
                "#f2a50c",
                "#0cf2a5",
                # "#f20ca5",
                # "#0cf20c",
                # "#f20c89",
                # "#0cf236"
            ]
            )
        
        with open(f'{output_dir}/images_l1/sense_distribution.png', 'wb') as f:
            fig.savefig(f)

        # visualize the clusters
        colors = generate_colors(len(unique_cluster_ids))
        np.random.shuffle(colors)
        node_color_map = {int(val): color for val, color in zip(unique_cluster_ids, colors)}
        # save node color map
        # with open(f'{output_dir}/node_color_map.json', 'w') as f:
        #     json.dump(node_color_map, f, indent=4)

        
        with open(f'{output_dir}/node_color_map.json', 'r') as f:
            node_color_map = json.load(f)
        
        node_color_map = {int(key): value for key, value in node_color_map.items()}


        for i, period in enumerate(periods):
            raw_clusters = aligned_clusters[period]

            # adding clusters to the node features
            new_feature = np.zeros((tg[i].node_features.shape[0], 1))

            for j, cluster in raw_clusters.items():
                for node in cluster:
                    new_feature[node] = j
            
            new_node_features = np.concatenate((tg[i].node_features, new_feature), axis= 1)
            clusted_tg[i] = WordGraph(
                index= tg[i].index,
                node_features= new_node_features,
                edge_index= tg[i].edge_index,
                edge_features= tg[i].edge_features,
                labels= tg[i].labels,
                label_mask= tg[i].label_mask
            )

            # visualize the clusters
            node_color_feature = -1
            # unique_clusters = np.unique(new_feature)
            # colors = generate_colors(len(unique_clusters))
            # node_color_map = {int(val): color for val, color in zip(unique_clusters, colors)}
              

            if not os.path.exists(f'{output_dir}/cluster_images_emp'):  
                os.mkdir(f'{output_dir}/cluster_images_emp')

            
            fig = visualize_graph(
                graph= clusted_tg[i],
                title= f'Graph of {period}',
                node_color_feature= node_color_feature,
                node_color_map= node_color_map,
                target_node= target_words[0],
                legend= True,
                **clustering_options['wordgraph']
                )
            
            with open(f'{output_dir}/cluster_images_emp/graph_{periods[i]}.png', 'wb') as f:
                fig.savefig(f)

            file_names.append(f'{output_dir}/cluster_images_emp/graph_{periods[i]}.png')




        animate(
            filenames = file_names, 
            save_path= f'{output_dir}/cluster_images_emp')
       
        
    

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

        view_graph = obsedian_options.get('view_graph', "wordgraph")
        if view_graph == "wordgraph":
            obsedian_graph = ObsedianGraph(
                vault_path= obsedian_vault,
                graph= tg[view_period],
                )
        
        elif view_graph == "clustering":
            obsedian_graph = ObsedianGraph(
                vault_path= obsedian_vault,
                graph= clusted_tg[view_period],
                )
        
        else:
            raise ValueError('View graph is not defined')
        
        obsedian_graph.generate_markdowns(folder= f'{periods[view_period]}', add_tag= f'{periods[view_period]}')
        obsedian_graph.JugglStyle()
        obsedian_graph.Filter(by_tag= f'{periods[view_period]}')
    


    if pipeline['fill_temporal_graph']:
        print('Filling the temporal Graph ...')
        roberta_path_template: str = model_paths.get('roberta_path', None)
        if roberta_path_template is None:
            raise ValueError('MLM path is not defined. Please add the path to your pretrained model to the config file. Check the RobertaInference class for more information')
        
        filled_tg = tg.copy()
        

        for i, period in enumerate(periods):
            roberta_path = roberta_path_template.format(output_dir, period)
            roberta = RobertaInference(pretrained_model_path= roberta_path)
            print(f'of {period} ...')
            filled_tg.ffill(
                snap_index= i, 
                dataset= corpora[period],
                mlm_model= roberta
                )
        
        filled_tg.label_graphs(label_feature_idx= None)
    

    if pipeline['save_filled_temporal_graph']:
        print('Saving the filled temporal Graph ...')
        if not os.path.exists(f'{output_dir}/inference_filled'):
            os.mkdir(f'{output_dir}/inference_filled') 

        with open(f'{output_dir}/inference_filled/index.json', 'w') as f:
            json.dump(dict(filled_tg[0].index), f, indent=4)
            
        
        for i, period in enumerate(periods):
            print(f'Saving the Graph of {period} ...')
            print('xs: ', filled_tg[i].node_features.shape)
            print('edge_indices: ', filled_tg[i].edge_index.shape)
            print('edge_features: ', filled_tg[i].edge_features.shape)
            print('ys: ', filled_tg[i].labels.shape)
            print('y_indices: ', filled_tg[i].label_mask.shape, '\n')

            if not os.path.exists(f'{output_dir}/inference_filled/inference_{period}'):
                os.mkdir(f'{output_dir}/inference_filled/inference_{period}')
            
            # nodes = dict(filled_tg.nodes[i])

            # with open(f'{output_dir}/inference_filled/inference_{period}/nodes.json', 'w') as f:
            #     json.dump(nodes, f, indent=4)

            with open(f'{output_dir}/inference_filled/inference_{period}/edge_indices.npy', 'wb') as f:
                np.save(f, filled_tg[i].edge_index)

            with open(f'{output_dir}/inference_filled/inference_{period}/edge_features.npy', 'wb') as f:
                np.save(f, filled_tg[i].edge_features)

            with open(f'{output_dir}/inference_filled/inference_{period}/xs.npy', 'wb') as f:
                np.save(f, filled_tg[i].node_features)

            with open(f'{output_dir}/inference_filled/inference_{period}/ys.npy', 'wb') as f:
                np.save(f, filled_tg[i].labels)

            with open(f'{output_dir}/inference_filled/inference_{period}/y_indices.npy', 'wb') as f:
                np.save(f, filled_tg[i].label_mask)


    if pipeline['load_filled_temporal_graph']:
        print('Loading the filled Graphs ...')
        xs = []
        ys = []
        edge_indices = []
        edge_features = []
        y_indices = []

        with open(f'{output_dir}/inference_filled/index.json', 'r') as f:
            d = json.load(f)

        index_to_key = dict(map(lambda item: (int(item[0]), item[1]), d['index_to_key'].items()))
        graph_index = GraphIndex(index_to_key= index_to_key, key_to_index= d['key_to_index'])
        index = [graph_index] * len(periods)

        for period in periods:
            print(f'Loading the filled Graph of {period} ...', '\n')
            with open(f'{output_dir}/inference_filled/inference_{period}/edge_indices.npy', 'rb') as f:
                edge_indices.append(np.load(f))
            
            with open(f'{output_dir}/inference_filled/inference_{period}/edge_features.npy', 'rb') as f:
                edge_features.append(np.load(f))

            with open(f'{output_dir}/inference_filled/inference_{period}/xs.npy', 'rb') as f:
                xs.append(np.load(f))

            with open(f'{output_dir}/inference_filled/inference_{period}/ys.npy', 'rb') as f:
                ys.append(np.load(f))

            with open(f'{output_dir}/inference_filled/inference_{period}/y_indices.npy', 'rb') as f:
                y_indices.append(np.load(f))


        filled_tg = TemporalGraph(
            index= index,
            xs= xs,
            ys= ys,
            edge_indices= edge_indices,
            edge_features= edge_features,
            y_indices= y_indices,
        )

    if pipeline['train_link_prediction']:
        print('Training the link prediction model ...')
        lp_path = model_paths['lp_path'].format(output_dir)
        lp_dir = str(Path(lp_path).parent)

        if not os.path.exists(lp_dir):
            os.mkdir(lp_dir)
        
        tgcn = TemporalGCNTrainer(
            node_features= filled_tg[0].node_features.shape[1],
            edge_features= filled_tg[0].edge_features.shape[1],
            size= 512,
            epochs= 200,
            learning_rate= 0.001,
        ).train(
            graph= filled_tg,
            output_dir= lp_path
        )
        
       
    
    if pipeline['train_tgcn']:
        print('Modeling the temporal graph with TGCN ...')

        tgcn_options = kwargs.get('tgcn_options', None)
        if tgcn_options is None:
            raise ValueError('TGCN options are not defined')
        

        tg_upto2012 = filled_tg.copy()
        del tg_upto2012[-1] # remove 2017
        del tg_upto2012[-1] # remove 2015


        number_node_features = xs[0].shape[1]
        number_edge_features = edge_features[0].shape[1]

        gnn = TemporalGCNTrainer(node_features= number_node_features, edge_features= number_edge_features, **tgcn_options)

        tgcn_path = model_paths['tgcn_path'].format(output_dir)
        tgcn_dir = str(Path(tgcn_path).parent)

        if not os.path.exists(tgcn_dir):
            os.mkdir(tgcn_dir)
        
        gnn.train(
            graph= tg_upto2012,
            output_dir= tgcn_path)
    
    if pipeline['load_tgcn']:
        lp_path = model_paths['lp_path'].format(output_dir)

        with open(f'{lp_path}.yaml', 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        
        tgcn = LPInference(
            pretrained_model_path= f'{lp_path}.pt',
            **config
        )

        losses = config['losses'][:100]

        fig = plotTseries(
            ts= [losses],
            legend= ['Loss'],
            title= 'The Training Loss',
        )
        with open(f'{output_dir}/images_desc/loss.png', 'wb') as f:
            fig.savefig(f)

    
    if pipeline['Infer_tgcn']:
        tg_2015 = filled_tg.copy()
        del tg_2015[-1] # remove 2017
        del tg_2015[:-1] # remove everything before 2015

        tg_2017 = filled_tg.copy()
        del tg_2017[:-1] # remove everything before 2017

        y = tg_2015[0].labels.reshape(-1, 1)
        print('y: ', y, '\n')
        print('Shape of y: ', y.shape, '\n')

        y_hat = tgcn.predict(graph= tg_2015)

       
        
        

        import torch
        num_nodes = tg_2017[0].node_features.shape[0]
        indices = torch.triu_indices(num_nodes, num_nodes, offset=1).numpy().T
        # convert list of lists to list of tuples
        indices = list(map(tuple, indices))

        # take only the indices where y_hat is 1
        indices = [indices[i] for i in np.where(y_hat == 1)[0]]
        indices = np.array(indices).T

        y_hat_graph_2017  = WordGraph(
            index= tg_2017[0].index,
            node_features= tg_2017[0].node_features,
            edge_index= indices,
            edge_features= tg_2017[0].edge_features,
            labels= tg_2017[0].labels,
            label_mask= tg_2017[0].label_mask

        )

        # clusted_tg[i] = WordGraph(
        #     index= tg[i].index,
        #     node_features= new_node_features,
        #     edge_index= tg[i].edge_index,
        #     edge_features= tg[i].edge_features,
        #     labels= tg[i].labels,
        #     label_mask= tg[i].label_mask
        # )

        # visualize the clusters
       
        # unique_clusters = np.unique(new_feature)
        # colors = generate_colors(len(unique_clusters))
        # node_color_map = {int(val): color for val, color in zip(unique_clusters, colors)}
            

        if not os.path.exists(f'{output_dir}/cluster_images'):  
            os.mkdir(f'{output_dir}/cluster_images')

        
        fig = visualize_graph(
            graph= y_hat_graph_2017,
            title= f'Predicted Graph of {period}',
            # node_color_feature= node_color_feature,
            # node_color_map= node_color_map,
            target_node= target_words[0],
            legend= False,
            color_bar=True
            # **clustering_options['wordgraph']
            )
        
        with open(f'{output_dir}/cluster_images/graph_{periods[i]}_predicted.png', 'wb') as f:
            fig.savefig(f)



    
    if pipeline['Tgcn_embeddings']:
        tgcn_path = model_paths['tgcn_path'].format(output_dir)
        tgcn_dir = str(Path(tgcn_path).parent)

        embeddings = tgcn.get_embedding(graph= tg, to_vector= 'flatten')
        print('Embeddings length: ', len(embeddings), '\n')

       
        # save the embeddings
        if not os.path.exists(f'{tgcn_dir}/embeddings'):
            os.mkdir(f'{tgcn_dir}/embeddings')
        
        for i, period in enumerate(periods):
            with open(f'{tgcn_dir}/embeddings/emb_{period}.npy', 'wb') as f:
                # print('Shape of embeddings: ', embeddings[i].shape, '\n')
                np.save(f, embeddings[i])


    if pipeline['load_embeddings']:
        tgcn_path = model_paths['tgcn_path'].format(output_dir)
        tgcn_dir = str(Path(tgcn_path).parent)

        embeddings = []
        for i, period in enumerate(periods):
            with open(f'{tgcn_dir}/embeddings/emb_{period}.npy', 'rb') as f:
                embeddings.append(np.load(f))
        
        # print('Embeddings length: ', len(embeddings), '\n')

    
    if pipeline['semantic_score']:
        if embeddings is None:
            raise ValueError('Embeddings are not defined')
        
        print('Calculating the semantic shift score ...')
        ss = SemanticShift(embeddings= embeddings)
        sorted_pairs = ss.get_pair_shift(labels = periods, top_n= 10)
        for pair in sorted_pairs:
            print(f"Pair {pair[0]}: Shift = {pair[1]}")
        
        print('\nCalculating the sequence shift score ...')
        sequence_shift = ss.get_sequence_shift(labels= periods, to_score= True)
        for pair in sequence_shift:
            print(f"Pair {pair[0]}: Shift = {pair[1]}")
        
        print('\nCalculating the reference shift score ...')
        reference_shift = ss.get_ref_shift(ref= -1, labels= periods)
        for pair in reference_shift:
            print(f"Pair {pair[0]}: Shift = {pair[1]}")
        
        print('\nCalculating the changepoints ...')
        breaks = ss.ChangePointDetection(
            penalty=0.1,
            labels= periods
            )
        print('Breaks: ', breaks, '\n')

        fig = visualize_change(
            ts = [x[1] for x in reference_shift],
            breaks = breaks,
            periods = periods
        )
        with open(f'{output_dir}/images/change.png', 'wb') as f:
            fig.savefig(f)
    

    
    print('Done!')

    
             
        







if __name__ == "__main__":
    with open('config.yaml', 'r') as f:
        configuration = yaml.load(f, Loader=yaml.FullLoader)

    main(**configuration)
    
    










