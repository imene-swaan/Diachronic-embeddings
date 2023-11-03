from semantics.data.data_loader import Loader
from semantics.data.data_preprocessing import PREPROCESS
from semantics.data.data_loader import split_xml
from semantics.feature_extraction.roberta import RobertaTrainer, RobertaInference
from semantics.feature_extraction.word2vec import Word2VecTrainer, Word2VecAlign, Word2VecInference
from semantics.graphs.temporal_graph import Nodes
import numpy as np
from pathlib import Path
import os



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
        print(f'Training Roberta from {period} ...', '\n')
        roberta_path = f'{output_dir}/MLM_roberta_{period}'
        trainor = RobertaTrainer(**kwargs['mlm_options'])
        trainor.train(
            data = corpora[period],
            output_dir = roberta_path
            )


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


    # feature extraction
    # print(f'Extracting features ...', '\n')
    # graph_inputs = {
    #     "nodes": [],
    #     "node_types": [],
    #     "node_features": [],
    #     "edges": [],
    #     "edge_features": [],
    #     "edge_types": []
    # }

    # for i, period in enumerate(periods):
    #     print(f'Extracting features from {period} ...', '\n')
    #     nodes = Nodes()

    #     print(f'Extracting context words ...', '\n')
    #     context_nodes, _ = nodes.get_context_nodes(
    #         word=target_word,
    #         model_path=f'{output_dir}/word2vec_aligned/word2vec_{period}_aligned.model',
    #         k=kwargs['inference_options']['Context_k']
    #         )
    #     print('Length of context nodes: ', len(context_nodes), '\n')
    #     print('Context nodes: ', context_nodes, '\n')


        

    #     print(f'Extracting similar words ...', '\n')
    #     similar_words = []
    #     for i, doc in enumerate(corpora[period][:20]):
    #         top_k_words = nodes.get_similar_nodes(
    #             word=target_word,
    #             sentence=doc,
    #             model_path=f'{output_dir}/MLM_roberta_{period}',
    #             k=kwargs['inference_options']['MLM_k']
    #             )
            
    #         if len(top_k_words) > 0:
    #             similar_words.extend(top_k_words) 

    #     similar_nodes = list(set(similar_words))
    #     print('Length of similar nodes: ', len(similar_nodes), '\n')
    #     print('Similar nodes: ', similar_nodes, '\n')





        # print(f'Creating graph inputs ...', '\n')

        # node_types = [] # 1 for context, 0 for similar
        # nodes = []
        # for node, type in zip(context_nodes + similar_nodes, [1] * len(context_nodes) + [0] * len(similar_nodes)):
        #     if node not in nodes:
        #         nodes.append(node)
        #         node_types.append(type)
        
        # graph_inputs['nodes'].append(nodes)
        # graph_inputs['node_types'].append(node_types)

        # print('Length of nodes: ', len(nodes), '\n')
        # print('Length of node types: ', len(node_types), '\n')
        # print('Example: ', nodes[:], '\n', node_types[:], '\n')


        # print(f'Node features ...', '\n')
        # word_embeddings = {node: [] for node in nodes}
        # for i, doc in enumerate(corpora[period]):
        #     for node in nodes:
        #         if node in doc:
        #             embedding = roberta.get_embedding(word=node, sentence=doc, mask=False)
        #             word_embeddings[node].append(embedding)
        
        # node_features = [np.mean(v) for _,v in word_embeddings.items()]
        # print('Length of node features: ', len(node_features), '\n')
        # graph_inputs['node_features'].append(node_features)
        
            
    # return graph_inputs
             
        
        




if __name__ == "__main__":
    output_dir = 'trial1'
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
        "Context_k": 30,
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
        max_documents = 10,
        shuffle = True,
        preprocessing_options = preprocessing_options,
        mlm_options = mlm_options,
        inference_options = inference_options
        )
    
    # with open(f"{output_dir}/results.json", 'w') as f:
    #     json.dump(r, f, indent=4)











