from src.data.data_loader import Loader
from src.data.data_preprocessing import PREPROCESS
from src.embeddings.roberta import RobertaMLM







def main(data_path, periods, **kwargs):

    print('*'*10, 'Loading data', '*'*10, '\n')
    file_type = data_path.split('.')[-1]
    corpora = {}
    if file_type == 'xml':
        xml_tag = kwargs['xml_tag']
        for period in periods:
            path = data_path.format(period)
            data = Loader.from_xml(path, xml_tag).forward()
            corpora[period] = data

    elif file_type == 'txt':
        for period in periods:
            path = file_path.format(period)
            data = Loader.from_txt(path).forward()
            corpora[period] = data
    else:
        raise ValueError('File type not supported')
    

    print('*'*10, 'Preprocessing data', '*'*10, '\n')
    preprocessing_options = kwargs['preprocessing_options']
    for period in periods:
        corpora[period] = list(map(lambda x: PREPROCESS().forward(x, **preprocessing_options), corpora[period]))
       

    print('*'*10, 'Masked language modeling (Diachronic Embeddings)', '*'*10, '\n')
        
        
    


    




if __name__ == "__main__":
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

    preprocessing_options = {"remove_stopwords": True, "remove_punctuation": True, "remove_numbers": True, "lowercase": True, "lemmatize": True}

    architecture = "Roberta"

    tokenizer = "distilroberta-base"
    tokenizer_options = {"max_length": 512, "padding": "max_length", "truncation": True}

    mlm_options = {"mlm_probability": 0.15, "max_predictions_per_seq": 20}

    model = "distilroberta-base"

    model_options = {"learning_rate": 2e-5, "num_train_epochs": 5, "weight_decay": 0.01}

    train_test_split = 0.8
    evaluation_options = {"perplexity": True}

    main(file_path, 
         periods, 
         xml_tag = 'fulltext',
         preprocessing_options = preprocessing_options,
         )











