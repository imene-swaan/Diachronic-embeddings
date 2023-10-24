import re




class PREPROCESS():
    def __init__(self):
        """
        This class is used to preprocess text data.
        """
        pass

    def forward(self,
                text,
                remove_punctuation: bool = True,
                remove_numbers: bool = True,
                lowercase: bool = True,
                lemmatize: bool = True,
                remove_stopwords: bool = True,            
        ):

        """
        This function preprocesses text data.

        Args:
            text (str): Text to be preprocessed.
            remove_punctuation (bool): Whether to remove punctuation. Defaults to True.
            remove_numbers (bool): Whether to remove numbers. Defaults to True.
            lowercase (bool): Whether to lowercase. Defaults to True.
            lemmatize (bool): Whether to lemmatize. Defaults to True.
            remove_stopwords (bool): Whether to remove stopwords. Defaults to True.
        
        Returns:
            newtext (str): Preprocessed text.
        """
    
        newtext = re.sub('\n', ' ', text) # Remove ordinary linebreaks (there shouldn't be, so this might be redundant)

        if remove_punctuation:
            newtext = re.sub(r'[^a-zA-Z0-9\s\.]', '', str(newtext)) # Remove anything that is not a space, a letter, a dot, or a number
        
        if remove_numbers:
            newtext = re.sub(r'[0-9]', '', str(newtext)) # Remove numbers
        
        if lowercase:
            newtext = str(newtext).lower() # Lowercase
        
        if lemmatize:
            from nltk.stem import WordNetLemmatizer
            from nltk.corpus import wordnet 

            lemmatizer = WordNetLemmatizer()
            newtext = ' '.join(list(map(lambda x: lemmatizer.lemmatize(x, wordnet.NOUN), newtext.split())))
        
        if remove_stopwords:
            from nltk.corpus import stopwords
            stop_words = set(stopwords.words('english'))
            newtext = ' '.join(list(filter(lambda x: x not in stop_words, newtext.split())))

        return newtext
        

if __name__ == '__main__':
    # Test
    text = 'This is a test. 1234'
    print('Original text: ', text)
    print('Preprocessed text: ', PREPROCESS().forward(text, remove_punctuation=True, remove_numbers=True, lowercase=True, lemmatize=True, remove_stopwords=True))
