import re



# ------------------- cleantxt --------------------
def cleantxt(text, **kwargs):
    
    newtext = re.sub('\n', ' ', text) # Remove ordinary linebreaks (there shouldn't be, so this might be redundant)

    if kwargs.get('remove_punctuation', True):
        newtext = re.sub(r'[^a-zA-Z0-9\s\.]', ' ', str(newtext)) # Remove anything that is not a space, a letter, a dot, or a number
    
    if kwargs.get('remove_numbers', True):
        newtext = re.sub(r'[0-9]', ' ', str(newtext)) # Remove numbers
    
    if kwargs.get('lowercase', True):
        newtext = str(newtext).lower() # Lowercase
    
    if kwargs.get('lemmatize', True):
        from nltk.stem import WordNetLemmatizer
        from nltk.corpus import wordnet 

        lemmatizer = WordNetLemmatizer()
        newtext = ' '.join(list(map(lambda x: lemmatizer.lemmatize(x, wordnet.NOUN), newtext.split())))
    
    if kwargs.get('remove_stopwords', True):
        from nltk.corpus import stopwords
        stop_words = set(stopwords.words('english'))
        newtext = ' '.join(list(filter(lambda x: x not in stop_words, newtext.split())))

    return newtext


class PREPROCESS():
    def __init__(self):
        pass

    def forward(self,
                text,
                **kwargs):
    
        return cleantxt(text, **kwargs)
        

    
        
    


if __name__ == '__main__':
    text = 'Hello! yes is a stopword. Needing to test \n for lemmatization ALSO!'
    print(cleantxt(text, remove_stopwords=True, lemmatize=True, lowercase=True, remove_punctuation=True))
