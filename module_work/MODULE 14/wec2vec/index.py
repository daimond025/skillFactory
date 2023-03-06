import re  # For preprocessing
import pandas as pd  # For data handling
from time import time
from collections import defaultdict

import spacy
nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])

import logging  # Setting up the loggings to monitor gensim
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)


# df = pd.read_csv('simpsons_dataset.csv')
# df = df.dropna().reset_index(drop=True)
#
#
# def cleaning(doc):
#     txt = [token.lemma_ for token in doc if not token.is_stop]
#     if len(txt) > 2:
#         return ' '.join(txt)
#
# brief_cleaning = (re.sub("[^A-Za-z']+", ' ', str(row)).lower() for row in df['spoken_words'])
#
# txt = [cleaning(doc) for doc in nlp.pipe(brief_cleaning, batch_size=5000)]
#
#
# df_clean = pd.DataFrame({'clean': txt})
# df_clean = df_clean.dropna().drop_duplicates()
# df_clean.to_csv( 'txt.csv',   index = False)

from gensim.models.phrases import Phrases, Phraser
df_clean = pd.read_csv('txt.csv')

sent = [row.split() for row in df_clean['clean']]

phrases = Phrases(sent, min_count=30, progress_per=10000)




bigram = Phraser(phrases)
sentences = bigram[sent]
print(sentences)
exit()

