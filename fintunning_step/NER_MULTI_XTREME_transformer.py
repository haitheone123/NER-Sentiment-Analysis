import flair.datasets
import torch
import sys
import os
from flair.datasets import CONLL_03
from flair.embeddings import WordEmbeddings, FlairEmbeddings, StackedEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.data import Corpus, MultiCorpus
from flair.datasets import ColumnCorpus


flair.device = "cuda:1"

# 1. get the corpus

## NER_MULTI_XTREME : Vietnamese
corpus_xtreme = flair.datasets.NER_MULTI_XTREME(languages="vi")

## PhoNer_Covid19
columns = {0: 'text', 1: 'ner' }
data_folder = '/home/thienlv/RL_team/NLP_demo/PhoNER_COVID19/data'
corpus_pho_word: Corpus = ColumnCorpus(data_folder, columns,
                              train_file='word/train_word.conll',
                              test_file='word/test_word.conll',
                              dev_file='word/dev_word.conll')

corpus_pho_syllable: Corpus = ColumnCorpus(data_folder, columns,
                              train_file='syllable/train_syllable.conll',
                              test_file='syllable/test_syllable.conll',
                              dev_file='syllable/dev_syllable.conll')


## NER_MULTI_WIKIANN : Vietnamese
corpus_wikiann = flair.datasets.NER_MULTI_WIKIANN(languages="vi")

multi_corpus = MultiCorpus([corpus_xtreme, corpus_pho_word, corpus_pho_syllable, corpus_wikiann])

# 2. what label do we want to predict?
label_type = 'ner'

# 3. make the label dictionary from the corpus
label_dict = multi_corpus.make_label_dictionary(label_type=label_type)
print(label_dict)

# 4. initialize embedding stack with Flair and GloVe
from flair.embeddings import TransformerWordEmbeddings

embeddings = TransformerWordEmbeddings(
    model='roberta-base',
    layers="-1",
    subtoken_pooling="first",
    fine_tune=True,
    use_context=True,
    allow_long_sentences=True,
)

# embeddings.tokenizer = PhobertTokenizer
# 5. initialize bare-bones sequence tagger (no CRF, no RNN, no reprojection)
from flair.models import SequenceTagger

tagger = SequenceTagger(
    hidden_size=256,
    embeddings=embeddings,
    tag_dictionary=label_dict,
    tag_type='ner',
    use_crf=False,
    use_rnn=False,
    reproject_embeddings=False,
)
# 6. initialize trainer with AdamW optimizer
from flair.trainers import ModelTrainer

trainer = ModelTrainer(tagger, multi_corpus)

# 7. run training with XLM parameters (20 epochs, small LR)
from torch.optim.lr_scheduler import OneCycleLR

# 7. start training
trainer.train('resources/taggers/sota-ner-roberta-vi',
              learning_rate=0.000012,
              min_learning_rate=5.0e-9,
              mini_batch_size=2,
              mini_batch_chunk_size=8,
              max_epochs=100,
              # scheduler=OneCycleLR,
              optimizer=torch.optim.AdamW,
              embeddings_storage_mode='none',
              weight_decay=0.,)