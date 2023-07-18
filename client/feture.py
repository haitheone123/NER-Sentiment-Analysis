import sys
sys.path.append('../server')

from flair.data import Sentence
from flair.nn import Classifier

def relation_tagger(text = "Flight tracking service Flightradar24's website showed an Embraer Legacy 600 jet, bearing identification codes that match a plane linked to Prigozhin in U.S. sanctions documents, descending to landing altitude near the Belarus capital Minsk" ):
    # 1. make example sentence
    sentence = Sentence(text)

    # 2. load entity tagger and predict entities
    tagger = Classifier.load('ner-fast')
    tagger.predict(sentence)

    # # check which named entities have been found in the sentence
    entities = sentence.get_labels('ner')
    # for entity in entities:
    #     print(entity)

    # 3. load relation extractor
    extractor = Classifier.load('relations')

    # predict relations
    extractor.predict(sentence)

    # check which relations have been found
    relations = sentence.get_labels('relation')
    # print(relations)
    # for relation in relations:
    #     print(relation)

    # # Use the `get_labels()` method with parameter 'relation' to iterate over all relation predictions. 
    # for label in sentence.get_labels('relation'):
    #     print(label)
        
    # return sentence.get_labels('ner')
    return relations

def named_entity_recognition(text = 'George Washington went to Washington.'):
    from flair.embeddings import TransformerWordEmbeddings
    from flair.splitter import SegtokSentenceSplitter
    
    embedding = Classifier.load('/home/thienlv/RL_team/NLP_demo/server/fintunning_step/resources/taggers/sota-ner-roberta-vi/best-model.pt')
    # load the model
    # tagger = Classifier.load('/home/thienlv/RL_team/NLP_demo/server/weight/ner_vi_best-model.pt')
    # tagger = Classifier.load('/home/thienlv/RL_team/NLP_demo/client/phobert-base/pytorch_model.bin')
    # make a sentence
    splitter = SegtokSentenceSplitter()
    sentences = splitter.split(text)
    # sentence = Sentence(text)
    # predict NER tags
    embedding.predict(sentences)
    # embedding.embed(sentences)
    
    return sentences

def sentiment(text = 'This movie is not at all bad.'):
    # load the model
    tagger = Classifier.load('sentiment')
    # make a sentence
    sentence = Sentence(text)
    # predict NER tagsThayThien@AIoT
    tagger.predict(sentence)
    
    return sentence

#test

# output = named_entity_recognition()
# print(type(output.__str__()))
# print(output.__str__())