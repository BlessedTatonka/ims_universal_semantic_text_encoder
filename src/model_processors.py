from abc import abstractmethod
import numpy as np

from .utils import average_pool

class AbstractModelProcesssor:
    @abstractmethod
    def __init__(self, model, tokenizer=None):
        pass
    
    @abstractmethod
    def get_embeddings(self, sentences, device):
        pass

    def get_sentences(self, dataset_processor, i):
        sample = dataset_processor.get_sample(i)
        if 'text_2' in sample.keys():
            sentences = sample['text_1'], sample['text_2']
        else:
            sentences = sample['text_1']
        return sentences
    
'''
'''
class T5ModelProcessor(AbstractModelProcesssor):
    def __init__(self, model, tokenizer, device):
        model.to(device)
        self.__model = model
        self.__tokenizer = tokenizer
        self.device = device
    
    def get_embeddings(self, sentences):
        enc = self.__tokenizer(sentences, \
                               max_length=512, padding=True, truncation=True, return_tensors='pt').to(self.device)
        outputs = self.__model.encoder(**enc)
        embeddings = average_pool(outputs.last_hidden_state, enc['attention_mask'].to(self.device))
        return embeddings


'''
'''
class E5ModelProcessor(AbstractModelProcesssor):
    def __init__(self, model, tokenizer, device):
        model.to(device)
        self.__model = model
        self.__tokenizer = tokenizer
        self.device = device
        
    def get_embeddings(self, sentences):
        enc = self.__tokenizer(sentences, \
                               max_length=512, padding=True, truncation=True, return_tensors='pt').to(self.device)
        outputs = self.__model(**enc)
        embeddings = average_pool(outputs.last_hidden_state, enc['attention_mask'].to(self.device))
        return embeddings
    
'''
'''
class LabseModelProcessor(AbstractModelProcesssor):
    def __init__(self, model, tokenizer, device):
        model.to(device)
        self.__model = model
        self.__tokenizer = tokenizer
        self.device = device
        
    def get_embeddings(self, sentences):
        enc = self.__tokenizer(sentences, \
                           max_length=512, padding=True, truncation=True, return_tensors='pt').to(self.device)
        outputs = self.__model(**enc)
        embeddings = outputs.pooler_output
        return embeddings

'''
'''
class MinilmModelProcessor(AbstractModelProcesssor):  
    def __init__(self, model, device):
        model.to(device)
        self.__model = model
        self.device = device
        
    def get_embeddings(self, sentences, device):
        embeddings = self.__model.encode(sentences, device=self.device)
        embeddings = torch.tensor(embeddings)
        return embeddings

'''
'''
class InstructorModelProcessor(AbstractModelProcesssor):
    def __init__(self, model, device):
        model.to(device)
        self.__model = model
        self.device = device
        
    def get_embeddings(self, sentences):
        embeddings = self.__model.encode(sentences, device=self.device)
        embeddings = torch.tensor(embeddings)
        return embeddings


'''
'''
class GloveModelProcessor(AbstractModelProcesssor):
    def __init__(self, model):
        self.__model = model
        
    def get_embeddings(self, sentences):
        embeddings = [self.__model.get_mean_vector(sentence) for sentence in sentences]
        embeddings = torch.tensor(embeddings)
        return embeddings

   