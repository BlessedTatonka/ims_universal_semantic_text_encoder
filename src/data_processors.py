from abc import abstractmethod
from datasets import load_dataset

class AbstractDatasetProcessor:
    @abstractmethod
    def get_sample(i: int):
        pass
    
'''
flores-250-rus-udm
'''
class FloresDatasetProcessor(AbstractDatasetProcessor):
    def __init__(self):
        dataset = load_dataset("data/flores-250-rus-udm/")
        self.__dataset = dataset['sentences']
        self.__len = len(self.__dataset)
        
    def __len__(self):
        return self.__len
    
    def get_sample(self, i: int):
        if i < 0 or i >= len(self):
            return None
        
        sample = {
            'text_1': self.__dataset['rus'][i],
            'text_2': self.__dataset['udm'][i]
        }
        
        return sample
        
'''
ru_paraphraser
'''  
class ParaphraserDatasetProcessor(AbstractDatasetProcessor):
    def __init__(self):
        dataset = load_dataset("data/ru_paraphraser/")
        self.__dataset = dataset['train']
        self.__len = len(self.__dataset)
        
    def __len__(self):
        return self.__len
    
    def get_sample(self, i: int):
        if i < 0 or i >= len(self):
            return None
        
        sample = {
            'text_1': self.__dataset['text_1'][i],
            'text_2': self.__dataset['text_2'][i],
            'class': self.__dataset['class'][i]
        }
        
        return sample
  

'''
toxi-text-3M
'''
class ToxiDatasetProcessor(AbstractDatasetProcessor):
    def __init__(self):
        dataset = load_dataset('data/toxi-text-3M/train_balanced/')
        self.__dataset = dataset['train']
        self.__len = len(self.__dataset)
        
    def __len__(self):
        return self.__len
    
    def get_sample(self, i: int):
        if i < 0 or i >= len(self):
            return None
        
        sample = {
            'text_1': self.__dataset['text'][i],
            'is_toxic': self.__dataset['is_toxic'][i],
            'lang': self.__dataset['lang'][i]
        }
        
        return sample
    
    def get_labels(self):
        return self.__dataset['is_toxic']
    
    def get_categories(self):
        return self.__dataset['lang']
    
    
    
class SummDatasetProcessor(AbstractDatasetProcessor):
    def __init__(self):
        dataset = load_dataset("data/sentence-compression/")
        self.__dataset = dataset['train']
        self.__len = len(self.__dataset)
        
    def __len__(self):
        return self.__len
    
    def get_sample(self, i: int):
        if i < 0 or i >= len(self):
            return None
        
        sample = {
            'text_1': self.__dataset['set'][i][0],
            'text_2': self.__dataset['set'][i][1],
        }
        
        return sample
   
'''
'''
class MGPDatasetProcessor(AbstractDatasetProcessor):
    def __init__(self):
        dataset = load_dataset("data/movie-genre-prediction/data")
        self.__dataset = dataset['train']
        self.__len = len(self.__dataset)
        
    def __len__(self):
        return self.__len
    
    def get_sample(self, i: int):
        if i < 0 or i >= len(self):
            return None
        
        sample = {
            'text_1': self.__dataset['synopsis'][i],
            'label': self.__dataset['genre'][i] 
        }
        
        return sample
    
    def get_labels(self):
        return self.__dataset['genre']