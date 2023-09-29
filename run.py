import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import BertTokenizerFast, BertModel
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer

from src.data_processors import *
from src.model_processors import *
from src.metrics import *

import random
from time import time

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {DEVICE}')

# Можно изменять порядок и добавлять модели с иным числом весов
def get_models_tokenizers_processors():
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
    model_processor = T5ModelProcessor(model, tokenizer, device=DEVICE)
    yield model_processor
    
    model = AutoModel.from_pretrained('intfloat/multilingual-e5-small')
    tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-small')
    model_processor = E5ModelProcessor(model, tokenizer, device=DEVICE)
    yield model_processor
    
    model = BertModel.from_pretrained("setu4993/LaBSE")
    tokenizer = BertTokenizerFast.from_pretrained("setu4993/LaBSE")
    model_processor = LabseModelProcessor(model, tokenizer, device=DEVICE)
    yield model_processor
    
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    tokenizer = None
    model_processor = MinilmModelProcessor(model, tokenizer, device=DEVICE)
    yield model_processor

    model = INSTRUCTOR('hkunlp/instructor-base')
    tokenizer = None
    model_processor = InstructorModelProcessor(model, tokenizer, device=DEVICE)
    yield model_processor
    
    model = gensim.downloader.load('glove-twitter-25')
    tokenizer = None
    model_processor = GloveModelProcessor(model, tokenizer, device=DEVICE)
    yield model_processor


def get_dataset_metric():
    dataset_processor = FloresDatasetProcessor()
    metric = calculate_UDM
    yield dataset_processor, metric, None
    
    dataset_processor = ParaphraserDatasetProcessor()
    metric = calculate_PAR
    yield dataset_processor, metric, None
    
    dataset_processor = ToxiDatasetProcessor()
    metric = calculate_TOX
    yield dataset_processor, metric, None
    
    # Ограничиваю 2000, иначе очень долго считать
    dataset_processor = SummDatasetProcessor()
    metric = calculate_SUM
    yield dataset_processor, metric, 2000
    
    dataset_processor = MGPDatasetProcessor()
    metric = calculate_MGP
    yield dataset_processor, metric, None


# Фиксирую случайность 
def seed_all(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

def main():
    seed_all()

    start_time = time()

    for model_processor in get_models_tokenizers_processors():
        for dataset_processor, metric, k in get_dataset_metric():
            if k is None:
                k = 1e9
            metric_result = metric(model_processor, dataset_processor, k)

            eval_time = time() - start_time

            with open('logs.txt', 'a') as logs:
                log = f'model:{model_processor.__class__}; metric:{metric.__name__}; \
                    result:{metric_result}; eval_time:{eval_time}.\n'
                logs.write(log)
                
                
if __name__ == "__main__":
    main()