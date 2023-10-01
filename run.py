import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import BertTokenizerFast, BertModel
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
from InstructorEmbedding import INSTRUCTOR
import gensim.downloader
import sys
import random
from datetime import datetime

from src.data_processors import *
from src.model_processors import *
from src.metrics import *

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {DEVICE}')

# Можно изменять порядок и добавлять модели с иным числом весов
def get_models_tokenizers_processors(names):
    if 't5' in names or 'all_models' in names:
        model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")
        tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
        model_processor = T5ModelProcessor(model, tokenizer, device=DEVICE)
        yield model_processor

    if 'e5' in names or 'all_models' in names:
        model = AutoModel.from_pretrained('intfloat/multilingual-e5-small')
        tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-small')
        model_processor = E5ModelProcessor(model, tokenizer, device=DEVICE)
        yield model_processor
    
    if 'labse' in names or 'all_models' in names:
        model = BertModel.from_pretrained("setu4993/LaBSE")
        tokenizer = BertTokenizerFast.from_pretrained("setu4993/LaBSE")
        model_processor = LabseModelProcessor(model, tokenizer, device=DEVICE)
        yield model_processor
    
    if 'sent' in names or 'all_models' in names:
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        tokenizer = None
        model_processor = MinilmModelProcessor(model, tokenizer, device=DEVICE)
        yield model_processor

    if 'instr' in names or 'all_models' in names:
        model = INSTRUCTOR('hkunlp/instructor-base')
        tokenizer = None
        model_processor = InstructorModelProcessor(model, tokenizer, device=DEVICE)
        yield model_processor
    
    if 'w2v' in names or 'all_models' in names:
        model = gensim.downloader.load('glove-twitter-25')
        tokenizer = None
        model_processor = GloveModelProcessor(model, tokenizer)
        yield model_processor


def get_dataset_metric(names):
    k = None
    if 'test_mode' in names:
        k = 100
        
    if 'UDM' in names or 'all_metrics' in names:
        dataset_processor = FloresDatasetProcessor()
        metric = calculate_UDM
        yield dataset_processor, metric, k

    if 'PAR' in names or 'all_metrics' in names:
        dataset_processor = ParaphraserDatasetProcessor()
        metric = calculate_PAR
        yield dataset_processor, metric, k

    if 'TOX' in names or 'all_metrics' in names:
        dataset_processor = ToxiDatasetProcessor()
        metric = calculate_TOX
        yield dataset_processor, metric, k

    if 'SUM' in names or 'all_metrics' in names:
        # Ограничиваю 2000, иначе очень долго считать
        dataset_processor = SummDatasetProcessor()
        metric = calculate_SUM
        yield dataset_processor, metric, 2000 # !!!

    if 'MGP' in names or 'all_metrics' in names:
        dataset_processor = MGPDatasetProcessor()
        metric = calculate_MGP
        yield dataset_processor, metric, k


# Фиксирую случайность 
def seed_all(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

def main():
    seed_all()

    for model_processor in get_models_tokenizers_processors(str(sys.argv)):
        for dataset_processor, metric, k in get_dataset_metric(str(sys.argv)):
            if k is None:
                k = int(1e9)
                        
            start_time = datetime.now()

#             try:
            metric_result = metric(model_processor, dataset_processor, k)
#             except:
#                 metric_result = 'ERROR'
            end_time = datetime.now()

            with open('logs.txt', 'a') as logs:
                log = f'model:{model_processor.__class__}; metric:{metric.__name__}; \
                    result:{metric_result}; start:{start_time}; end:{end_time}.\n'
                logs.write(log)
                
                
if __name__ == "__main__":
    main()
