from tqdm import tqdm
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import entropy as ss_entropy


from src.data_processors import AbstractDatasetProcessor
from src.model_processors import AbstractModelProcesssor
from .utils import get_cosine_sim


MAX_VALUE = int(1e9)

'''
'''
def calculate_UDM(model_processor: AbstractModelProcesssor, dataset_processor: AbstractDatasetProcessor, k: int=MAX_VALUE):
    results = []
    
    for i in tqdm(range(min(len(dataset_processor), k))):
        sentences = model_processor.get_sentences(dataset_processor, i)
        embeddings = model_processor.get_embeddings(sentences)
        cosine_sim = get_cosine_sim(embeddings)
        results.append(cosine_sim)
        
    return np.mean(results)


'''
'''
def calculate_PAR(model_processor: AbstractModelProcesssor, dataset_processor: AbstractDatasetProcessor, k: int=MAX_VALUE):
    results = []
    
    for i in tqdm(range(min(len(dataset_processor), k))):
        sentences = model_processor.get_sentences(dataset_processor, i)
        label = dataset_processor.get_sample(i)['class']
        if label != 0:
            embeddings = model_processor.get_embeddings(sentences)
            cosine_sim = get_cosine_sim(embeddings) if label == 1 else 1 - get_cosine_sim(embeddings)
            results.append(cosine_sim)
        
    return np.mean(results)


'''
'''
def calculate_SUM(model_processor: AbstractModelProcesssor, dataset_processor: AbstractDatasetProcessor, k: int=MAX_VALUE):
    results = []
    
    for i in tqdm(range(min(len(dataset_processor), k))):
        sentences = model_processor.get_sentences(dataset_processor, i)
        embeddings = model_processor.get_embeddings(sentences)
        cosine_sim = get_cosine_sim(embeddings)
        results.append(cosine_sim)
        
    return np.mean(results)


'''
'''
def calculate_TOX(model_processor: AbstractModelProcesssor, dataset_processor: AbstractDatasetProcessor, k: int=MAX_VALUE):
    results = []
    labels = np.array(dataset_processor.get_labels())
    
    for i in tqdm(range(min(len(dataset_processor), k))):
        sentences = model_processor.get_sentences(dataset_processor, i)
        embeddings = model_processor.get_embeddings([sentences])
        results.append(embeddings.flatten().cpu().detach().numpy())
        
    X = np.array(results)
    acc_scores_array = []
    
    categories = np.array(dataset_processor.get_categories())
    cat_names = np.unique(categories)
    scores_over_categories = {}
    for cat in cat_names:
        scores_over_categories[cat] = []
    
    for random_state in range(5):
        idx_train, idx_val = train_test_split(np.arange(len(X)), test_size=0.2, random_state=42)
        X_train, X_val = X[idx_train], X[idx_val]
        y_train, y_val = labels[:k][idx_train], labels[:k][idx_val]
        cat_val = categories[idx_val]
                
        clf = KNeighborsClassifier(n_neighbors=11, weights='distance', metric='cosine')
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        
        acc_scores_array.append(accuracy_score(y_val, y_pred))
        
        # prediction over category
        for cat in cat_names:
            cat_idx = np.where(cat_val == cat)[0]
            if len(cat_idx) > 0:
                cat_acc = accuracy_score(y_val[cat_idx], y_pred[cat_idx])
                scores_over_categories[cat].append(cat_acc)
                
    acc_over_categories = []
    for cat in cat_names:
        if len(scores_over_categories[cat]) > 0:
            acc_over_categories.append(np.mean(scores_over_categories[cat]))
        
    entropy = ss_entropy(acc_over_categories, base=len(acc_over_categories))
        
    return np.mean(acc_scores_array) * entropy


'''
'''
def calculate_MGP(model_processor: AbstractModelProcesssor, dataset_processor: AbstractDatasetProcessor, k: int=MAX_VALUE):
    
    results = []
    labels = np.array(dataset_processor.get_labels())
    
    print(k)
    
    for i in tqdm(range(min(len(dataset_processor), k))):
        sentences = model_processor.get_sentences(dataset_processor, i)
        embeddings = model_processor.get_embeddings([sentences])
        results.append(embeddings.flatten().cpu().detach().numpy())
        
        
    X = np.array(results)
    f1_scores_array = []
    for random_state in range(5):
        X_train, X_val, y_train, y_val = train_test_split(X, labels[:k], test_size=0.2, random_state=42)

        clf = KNeighborsClassifier(n_neighbors=11, weights='distance', metric='cosine')
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        
        f1_scores_array.append(f1_score(y_val, y_pred, average='macro'))
        
    return np.mean(f1_scores_array)
