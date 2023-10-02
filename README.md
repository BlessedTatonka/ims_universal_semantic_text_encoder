# ims_universal_semantic_text_encoder
Тестовое задание для проекта мастерской по виртуальным и платформенным решениям (СПб)  — Универсальный семантический энкодер текстов

-----

**Возникшие вопросы**
   1. Почему в задании указано, что необходимо выбрать *датасеты*, а в примерах *бенчмарки*? При использовании бенчмарков теряется весь смысл задания, ведь они уже содержат множество датасетов и метрик с описаниями. 
   
**План проекта**
   1. Выбор датасетов.
   
   2. Определение метрик.
   
   3. Выбор подходов для сравнения.
   
   4. Постановка гипотез.
   
   5. Проведение экспериментов.
   
   6. Выводы.
   
-----

### Датасеты:
+ Параллельный перевод для удмуртского языка (как языка малой народности) [UDM](https://huggingface.co/datasets/udmurtNLP/flores-250-rus-udm).
+ Парафразы [PAR](https://huggingface.co/datasets/merionum/ru_paraphraser). Кроме того, содержит "непарафразы", что также можно использовать для проверки качества.
+ Всегда актуальная токсичность [TOX](https://huggingface.co/datasets/FredZhang7/toxi-text-3M). Я выбрал именно этот датасет, так как он содержит 50+ языков, и это наибльшая выборка для этой задачи, которую я сумел найти. В ней слишком много английского, турецкого и арабского. 
+ Суммаризация [SUM](https://huggingface.co/datasets/embedding-data/sentence-compression). В этом датасете тексты относительно небольшие и не требуют больших вычислительных мощностей.
+ Мультиклассовая классификация [MGP](https://huggingface.co/datasets/datadrivenscience/movie-genre-prediction). Модель с описанием фильмов и жанрами.

| Датасет | Размер* | Число языков |
|----------|----------|----------|
| flores-250-rus-udm    | 250   | 2   |
| ru_paraphraser    | 7227   | 1   |
| instructor-base    | 37910   | 45   |
|sentence-compression |     2000     |     1       |
|movie-genre-prediction |     54000        |   1      |

### Метрики:
+ *UDM* - косинусная близость русских и удмуртских эмбеддингов. Универсальный энкодер должен хорошо справляться не только с широко представленными языками и успешно извлекать смыслы из всего, что ему передают.
+ *PAR* - считаем косинусную близость для пар с классом "precise paraphrases" и (1 - косинусная близость) для пар с классом "non-paraphrases". Класс "near paraphrases" игнорируем. Метрика, в какой-то степени, дополняет метрику *UDM*, показывая, что модель не просто располагает все эмбеддинги в одной точки векорного пространства.
+ *SUM* - косинусная близость эмбеддингов текста и краткого изложения. Хотим знать, насколько хорошо модель справляется с извлечением смысла из больших текстов.
+ *TOX* - произведение средней точности KNN** и энтропии точностей по классам (по языкам) $A \cdot entropy([a_1, a_2, \ldots, a_n]), a_i = acc_{c_i}$. Метрика показывает, насколько хорошо подель справляется с бинарной классификацией. Если не дополнять вычисления энтропией, то при использовании модели в дальнейшем на каком-то языке модель может показать результат, гораздо ниже ожидаемого. Текущая метрика будет сильно ниже, если модель справляется с предсказаниями на каких-то отдельных языках хуже, чем на других.
+ *MGP* - f1_score macro на предсказания KNN. Проверка на задаче мультиклассовой классификации.

### Подходы:
+ *Word2Vec* - бейзлайн и наиболее примитивный подход. Для получения эмбеддингов предложения, буду брать усреднение по эмбеддингам слов. Я взял веса от предобученной модели [*glove-twitter-25*](https://radimrehurek.com/gensim/models/word2vec.html#other-embeddings).
+ *T5-flan* [google/flan-t5-small](https://huggingface.co/google/flan-t5-small)  Использую усреднение эмбеддингов энкодера, так как в [этой статье](https://arxiv.org/abs/2108.08877) такой подход показал лучший результат для zero-shot. Модель обучалась на достаточно большом корпусе языков.
+ [*InstructOR*](https://arxiv.org/pdf/2212.09741.pdf) - ещё одна T5-based модель, которая обучалась на каком-то невероятно большом корпусе данных и различных задач специально для создания векторных представлений.
+ [*intfloat/multilingual-e5-small*](https://huggingface.co/intfloat/multilingual-e5-small) - трансформер, обученный специально для задачи выделения эмбеддингов текстов. В обучающей выборке содержалось огромное количество языков.
+ [*sentence-transformers/all-MiniLM-L6-v2*](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) - bert-based решение для задачи получения векторных представлений текста.
+ [*setu4993/LaBSE*](https://huggingface.co/setu4993/LaBSE) - ещё одно bert-based решение, но новее больше и обучено на огромном количестве языков.

| Модель | Вес | На скольки языках обучена  |
|----------|----------|----------|
| glove-twitter-25              | 110 MB   | 1   |
| google/flan-t5-small          | 308 MB   | 50+   |
| hkunlp/instructor-base        | 439 MB   | ?   |
| intfloat/multilingual-e5-small |     471 MB     |     90+       |
| all-MiniLM-L6-v2              |     91 MB        |   1      |
| setu4993/LaBSE                | **1.88 GB**| **109** |

### Гипотезы:
1. Word2Vec покажет себя хуже всего в среднем по всем метрикам, так как не предназначен для задачи создания векторных представлений текста.
2. Лучше всего себя покажет LaBSE, она содержит наибольшее число параметров и обучена на наибольшем количестве языков.
3. Модель InstructOR покажет себя лучше, чем T5-flan, так как они имеют схожую архитектуру, но она была обучена специально для задачи получения векторных представлений и испытана на широком классе задач. Но, я не смог найти, какое число языков было в обучающей выборке. У google/flan-t5-small это значение достаточно велико.

### Реализация:
В папке *src* находятся следующие файлы:
+ *data_processors.py* - содержит классы для каждого датасета. 
+ *model_processors.py* - для каждого типа модели свой класс-обертка, чтобы можно было использовать общий пайплайн. При этом можно заменять модели на более большие, но в текущих экспериментах я старался брать наименьшие.
+ *metrics.py* - файл с методами, для расчета итоговых метрик.
+ *utils.py* - дополнительные методы.

### Результаты:

Все значения огруглены до 3 знаков после запятой.

Для метрик UDM, PAR и SUM и я вычисляю общее значение UPS по формуле:

$UPS = \frac{\frac{UDM + SUM}{2} \cdot PAR}{\frac{UDM + SUM}{2} + PAR}$.

Так более репрезентативно, т.к. часть моделей, например InstructOR, показывают близкое к 1 значение на UDM и практически 0 на PAR. Их среднее арифметическое означает, что модель располагает тексты с одинаковыми смыслами близко в векторном пространстве, но при этом занимает векторное пространство равномерно, а не какую-то его малую часть. Интуиция для такой метрики основана на f_score.

| Модель | UPS | TOX | MGP | Mean |
|----------|----------|----------|----------|----------|
| glove-twitter-25              | 0.006  |  0.732   |  0.09   |  0.276   | 
| google/flan-t5-small          | 0.09   |  0.775  |  0.215   |   0.36   |
| hkunlp/instructor-base        | 0.016  |  0.795   |   **0.295**   | 0.37 |
|intfloat/multilingual-e5-small | 0.072  | **0.82**   |  0.265   |  0.386   | 
| all-MiniLM-L6-v2              | 0.16  |  0.8  |  0.272   |   **0.411**   |
| setu4993/LaBSE                | **0.176**   |  0.8  |  0.24   |  0.405    |

### Выводы:

1. Гипотеза подтвердилась, word2vec показал наихудшие результаты в каждой из метрик. Это было ожидаемо, так как для этой задачи стоит использовать doc2vec, в больших текстах агрегированные значения векторов word2vec становятся менее репрезентативны.
2. Гипотеза не подтвердилась, но LaBSE во всех метриках очень близка к лучшей (в UPS лучшая) и практически победитель по усреднённому показателю.
3. Гипотеза подтвердилась, InstructOR немного лучше, чем t5. Но он сильно хуже в метриках, связанных с косинусной близостью.

Неожиданно, что лучшей моделью в среднем по метрикам стала all-MiniLM-L6-v2, ведь она весит меньше всех остальных и её датасет для обучения состоял только из английского языка. 

------

**Несколько уточнений по-поводу датасетов:*
1. *Если данные поделены на train/test/val, я оставляю только train.*

2. *Часть данных в не parquet формате я перевёл в parquet для ускорения.*

3. *Я уменьшил размер toxi-text-3M с 2880667 до 37910 семплов. Ограничился 1000 семплов сверху и 100 снизу для каждого языка. Сверху, чтобы я успел провести эксперименты, а снизу, чтобы рассчеты среднего по языкам были более репрезентативны. Для возможности воспроизвести вычисления, файл с обновленным датасетом я добавлю в репозиторий. Кроме того, в ноутбуке **toxic_dataset_preparation.ipynb** находится полный процесс изменения датасета.* 

4. Датасет *sentence_compression* состоит из 180000 семплов, однако большие тексты требуют слишком много времени для вычисления, я ограничился 2000 семплов.


***Во всех метриках я испольхую KNN с параметрами: n_neighbors=11, weights='distance', metric='cosine'. Основательно параметры я не подбирал кроме числа соседей, в остальном - интуиция. На train/val я разбиваю с random_state=\[0;5\) и беру усреднение из 5 моделей.*

------

### Воспроизведение экспериментов 

1. pip install -r requirements.txt
2. Скачать все датасеты в папку data
3. Установить библиотеку [InstructOR](https://github.com/xlang-ai/instructor-embedding)
4. python3 run.py
