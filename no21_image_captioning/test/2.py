from tensorflow.contrib.learn import preprocessing
import numpy as np

def test():
    text_list = ['苹果 是 什么 垃圾', '塑料瓶 是 那种 垃圾']#先用结巴分好词
    max_words_length = 10
    vocab_processor = preprocessing.VocabularyProcessor(max_document_length=max_words_length)
    x = np.array(list(vocab_processor.fit_transform(text_list)))

    print('x:\n', x)

    print('词-索引映射表：\n', vocab_processor.vocabulary_._mapping)

    print('词汇表：\n', vocab_processor.vocabulary_._reverse_mapping)


    #保存vocabulary
    vocab_processor.save('vocab.pkl')
