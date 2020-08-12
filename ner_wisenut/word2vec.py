from gensim.models import word2vec
from eunjeon import Mecab
import re

'''
size = 워드 벡터의 특징 값. 즉, 임베딩 된 벡터의 차원.
window = 컨텍스트 윈도우 크기
min_count = 단어 최소 빈도 수 제한 (빈도가 적은 단어들은 학습하지 않는다.)
workers = 학습을 위한 프로세스 수
sg = 0은 CBOW, 1은 Skip-gram.
'''


def make_raw_sentence(sentence):
    raw_sentence = re.sub(':[A-Z]{2}', '', sentence).replace('\ufeff', '').replace('<', ' ').replace('>', ' ')
    return raw_sentence


def get_raw_sentence(path):
    mecab = Mecab()

    with open(path, 'r', encoding='utf-8-sig') as f:
        sentences = f.readlines()

    sentences = list(map(make_raw_sentence, sentences))
    sentences = [mecab.morphs(sentence.strip()) for sentence in sentences]
    return sentences


# 모델 만들기, 저장
def make_w2v_model(sentences, model_save_path):
    word_model = word2vec.Word2Vec(sentences, size=100, window=5, min_count=5, workers=4, sg=1)
    word_model.save(model_save_path)

    print('model save success!')


def main():
    data_path = 'originData/mergedata.txt'
    model_save_path = 'model/word2vec/merge_w2v.model'
    sentences = get_raw_sentence(data_path)
    make_w2v_model(sentences, model_save_path)


if __name__ == '__main__':
    main()
