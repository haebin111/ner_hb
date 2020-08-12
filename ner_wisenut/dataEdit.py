import json
import argparse
import re
from eunjeon import Mecab


# 데이터의 형식을 바꾸는 코드
# 문장 중 <단어:태그> 형태의 데이터를
# 단어  태그
# 단어  태그
# 형태의 데이터로 바꾼다.

re_word = re.compile('<(.+?):[A-Z]{2}>')
mecab = Mecab()


def data_edit(lines, outfile_path):
    with open(outfile_path, 'w', encoding='utf-8-sig') as ef:
        for line in lines:
            """
            한 줄이 통째로 빈 경우
            """
            if line == '\n':
                continue
            morphs, tags = make_morph_tag(line)

            for morph, tag in zip(morphs, tags):
                ef.write(morph + '\t' + tag + '\n')
            ef.write('\n')  # 문장끼리 구분하기 위한 공백 추가
    return


def make_morph_tag(text):

    """
    ###### example ######

    input  : text = '<두산:OG> <홍성흔:PS>(31)의 입담은 여전했다 .'

    output : (['두산', '홍성흔', '(', '31', ')', '의', '입담', '은', '여전', '했', '다', '.'],
              ['OG', 'PS', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-'])
    """

    entity_compile = re.compile('[<](.*?)[:]')
    tag_compile = re.compile('[:]([A-Z]{2})[>]')

    raw_text = re.sub(':[A-Z]{2}', '', text).replace('\ufeff', '').replace('<', ' ').replace('>', ' ')

    entity_words = list(
        map(lambda x: x.strip().replace(' ', ''), re.findall(entity_compile, text)))  # 개체명만 모아둔 리스트(띄어쓰기 전부 없앤다)
    entity_tags = re.findall(tag_compile, text)  # 각 개체명에 할당된 태그

    morph_pos_text = mecab.pos(raw_text)  # [(morph, pos), (morph, pos), ...]

    # 위의 리스트에서 morph만 가져온다
    morphs = [morph for morph, pos in morph_pos_text]
    tags = ['-'] * len(morphs)  # 형태소의 모든 tag 값을 '-' 로 초기화 시켜놓는다

    idx = 0  # entity_tags 의 인덱스를 가리킨다
    start = 0  # morphs에서 entity에 해당되는 형태소가 시작하는 위치를 가리킨다
    end = 0  # morphs에서 entity에 해당되는 형태소가 끝나는 위치를 가리킨다
    i = 0
    morph_len = 0  # 개체명이라고 생각되는 형태소의 길이를 저장한다. 나중에 entity_words에 있는 진짜 개체명 단어와 길이를 비교하기 위해 사용한다
    entity_len = 0  # entity_words에 있는 진짜 개체명 단어의 길이를 저장한다

    for i in range(len(morphs) - 1):

        morph = morphs[i]  # morphs를 차례대로 하나씩 꺼낸다

        try:
            entity = entity_words[idx]  # 더이상 문장에 개체명에 해당되는 형태소가 없다면 for문을 종료한다
        except:
            break

        if morph in entity and entity.find(morph) == 0:  # 형태소가 개체명 단어의 처음 부분일 때
            start = i
            morph_len = 0
            entity_len = 0

            morph_len += len(morph)
            entity_len = len(entity)

            if morph_len == entity_len:
                tags[i] = entity_tags[idx] + '_B'

                morph_len = 0
                entity_len = 0
                idx += 1

        elif morph in entity and entity.find(morph) != 0:  # 형태소가 개체명 단어의 일부이지만 처음 부분은 아닐 때

            if morph_len == 0:
                continue

            morph_len += len(morph)
            entity_len = len(entity)

            if morph_len < entity_len:
                continue

            elif morph_len == entity_len:
                end = i
                if ''.join(morphs[start:end + 1]) == entity:

                    tags[start] = entity_tags[idx] + '_B'

                    for index in range(start + 1, end + 1):
                        tags[index] = entity_tags[idx] + '_I'

                    morph_len = 0
                    entity_len = 0
                    idx += 1

        elif morph not in entity:  # 형태소가 개체명이 아닐 때
            morph_len += len(morph)
            entity_len = len(entity)

            if morph_len >= entity_len:
                morph_len = 0
                entity_len = 0

    return morphs, tags


def get_lines(read_file):
    with open(read_file, 'r', encoding='utf-8-sig') as of:
        lines = of.readlines()
    return lines


def make_morphs(text):
    morphs = []
    for t in text:
        if t == '\n':
            continue
        morph, tags = make_morph_tag(t)
        morphs.append(morph)
    return morphs


def main(args):
    lines = get_lines(args.file)
    data_edit(lines, args.result_file)
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--file', type=str, default='./originData/mergeData.txt')
    parser.add_argument('--result_file', type=str, default='./editData/merge.txt')

    args = parser.parse_args()
    main(args)
