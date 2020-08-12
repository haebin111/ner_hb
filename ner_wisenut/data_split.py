"""
데이터 병합 + 분리
"""
import argparse
import random
import dataEdit
from sklearn.model_selection import train_test_split


def data_merge(first_file, second_file, merge_file):
    """
    :param first_file: 첫 번째 병합할 파일의 경로
    :param second_file: 두 번째 병합할 파일의 경로
    :param merge_file: 최종 출력 데이터 파일의 경로
    :return: None
    """

    with open(first_file, 'r', encoding='utf-8-sig') as ft, open(second_file, 'r', encoding='utf-8-sig') as sf:
        flines = ft.readlines()
        slines = sf.readlines()

    result = flines + slines
    random.shuffle(result)

    with open(merge_file, 'w', encoding='utf-8-sig') as f:

        f.writelines(result)
    return None


def main(args):
    random.seed(2020)

    data_merge(args.first_file, args.second_file, args.merge_file)
    data = dataEdit.get_lines(args.merge_file) # 한 줄 씩 읽어서 리스트에 저장한 값을 반환(태깅 된 상태)

    train, test = train_test_split(data, train_size=0.9, test_size=0.1, random_state=2020, shuffle=True)
    dataEdit.data_edit(train, args.train)
    dataEdit.data_edit(test, args.test)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--first_file', type=str, default='./originData/EXOBRAIN_NE_CORPUS_10000.txt')
    parser.add_argument('--second_file', type=str, default='./originData/wisenut_final.txt')

    parser.add_argument('--merge_file', type=str, default='./originData/mergeData.txt')

    parser.add_argument('--train', type=str, default='./editData/train.txt')
    parser.add_argument('--test', type=str, default='./editData/test.txt')


    args = parser.parse_args()
    main(args)
