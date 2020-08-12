import bilstm_crf
import argparse
import dataEdit
import torch
import utils
import random
from vocab import Vocab
import re


def compare_sentence(text, sentence, tags):
    re_text =[]
    for t in text:
        if t == '\n':
            continue
        t = t.replace('\n', '')
        re_text.append(t)

    text_temp = []
    for t in re_text:
        temp = t.replace(' ', '')
        text_temp.append(temp)

    sent = []
    tag_list = []
    for t in text_temp:
        for s, tag in zip(sentence, tags):
            if t == s:
                sent.append(s)
                tag_list.append(tag)

    for i in range(len(re_text)):
        length = len(re_text[i])
        for j in range(length):
            if re_text[i][j] == " ":
                text = sent[i][:j] + " " + sent[i][j:]
                sent[i] = text
    return sent, tag_list


def make_sentence(lines):
    sentences, tags = [], []
    sent = ['<START>']
    for line in lines:
        line = line.replace('\ufeff', '')
        if line == '\n':
            if len(sent) > 1:
                sentences.append(sent + ['<END>'])
            sent = ['<START>']
        else:
            line = line.split('\t')
            sent.append(line[0].strip())
    return sentences


def batch_iter(data, batch_size=32, shuffle=True):
    """ Yield batch of (sent, tag), by the reversed order of source length.
    Args:
        data: list of tuples, each tuple contains a sentence and corresponding tag.
        batch_size: batch size
        shuffle: bool value, whether to random shuffle the data
    """
    data_size = len(data)
    indices = list(range(data_size))
    if shuffle:
        random.shuffle(indices)
    batch_num = (data_size + batch_size - 1) // batch_size
    for i in range(batch_num):
        batch = [data[idx] for idx in indices[i * batch_size: (i + 1) * batch_size]]
        batch = sorted(batch, key=lambda x: len(x), reverse=True)
        sentences = [x for x in batch]
        yield sentences


def main(args):
    device = torch.device('cpu')
    model = bilstm_crf.BiLSTMCRF.load(args.MODEL, device)

    text = dataEdit.get_lines(args.sample_data)
    morph_lines = dataEdit.make_morphs(text)

    lines = sorted(morph_lines, key=lambda x: len(x), reverse=True)

    sent_vocab = Vocab.load(args.SENT_VOCAB)
    tag_vocab = Vocab.load(args.TAG_VOCAB)
    sentences = utils.words2indices(lines, sent_vocab)

    for sentences in batch_iter(sentences, 64, shuffle=False):
        sentences, sent_lengths = utils.pad(sentences, sent_vocab[sent_vocab.PAD], device)
        predicted_tags = model.predict(sentences, sent_lengths)

    tags = utils.indices2words(predicted_tags, tag_vocab)

    sent = []
    for line in range(len(sentences)):
        word = []
        for i in range(len(sentences[line])):
            w = sent_vocab.id2word(sentences[line][i])
            if w == '<PAD>':
                continue
            if w == '<UNK>':
                w = lines[line][i]
            word.append(w)
        sent.append(word)

    # print(sent)
    answer = []
    for word in sent:
        result = ""
        for w in word:
            result += w
        answer.append(result)
    answer, tag_list = compare_sentence(text, answer, tags)

    for i in range(len(answer)):
        temp = answer[i]

        for j in range(len(morph_lines[i])):
            if tag_list[i][j] == '-':
                # temp += answer[i][:answer[i].find(morph_lines[i][j])] + answer[i][answer[i].find(morph_lines[i][j]):]
                continue
            else:
                temp = (temp[:temp.find(morph_lines[i][j])] + '<' + tag_list[i][j].split('_')[0] + '>' + temp[temp.find(morph_lines[i][j]):])
        print(temp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--MODEL', type=str, default='./model/model.pth')
    parser.add_argument('--sample_data', type=str, default='./originData/sample_data.txt')
    parser.add_argument('--SENT_VOCAB', type=str, default='./vocab/sent_vocab.json')
    parser.add_argument('--TAG_VOCAB', type=str, default='./vocab/tag_vocab.json')
    args = parser.parse_args()

    main(args)