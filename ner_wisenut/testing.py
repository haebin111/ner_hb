import bilstm_crf
import argparse
import dataEdit
import torch
import utils
from vocab import Vocab


def print_line(line, model_path='./model/model.pth', sent_vocab_path='./vocab/sent_vocab.json', tag_vocab_path='./vocab/tag_vocab.json'):
    """
    :param line: text line
    :param model_path: model path
    :param sent_vocab_path: sentence vocab path
    :param tag_vocab_path: tag vocab path
    :return: print text and tag
    """
    #device = torch.device('cuda:0')
    sent_vocab = Vocab.load(sent_vocab_path)
    tag_vocab = Vocab.load(tag_vocab_path)
    device = torch.device("cpu")
    model = bilstm_crf.BiLSTMCRF.load(model_path, device)

    morph_line, tags = dataEdit.make_morph_tag(line)

    sentences = utils.words2indices([morph_line], sent_vocab)
    sentences, sent_lengths = utils.pad(sentences, sent_vocab[sent_vocab.PAD], device)
    predicted_tags = model.predict(sentences, sent_lengths)
    tags = utils.indices2words(predicted_tags, tag_vocab)

    print(line)
    OG = []; PS = []; DT = []; LC = []; TI = []
    OG_i = PS_i = DT_i = LC_i = TI_i = 0
    before_tag = None
    for word, tag in zip(morph_line, tags[0]):
        if tag != '-' and tag != '<START>' and tag != '<END>' and tag != '<PAD>':
            if tag.split('_')[1] == 'I' and before_tag is not None and before_tag.split('_')[1] == 'B':
                if tag.split('_')[0] == 'OG':
                    OG[OG_i-1] = OG[OG_i-1] + word
                elif tag.split('_')[0] == 'PS':
                    PS[PS_i-1] = PS[PS_i-1] + word
                elif tag.split('_')[0] == 'DT':
                    DT[DT_i-1] = DT[DT_i-1] + word
                elif tag.split('_')[0] == 'LC':
                    LC[LC_i-1] = LC[LC_i-1] + word
                elif tag.split('_')[0] == 'TI':
                    TI[TI_i-1] = TI[TI_i-1] + word
            elif tag.split('_')[1] == 'B':
                if tag.split('_')[0] == 'OG':
                    OG.append(word)
                    OG_i += 1
                elif tag.split('_')[0] == 'PS':
                    PS.append(word)
                    PS_i += 1
                elif tag.split('_')[0] == 'DT':
                    DT.append(word)
                    DT_i += 1
                elif tag.split('_')[0] == 'LC':
                    LC.append(word)
                    LC_i += 1
                elif tag.split('_')[0] == 'TI':
                    TI[TI_i] = word
                    TI_i += 1
        before_tag = tag

    print('OG: ', end=''); print(OG)
    print('PS: ', end=''); print(PS)
    print('DT: ', end=''); print(DT)
    print('LC: ', end=''); print(LC)
    print('TI: ', end=''); print(TI)

    return


def main():
    lines = dataEdit.get_lines(args.sample_data)
    lines = list(map(lambda s: s.strip(), lines))
    for line in lines:
        if line == '':
            continue
        print_line(line, args.MODEL, args.SENT_VOCAB, args.TAG_VOCAB)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--MODEL', type=str, default='./model/model.pth')
    parser.add_argument('--sample_data', type=str, default='./originData/sample_data.txt')
    parser.add_argument('--SENT_VOCAB', type=str, default='./vocab/sent_vocab.json')
    parser.add_argument('--TAG_VOCAB', type=str, default='./vocab/tag_vocab.json')
    args = parser.parse_args()

    main()
