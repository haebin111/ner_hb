"""
train data: us | test data: us
epoch: 125
tp = 1195, fp = 342, fn = 1051
Precision: 0.777489, Recall: 0.532057, F1 score: 0.631774

epoch: 50
tp = 1266, fp = 327, fn = 980
Precision: 0.794727, Recall: 0.563669, F1 score: 0.659547

train data: exo | test data: us
epoch: 50
tp = 51, fp = 2242, fn = 2195
Precision: 0.022242, Recall: 0.022707, F1 score: 0.022472
"""

from vocab import Vocab
import time
import torch
import torch.nn as nn
import bilstm_crf
import utils
import random
import gensim
import argparse


def train(args):
    """ Training BiLSTMCRF model
    Args:
        args: dict that contains options in command
    """
    sent_vocab = Vocab.load(args.SENT_VOCAB)
    tag_vocab = Vocab.load(args.TAG_VOCAB)
    train_data, dev_data = utils.generate_train_dev_dataset(args.TRAIN, sent_vocab, tag_vocab)
    print('num of training examples: %d' % (len(train_data)))
    print('num of development examples: %d' % (len(dev_data)))

    max_epoch = int(args.max_epoch)
    log_every = int(args.log_every)
    validation_every = int(args.validation_every)
    model_save_path = args.model_save_path
    optimizer_save_path = args.optimizer_save_path
    min_dev_loss = float('inf')
    #device = torch.device('cuda' if args.cuda else 'cpu')
    # print('cuda is available: ', torch.cuda.is_available())
    # print('cuda device count: ', torch.cuda.device_count())
    # print('cuda device name: ', torch.cuda.get_device_name(0))
    device = torch.device("cpu")
    patience, decay_num = 0, 0

    # 현재 미사용 word2vec
    ko_model = gensim.models.Word2Vec.load(args.word2vec_path)
    word2vec_matrix = ko_model.wv.vectors

    model = bilstm_crf.BiLSTMCRF(sent_vocab, tag_vocab, word2vec_matrix, float(args.dropout_rate), int(args.embed_size),
                                 int(args.hidden_size)).to(device)
    for name, param in model.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, 0, 0.01)
        else:
            nn.init.constant_(param.data, 0)

    # optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    optimizer = torch.optim.RMSprop(model.parameters(), lr=float(args.lr))
    train_iter = 0  # train iter num
    record_loss_sum, record_tgt_word_sum, record_batch_size = 0, 0, 0  # sum in one training log
    cum_loss_sum, cum_tgt_word_sum, cum_batch_size = 0, 0, 0  # sum in one validation log
    record_start, cum_start = time.time(), time.time()

    print('start training...')
    for epoch in range(max_epoch):
        n_correct, n_total = 0, 0
        for sentences, tags in utils.batch_iter(train_data, batch_size=int(args.batch_size)):
            train_iter += 1
            current_batch_size = len(sentences)
            sentences, sent_lengths = utils.pad(sentences, sent_vocab[sent_vocab.PAD], device)
            tags, _ = utils.pad(tags, tag_vocab[tag_vocab.PAD], device)

            # back propagation
            optimizer.zero_grad()
            batch_loss = model(sentences, tags, sent_lengths)  # shape: (b,)
            loss = batch_loss.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.clip_max_norm))
            optimizer.step()

            record_loss_sum += batch_loss.sum().item()
            record_batch_size += current_batch_size
            record_tgt_word_sum += sum(sent_lengths)

            cum_loss_sum += batch_loss.sum().item()
            cum_batch_size += current_batch_size
            cum_tgt_word_sum += sum(sent_lengths)

            if train_iter % log_every == 0:
                print('log: epoch %d, iter %d, %.1f words/sec, avg_loss %f, time %.1f sec' %
                      (epoch + 1, train_iter, record_tgt_word_sum / (time.time() - record_start),
                       record_loss_sum / record_batch_size, time.time() - record_start))
                record_loss_sum, record_batch_size, record_tgt_word_sum = 0, 0, 0
                record_start = time.time()

            if train_iter % validation_every == 0:
                print('dev: epoch %d, iter %d, %.1f words/sec, avg_loss %f, time %.1f sec' %
                      (epoch + 1, train_iter, cum_tgt_word_sum / (time.time() - cum_start),
                       cum_loss_sum / cum_batch_size, time.time() - cum_start))
                cum_loss_sum, cum_batch_size, cum_tgt_word_sum = 0, 0, 0

                dev_loss = cal_dev_loss(model, dev_data, 64, sent_vocab, tag_vocab, device)
                cal_f1_score(model, dev_data, 64, sent_vocab, tag_vocab, device)
                if dev_loss < min_dev_loss * float(args.patience_threshold):
                    min_dev_loss = dev_loss
                    model.save(model_save_path)
                    torch.save(optimizer.state_dict(), optimizer_save_path)
                    patience = 0
                else:
                    patience += 1
                    if patience == int(args.max_patience):
                        decay_num += 1
                        if decay_num == int(args.max_decay):
                            print('Early stop. Save result model to %s' % model_save_path)
                            return
                        lr = optimizer.param_groups[0]['lr'] * float(args.lr_decay)
                        model = bilstm_crf.BiLSTMCRF.load(model_save_path, device)
                        optimizer.load_state_dict(torch.load(optimizer_save_path))
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                        patience = 0
                print('dev: epoch %d, iter %d, dev_loss %f, patience %d, decay_num %d' %
                      (epoch + 1, train_iter, dev_loss, patience, decay_num))
                cum_start = time.time()
                if train_iter % log_every == 0:
                    record_start = time.time()
    print('Reached %d epochs, Save result model to %s' % (max_epoch, model_save_path))


def test(args):
    """ Testing the model
    Args:
        args: dict that contains options in command
    """
    sent_vocab = Vocab.load(args.SENT_VOCAB)
    tag_vocab = Vocab.load(args.TAG_VOCAB)
    sentences, tags = utils.read_corpus(args.TEST)
    sentences = utils.words2indices(sentences, sent_vocab)
    tags = utils.words2indices(tags, tag_vocab)
    test_data = list(zip(sentences, tags))
    print('num of test samples: %d' % (len(test_data)))

    # device = torch.device('cuda' if args.cuda else 'cpu')
    # device = torch.device('cuda:0')
    device = torch.device("cpu")
    model = bilstm_crf.BiLSTMCRF.load(args.MODEL, device)
    print('start testing...')
    print('using device', device)

    start = time.time()
    n_iter, num_words = 0, 0
    tp, fp, fn = 0, 0, 0

    model.eval()
    with torch.no_grad():
        for sentences, tags in utils.batch_iter(test_data, batch_size=int(args.batch_size), shuffle=False):
            sentences, sent_lengths = utils.pad(sentences, sent_vocab[sent_vocab.PAD], device)
            predicted_tags = model.predict(sentences, sent_lengths)
            n_iter += 1
            num_words += sum(sent_lengths)
            for tag, predicted_tag in zip(tags, predicted_tags):
                current_tp, current_fp, current_fn = cal_statistics(tag, predicted_tag, tag_vocab)
                tp += current_tp
                fp += current_fp
                fn += current_fn
            if n_iter % int(args.log_every) == 0:
                print('log: iter %d, %.1f words/sec, precision %f, recall %f, f1_score %f, time %.1f sec' %
                      (n_iter, num_words / (time.time() - start), tp / (tp + fp), tp / (tp + fn),
                       (2 * tp) / (2 * tp + fp + fn), time.time() - start))
                num_words = 0
                start = time.time()
    print('tp = %d, fp = %d, fn = %d' % (tp, fp, fn))
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = (2 * tp) / (2 * tp + fp + fn)
    print('Precision: %f, Recall: %f, F1 score: %f' % (precision, recall, f1_score))


def cal_dev_loss(model, dev_data, batch_size, sent_vocab, tag_vocab, device):
    """ Calculate loss on the development data
    Args:
        model: the model being trained
        dev_data: development data
        batch_size: batch size
        sent_vocab: sentence vocab
        tag_vocab: tag vocab
        device: torch.device on which the model is trained
    Returns:
        the average loss on the dev data
    """
    is_training = model.training
    model.eval()
    loss, n_sentences = 0, 0
    with torch.no_grad():
        for sentences, tags in utils.batch_iter(dev_data, batch_size, shuffle=False):
            sentences, sent_lengths = utils.pad(sentences, sent_vocab[sent_vocab.PAD], device)
            tags, _ = utils.pad(tags, tag_vocab[sent_vocab.PAD], device)
            batch_loss = model(sentences, tags, sent_lengths)  # shape: (b,)
            loss += batch_loss.sum().item()
            n_sentences += len(sentences)
    model.train(is_training)
    return loss / n_sentences

def cal_f1_score(model, dev_data, batch_size, sent_vocab, tag_vocab, device):
    """
    :param model:
    :param dev_data:
    :param batch_size:
    :param sent_vocab:
    :param tag_vocab:
    :param device:
    :return:
    """
    is_training = model.training
    n_iter, num_words = 0, 0
    tp, fp, fn = 0, 0, 0
    model.eval()
    with torch.no_grad():
        for sentences, tags in utils.batch_iter(dev_data, batch_size=int(batch_size), shuffle=False):
            sentences, sent_lengths = utils.pad(sentences, sent_vocab[sent_vocab.PAD], device)
            predicted_tags = model.predict(sentences, sent_lengths)

            n_iter += 1
            num_words += sum(sent_lengths)
            for tag, predicted_tag in zip(tags, predicted_tags):
                current_tp, current_fp, current_fn = cal_statistics(tag, predicted_tag, tag_vocab)
                tp += current_tp
                fp += current_fp
                fn += current_fn

            if n_iter % int(args.log_every) == 0:
                print('log: iter %d precision %f, recall %f, f1_score %f' %
                      (n_iter, tp / (tp + fp), tp / (tp + fn),
                       (2 * tp) / (2 * tp + fp + fn)))
                num_words = 0
                start = time.time()
    print('tp = %d, fp = %d, fn = %d' % (tp, fp, fn))
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = (2 * tp) / (2 * tp + fp + fn)
    print('Precision: %f, Recall: %f, F1 score: %f' % (precision, recall, f1_score))
    model.train(is_training)



def cal_statistics(tag, predicted_tag, tag_vocab):
    """ Calculate TN, FN, FP for the given true tag and predicted tag.
    Args:
        tag (list[int]): true tag
        predicted_tag (list[int]): predicted tag
        tag_vocab: tag vocab
    Returns:
        tp: true positive
        fp: false positive
        fn: false negative
    """
    tp, fp, fn = 0, 0, 0

    def func(tag1, tag2):
        a, b, i = 0, 0, 0
        while i < len(tag1):
            if tag1[i] == tag_vocab['-']:
                i += 1
                continue
            begin, end = i, i
            while end + 1 < len(tag1) and tag1[end + 1] != tag_vocab['-']:
                end += 1
            equal = True
            for j in range(max(0, begin - 1), min(len(tag1), end + 2)):
                if tag1[j] != tag2[j]:
                    equal = False
                    break
            a, b = a + equal, b + 1 - equal
            i = end + 1
        return a, b
    t, f = func(tag, predicted_tag)
    tp += t
    fn += f
    t, f = func(predicted_tag, tag)
    fp += f
    return tp, fp, fn


def main(args):

    for i in range(1):
        random.seed(0)
        torch.manual_seed(0)

        if args.train:
            train(args)
        if args.test:
            test(args)


if __name__ == '__main__':
    """
    Options:
        --dropout_rate=<float>              dropout rate [default: 0.5]
        --embed_size=<int>                  size of word embedding [default: 256]
        --hidden_size=<int>                 size of hidden state [default: 256]
        --batch_size=<int>                  batch-size [default: 32]
        --max_epoch=<int>                   max epoch [default: 10]
        --clip_max_norm=<float>             clip max norm [default: 5.0]
        --lr=<float>                        learning rate [default: 0.001]
        --log_every=<int>                   log every [default: 10]
        --validation_every=<int>            validation every [default: 250]
        --patience_threshold=<float>        patience threshold [default: 0.98]
        --max_patience=<int>                time of continuous worse performance to decay lr [default: 4]
        --max_decay=<int>                   time of lr decay to early stop [default: 4]
        --lr_decay=<float>                  decay rate of lr [default: 0.5]
        --model_save_path=<file>            model save path [default: ./model/model.pth]
        --optimizer_save_path=<file>        optimizer save path [default: ./model/optimizer.pth]
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--cpu', type=int, default=1)

    parser.add_argument('--train', type=bool, default=False)
    parser.add_argument('--test', type=bool, default=True)

    parser.add_argument('--TRAIN', type=str, default='./editData/train.txt')
    parser.add_argument('--TEST', type=str, default='./editData/test.txt')

    parser.add_argument('--SENT_VOCAB', type=str, default='./vocab/merge_sent_vocab.json')
    parser.add_argument('--TAG_VOCAB', type=str, default='./vocab/merge_tag_vocab.json')

    parser.add_argument('--MODEL', type=str, default='./model/model_merge.pth')

    parser.add_argument('--word2vec_path', type=str, default='./model/word2vec/ko.bin')

    parser.add_argument('--dropout_rate', type=float, default=0.5)
    parser.add_argument('--embed_size', type=int, default=256)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_epoch', type=int, default=8)
    parser.add_argument('--clip_max_norm', type=float, default=5.0)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--log_every', type=int, default=10)
    parser.add_argument('--validation_every', type=int, default=250)
    parser.add_argument('--patience_threshold', type=float, default=0.98)
    parser.add_argument('--max_patience', type=int, default=4)
    parser.add_argument('--max_decay', type=int, default=6)
    parser.add_argument('--lr_decay', type=float, default=0.5)
    parser.add_argument('--model_save_path', type=str, default='./model/model.pth')
    parser.add_argument('--optimizer_save_path', type=str, default='./model/optimizer.pth')

    args = parser.parse_args()
    main(args)
