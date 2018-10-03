import tensorflow as tf
import random
from tqdm import tqdm
import spacy
import ujson as json
from collections import Counter
import numpy as np
from codecs import open

'''
This file is taken and modified from R-Net by HKUST-KnowComp
https://github.com/HKUST-KnowComp/R-Net
'''

nlp = spacy.blank("en")
nlp.add_pipe(nlp.create_pipe('sentencizer'))


def tokenize(text, tokenzie_sent=False):
    if tokenzie_sent:
        return [[token.text for token in nlp(sent.text)] for sent in nlp(text).sents]
    else:
        return [token.text for token in nlp(text)]


def convert_idx(text, sents):
    current = 0
    spans = []
    for sent in sents:
        spans.append([])
        for token in sent:
            current = text.find(token, current)
            if current < 0:
                print("Token {} cannot be found".format(token))
                raise Exception()
            spans[-1].append((current, current + len(token)))
            current += len(token)
    return spans


def process_file(filename, data_type, word_counter, char_counter, answer_notation=False, lower_word=False):
    print("Generating {} examples...".format(data_type))
    examples = []
    eval_examples = {}
    sent_counts = []
    total = 0
    max_s, max_w, max_q, max_a = 0, 0, 0, 0
    with open(filename, "r") as fh:
        source = json.load(fh)
        for article in tqdm(source["data"]):
            for para in article["paragraphs"]:
                context = para["context"].replace(
                        u"''", u'" ').replace(u"``", u'" ').replace(u'\u2013', '-')
                if lower_word:
                    context = context.lower()
                context_tokens = tokenize(context, tokenzie_sent=True)
                max_s = max(max_s, len(context_tokens))
                sent_counts.append(len(context_tokens))
                max_w = max(max_w, max(map(len, context_tokens)))
                context_chars = [[list(token) for token in sent] for sent in context_tokens]
                spans = convert_idx(context, context_tokens)
                for sent in context_tokens:
                    for token in sent:
                        word_counter[token] += len(para["qas"])
                        for char in token:
                            char_counter[char] += len(para["qas"])
                for qa in para["qas"]:
                    total += 1
                    ques = qa["question"].replace(
                            "''", '" ').replace("``", '" ').replace(u'\u2013', '-')
                    if lower_word:
                        ques = ques.lower()
                    ques_tokens = ["--GO--"] + tokenize(ques) + ["--EOS--"]
                    max_q = max(max_q, len(ques_tokens))
                    ques_chars = [list(token) for token in ques_tokens]
                    for token in ques_tokens:
                        word_counter[token] += 1
                        for char in token:
                            char_counter[char] += 1
                    y1s, y2s = [], []
                    answer_texts = []
                    answer_tokens = []
                    answer_chars = []
                    for answer in qa["answers"]:
                        answer_text = answer["text"].replace(u'\u2013', '-')
                        if lower_word:
                            answer_text = answer_text.lower()
                        answer_start = answer['answer_start']
                        answer_end = answer_start + len(answer_text)
                        answer_texts.append(answer_text)
                        answer_tokens.append(["--GO--"] + tokenize(answer_text)+ ["--EOS--"])
                        answer_chars.append([list(token) for token in answer_tokens[-1]])
                        max_a = max(max_a, len(answer_tokens[-1]))
                        answer_span = []
                        for idx, sent in enumerate(spans):
                            for jdx, span in enumerate(sent):
                                if not (answer_end <= span[0] or answer_start >= span[1]):
                                    answer_span.append((idx, jdx))
                        y1, y2 = answer_span[0], answer_span[-1]
                        y1s.append(y1)
                        y2s.append(y2)
                    example = {"context_tokens": context_tokens, "context_chars": context_chars,
                               "ques_tokens": ques_tokens, "ques_chars": ques_chars,
                               "ans_tokens": answer_tokens, "ans_chars": answer_chars,
                               "y1s": y1s, "y2s": y2s, "id": total}
                    examples.append(example)
                    eval_examples[str(total)] = {
                        "context": context, "context_tokens": context_tokens,
                        "spans": spans, "questions": [ques], "answers": answer_texts, "uuid": qa["id"]}
        random.shuffle(examples)
        print("{} questions in total".format(len(examples)))
        print("max_s, max_w, max_q, max_a: {}, {}, {}, {}".format(max_s, max_w, max_q, max_a))
        print Counter(sent_counts)
    return examples, eval_examples


def get_embedding(counter, data_type, limit=0, emb_file=None, size=None, vec_size=None, size_limit=0):
    print("Generating {} embedding...".format(data_type))
    embedding_dict = {}
    filtered_elements = counter
    if limit > 0:
        filtered_elements = {k: v for k, v in counter.items() if v > limit}
    if size_limit > 0:
        filtered_elements = dict(sorted(counter.items(), key=lambda x: x[1], reverse=True)[:size_limit])
    if emb_file is not None:
        assert size is not None
        assert vec_size is not None
        with open(emb_file, "r", encoding="utf-8") as fh:
            for line in tqdm(fh, total=size):
                array = line.split()
                word = "".join(array[0:-vec_size])
                vector = list(map(float, array[-vec_size:]))
                if word in filtered_elements:
                    embedding_dict[word] = vector
        print("{} / {} tokens have corresponding {} embedding vector".format(
                len(embedding_dict), len(filtered_elements), data_type))
    else:
        assert vec_size is not None
        for token in filtered_elements:
            embedding_dict[token] = [np.random.normal(
                    scale=0.1) for _ in range(vec_size)]
        print("{} tokens have corresponding embedding vector".format(
                len(filtered_elements)))

    GO  = "--GO--"
    NULL = "--NULL--"
    OOV = "--OOV--"
    EOS = "--EOS--"

    token2idx_dict = {token: idx for idx, token in enumerate(embedding_dict.keys(), 4)}
    token2idx_dict[NULL] = 0
    token2idx_dict[OOV] = 1
    token2idx_dict[GO] = 2
    token2idx_dict[EOS] = 3
    embedding_dict[NULL] = np.random.normal(size=vec_size)
    embedding_dict[OOV] = np.random.normal(size=vec_size)
    embedding_dict[GO] = np.random.normal(size=vec_size)
    embedding_dict[EOS] = np.random.normal(size=vec_size)

    idx2emb_dict = {idx: embedding_dict[token]
                    for token, idx in token2idx_dict.items()}
    emb_mat = [idx2emb_dict[idx] for idx in range(len(idx2emb_dict))]
    return emb_mat, token2idx_dict


def build_features(config, examples, data_type, out_file, word2idx_dict,
                   char2idx_dict, is_test=False, answer_notation=False):
    num_sent_limit = config.test_num_sent_limit if is_test else config.num_sent_limit
    sent_limit = config.test_sent_limit if is_test else config.sent_limit
    ques_limit = config.test_ques_limit if is_test else config.ques_limit
    ans_limit = config.test_ans_limit if is_test else config.ans_limit
    char_limit = config.char_limit

    def filter_func(example, is_test=False):
        return len(example["context_tokens"]) > num_sent_limit or \
               max(map(len, example["context_tokens"])) > sent_limit or \
               len(example["ques_tokens"]) > ques_limit or \
               len(example["ans_tokens"][0]) > ans_limit

    print("Processing {} examples...".format(data_type))
    writer = tf.python_io.TFRecordWriter(out_file)
    total = 0
    total_ = 0
    meta = {}
    for example in tqdm(examples):
        total_ += 1

        if filter_func(example, is_test):
            continue

        total += 1
        context_idxs = np.zeros([num_sent_limit, sent_limit], dtype=np.int32)
        context_char_idxs = np.zeros([num_sent_limit, sent_limit, char_limit], dtype=np.int32)
        ques_idxs = np.zeros([ques_limit], dtype=np.int32)
        ques_char_idxs = np.zeros([ques_limit, char_limit], dtype=np.int32)
        ans_idxs = np.zeros([ans_limit], dtype=np.int32)
        ans_char_idxs = np.zeros([ans_limit, char_limit], dtype=np.int32)
        y1 = np.zeros([num_sent_limit, sent_limit], dtype=np.float32)
        y2 = np.zeros([num_sent_limit, sent_limit], dtype=np.float32)

        start, end = example["y1s"][0], example["y2s"][0]
        y1[start[0]][start[1]] = 1.0
        y2[end[0]][end[1]] = 1.0

        def _get_word(word, i):
            for each in (word, word.lower(), word.capitalize(), word.upper()):
                if each in word2idx_dict:
                    return word2idx_dict[each]
            return 1

        def _get_char(char):
            if char in char2idx_dict:
                return char2idx_dict[char]
            return 1

        for i, sent in enumerate(example["context_tokens"]):
            for j, token in enumerate(sent):
                wid = _get_word(token, i)
                context_idxs[i, j] = wid

        for i, sent in enumerate(example["ques_tokens"]):
            for j, token in enumerate(sent):
                wid = _get_word(token, i)
                ques_idxs[i, j] = wid

        for i, sent in enumerate(example["ans_tokens"][0]):
            for j, token in enumerate(sent):
                wid = _get_word(token, i)
                ans_idxs[i] = wid

        for i, sent in enumerate(example["context_chars"]):
            for j, token in enumerate(sent):
                for k, char in enumerate(token):
                    if k == char_limit:
                        break
                    context_char_idxs[i, j, k] = _get_char(char)

        for i, sent in enumerate(example["ques_chars"]):
            for j, token in enumerate(sent):
                for k, char in enumerate(token):
                    if k == char_limit:
                        break
                    ques_char_idxs[i, j, k] = _get_char(char)

        for i, token in enumerate(example["ans_chars"][0]):
            for j, char in enumerate(token):
                if j == char_limit:
                    break
                ans_char_idxs[i, j] = _get_char(char)

        record = tf.train.Example(features=tf.train.Features(feature={
            "context_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[context_idxs.tostring()])),
            "ques_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[ques_idxs.tostring()])),
            "ans_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[ans_idxs.tostring()])),
            "context_char_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[context_char_idxs.tostring()])),
            "ques_char_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[ques_char_idxs.tostring()])),
            "ans_char_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[ans_char_idxs.tostring()])),
            "y1": tf.train.Feature(bytes_list=tf.train.BytesList(value=[y1.tostring()])),
            "y2": tf.train.Feature(bytes_list=tf.train.BytesList(value=[y2.tostring()])),
            "id": tf.train.Feature(int64_list=tf.train.Int64List(value=[example["id"]]))
        }))
        writer.write(record.SerializeToString())
    print("Built {} / {} instances of features in total".format(total, total_))
    meta["total"] = total
    writer.close()
    return meta


def save(filename, obj, message=None):
    if message is not None:
        print("Saving {}...".format(message))
        with open(filename, "w") as fh:
            json.dump(obj, fh)


def bprepro(config):
    word_counter, char_counter = Counter(), Counter()
    train_examples, train_eval = process_file(config.train_file, "train", word_counter,
            char_counter, answer_notation=config.answer_notation, lower_word=config.lower_word)
    dev_examples, dev_eval = process_file(config.dev_file, "dev", word_counter,
            char_counter, answer_notation=config.answer_notation, lower_word=config.lower_word)
    test_examples, test_eval = process_file(config.test_file, "test", word_counter,
            char_counter, answer_notation=config.answer_notation, lower_word=config.lower_word)
    exit()

    word_emb_file = config.glove_word_file
    char_emb_file = config.glove_char_file if config.pretrained_char else None
    char_emb_size = config.glove_char_size if config.pretrained_char else None
    char_emb_dim = config.glove_dim if config.pretrained_char else config.char_dim

    word_emb_mat, word2idx_dict = get_embedding(word_counter, "word", emb_file=word_emb_file,
            size=config.glove_word_size, vec_size=config.glove_dim, limit=config.vocab_count_limit)
    print len(word2idx_dict)
    char_emb_mat, char2idx_dict = get_embedding(char_counter, "char", emb_file=char_emb_file,
            size=char_emb_size, vec_size=char_emb_dim, limit=config.char_count_limit)
    print len(char2idx_dict)

    build_features(config, train_examples, "train", config.train_record_file, word2idx_dict,
                   char2idx_dict, answer_notation=config.answer_notation)
    dev_meta = build_features(config, dev_examples, "dev", config.dev_record_file, word2idx_dict, char2idx_dict,
                              answer_notation=config.answer_notation)
    test_meta = build_features(config, test_examples, "test", config.test_record_file, word2idx_dict, char2idx_dict,
                               is_test=True, answer_notation=config.answer_notation)

    save(config.word_emb_file, word_emb_mat, message="word embedding")
    save(config.char_emb_file, char_emb_mat, message="char embedding")
    save(config.train_eval_file, train_eval, message="train eval")
    save(config.dev_eval_file, dev_eval, message="dev eval")
    save(config.test_eval_file, test_eval, message="test eval")
    save(config.dev_meta, dev_meta, message="dev meta")
    save(config.test_meta, test_meta, message="test meta")
    save(config.word_dictionary, word2idx_dict, message="word dictionary")
    save(config.char_dictionary, char2idx_dict, message="char dictionary")


def prepro_rerank(config):
    para_limit = config.test_para_limit
    ques_limit = config.test_ques_limit
    ans_limit = config.test_ans_limit
    char_limit = config.char_limit

    with open(config.beam_search_file, "r") as fh:
        answer_dict = json.load(fh)
    with open(config.word_dictionary, "r") as fh:
        word2idx_dict = json.load(fh)
    with open(config.char_dictionary, "r") as fh:
        char2idx_dict = json.load(fh)

    word_counter, char_counter = Counter(), Counter()
    test_examples, test_eval = process_file(
         config.test_file, "test", word_counter, char_counter)

    writer = tf.python_io.TFRecordWriter(config.rerank_file)

    total = 0
    for test_example in test_examples:
        candidate_answers = answer_dict[str(test_example["id"])]
        candidate_answer_tokens = []
        candidate_answer_chars = []
        for candidate_answer, _, _, _ in candidate_answers:
            candidate_answer_tokens.append(word_tokenize(candidate_answer))
            candidate_answer_chars.append([list(token) for token in candidate_answer_tokens[-1]])

        context_idxs = np.zeros([para_limit], dtype=np.int32)
        context_char_idxs = np.zeros([para_limit, char_limit], dtype=np.int32)
        # context_voc = np.zeros([len(word2idx_dict) + para_limit], dtype=np.int32)
        ques_idxs = np.zeros([ques_limit], dtype=np.int32)
        ans_idxs = [np.zeros([ans_limit], dtype=np.int32) for _ in range(len(candidate_answers))]
        ques_char_idxs = np.zeros([ques_limit, char_limit], dtype=np.int32)
        ans_char_idxs = [np.zeros([ans_limit, char_limit], dtype=np.int32) for _ in range(len(candidate_answers))]
        y1 = np.zeros([para_limit], dtype=np.float32)
        y2 = np.zeros([para_limit], dtype=np.float32)

        def _get_word(word, i):
            for each in (word, word.lower(), word.capitalize(), word.upper()):
                if each in word2idx_dict:
                    return word2idx_dict[each]
            return len(word2idx_dict) + i

        def _get_char(char):
            if char in char2idx_dict:
                return char2idx_dict[char]
            return 1

        for i, token in enumerate(test_example["context_tokens"]):
            wid = _get_word(token, i)
            context_idxs[i] = wid
            # if wid < len(word2idx_dict):
            #     context_voc[wid] = 1

        for i, token in enumerate(test_example["ques_tokens"]):
            ques_idxs[i] = _get_word(token, i)

        for k, candidate_answer_token in enumerate(candidate_answer_tokens):
            for i, token in enumerate(candidate_answer_token):
                ans_idxs[k][i] = _get_word(token, i)

        for k, candidate_answer_char in enumerate(candidate_answer_chars):
            for i, token in enumerate(candidate_answer_char):
                for j, char in enumerate(token):
                    if j == char_limit:
                        break
                    ans_char_idxs[k][i, j] = _get_char(char)

        for i, token in enumerate(test_example["context_chars"]):
            for j, char in enumerate(token):
                if j == char_limit:
                    break
                context_char_idxs[i, j] = _get_char(char)

        for i, token in enumerate(test_example["ques_chars"]):
            for j, char in enumerate(token):
                if j == char_limit:
                    break
                ques_char_idxs[i, j] = _get_char(char)

        start, end = test_example["y1s"][-1], test_example["y2s"][-1]
        y1[start], y2[end] = 1.0, 1.0

        for i, (ans_idx, ans_char_idx) in enumerate(zip(ans_idxs, ans_char_idxs)):
            total += 1
            record = tf.train.Example(features=tf.train.Features(feature={
                "context_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[context_idxs.tostring()])),
                "ques_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[ques_idxs.tostring()])),
                "ans_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[ans_idx.tostring()])),
                "context_char_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[context_char_idxs.tostring()])),
                "ques_char_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[ques_char_idxs.tostring()])),
                "ans_char_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[ans_char_idx.tostring()])),
                "y1": tf.train.Feature(bytes_list=tf.train.BytesList(value=[y1.tostring()])),
                "y2": tf.train.Feature(bytes_list=tf.train.BytesList(value=[y2.tostring()])),
                "id": tf.train.Feature(int64_list=tf.train.Int64List(value=[test_example["id"]])),
                "cid": tf.train.Feature(int64_list=tf.train.Int64List(value=[i]))
            }))
            writer.write(record.SerializeToString())
    writer.close()
    meta = {}
    print total
    meta["total"] = total
    save(config.rerank_meta, meta, message="test meta")