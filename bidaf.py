import tensorflow as tf
from layers import conv, highway, trilinear, mask_logits, optimized_trilinear_for_attention


class BiDAFModel(object):
    def __init__(self, context, context_mask, context_char, question, question_mask, ques_char,
                 y1, y2, word_mat, char_mat, dropout, batch_size, num_sent_limit, sent_limit,
                 ques_limit, char_limit, hidden, char_dim, word_dim):
        self.c = context
        self.c_mask = context_mask
        self.ch = context_char
        self.q = question
        self.q_mask = question_mask
        self.qh = ques_char
        self.y1 = y1
        self.y2 = y2
        self.word_mat = word_mat
        self.char_mat = char_mat
        self.c_len = tf.reduce_sum(tf.cast(self.c_mask, tf.int32), axis=-1)
        self.q_len = tf.reduce_sum(tf.cast(self.q_mask, tf.int32), axis=-1)

        self.dropout = dropout
        self.N = batch_size
        self.NS = num_sent_limit
        self.SL = sent_limit
        self.QL = ques_limit
        self.CL = char_limit
        self.d = hidden
        self.dc = char_dim
        self.dw = word_dim

        self.cell = tf.nn.rnn_cell.BasicLSTMCell(hidden)
        self.d_cell = tf.nn.rnn_cell.DropoutWrapper(self.cell, input_keep_prob=self.dropout)

    def build_model(self, global_step):
        # word, character embedding
        c_emb, q_emb = self.input_embedding()
        c, q = self.input_encoder(c_emb, q_emb)
        attention_outputs = self.optimized_bidaf_attention(c, q)
        self.model_encoder(attention_outputs)


    def input_embedding(self):
        with tf.variable_scope("Input_Embedding_Layer"):
            ch_emb = tf.reshape(tf.nn.embedding_lookup(
                    self.char_mat, self.ch), [self.N * self.NS * self.SL, self.CL, self.dc])
            qh_emb = tf.reshape(tf.nn.embedding_lookup(
                    self.char_mat, self.qh), [self.N * self.QL, self.CL, self.dc])
            ch_emb = tf.nn.dropout(ch_emb, 1.0 - 0.5 * self.dropout)
            qh_emb = tf.nn.dropout(qh_emb, 1.0 - 0.5 * self.dropout)

            # Bidaf style conv-highway encoder
            ch_emb = conv(ch_emb, self.dc, bias=True, activation=tf.nn.relu, kernel_size=5, name="char_conv",
                          reuse=None)
            qh_emb = conv(qh_emb, self.dc, bias=True, activation=tf.nn.relu, kernel_size=5, name="char_conv",
                          reuse=True)

            ch_emb = tf.reduce_max(ch_emb, axis=1)
            qh_emb = tf.reduce_max(qh_emb, axis=1)

            ch_emb = tf.reshape(ch_emb, [self.N * self.NS, self.SL, -1])
            qh_emb = tf.reshape(qh_emb, [self.N, self.QL, -1])

            c_emb = tf.nn.dropout(tf.nn.embedding_lookup(self.word_mat, self.c), 1.0 - self.dropout)
            c_emb = tf.reshape(c_emb, [self.N * self.NS, self.SL, -1])
            q_emb = tf.nn.dropout(tf.nn.embedding_lookup(self.word_mat, self.q), 1.0 - self.dropout)

            c_emb = tf.concat([c_emb, ch_emb], axis=-1)
            q_emb = tf.concat([q_emb, qh_emb], axis=-1)

            c_emb = highway(c_emb, scope="highway", dropout=self.dropout, reuse=None)
            q_emb = highway(q_emb, scope="highway", dropout=self.dropout, reuse=True)

            return c_emb, q_emb

    def input_encoder(self, c_emb, q_emb):
        with tf.variable_scope("Input_Encoder_Layer"):
            (ch_fw, ch_bw), _ = tf.nn.bidirectional_dynamic_rnn(self.d_cell, self.d_cell, c_emb,
                                                                sequence_length=tf.reshape(self.c_len, [-1]),
                                                                dtype='float', scope='input_encoder')
            c = tf.concat([ch_fw, ch_bw], axis=-1)
            (qh_fw, qh_bw), _ = tf.nn.bidirectional_dynamic_rnn(self.d_cell, self.d_cell, q_emb,
                                                                sequence_length=self.q_len,
                                                                dtype='float', scope='input_encoder')
            q = tf.concat([qh_fw, qh_bw], axis=-1)

            return c, q

    def bidaf_attention(self, c, q):
        with tf.variable_scope("BiDAF"):
            # reshape
            c = tf.reshape(c, [self.N, self.NS * self.SL, -1])
            c_mask = tf.reshape(self.c_mask, [self.N, self.NS * self.SL])
            # BiDAF
            C = tf.tile(tf.expand_dims(c, 2), [1, 1, self.QL, 1])
            Q = tf.tile(tf.expand_dims(q, 1), [1, self.NS * self.SL, 1, 1])
            S = trilinear([C, Q, C * Q], input_keep_prob=1.0 - self.dropout)
            mask_q = tf.expand_dims(self.q_mask, 1)
            S_ = tf.nn.softmax(mask_logits(S, mask=mask_q))
            mask_c = tf.expand_dims(c_mask, 2)
            S_T = tf.transpose(tf.nn.softmax(mask_logits(S, mask=mask_c), dim=1), (0, 2, 1))
            self.c2q = tf.matmul(S_, q)
            self.q2c = tf.matmul(tf.matmul(S_, S_T), c)
            attention_outputs = [c, self.c2q, c * self.c2q, c * self.q2c]
            return attention_outputs

    def optimized_bidaf_attention(self, c, q):
        with tf.variable_scope("BiDAF"):
            # reshape
            c = tf.reshape(c, [self.N, self.NS * self.SL, -1])
            c_mask = tf.reshape(self.c_mask, [self.N, self.NS * self.SL])
            # BiDAF
            S = optimized_trilinear_for_attention([c, q], self.NS * self.SL, self.QL, input_keep_prob=1.0 - self.dropout)
            mask_q = tf.expand_dims(self.q_mask, 1)
            S_ = tf.nn.softmax(mask_logits(S, mask=mask_q))
            mask_c = tf.expand_dims(c_mask, 2)
            S_T = tf.transpose(tf.nn.softmax(mask_logits(S, mask=mask_c), dim=1), (0, 2, 1))
            self.c2q = tf.matmul(S_, q)
            self.q2c = tf.matmul(tf.matmul(S_, S_T), c)
            attention_outputs = [c, self.c2q, c * self.c2q, c * self.q2c]
            return attention_outputs

    def model_encoder(self, attention_outputs, num_layers=3):
        with tf.variable_scope("Model_Encoder_Layer"):
            inputs = tf.reshape(tf.concat(attention_outputs, axis=-1), [self.N * self.NS, self.SL, -1])
            print inputs
            d_cell = self.d_cell
            (g0h_fw, g0h_bw), _ = tf.nn.bidirectional_dynamic_rnn(d_cell, d_cell, inputs,
                                                              sequence_length=tf.reshape(self.c_len, [-1]),
                                                              dtype='float')
            exit()
            # g0 = tf.concat([g0h_fw, g0h_bw], axis=-1)
            # (g1h_fw, g1h_bw), _ = tf.nn.bidirectional_dynamic_rnn(self.d_cell, self.d_cell, g0,
            #                                                   sequence_length=tf.reshape(self.c_len, [-1]),
            #                                                   dtype='float', scope='model_encoder1')
            # g1 = tf.concat([g1h_fw, g1h_bw], axis=-1)
            # print inputs, g0, g1
            # exit()
            # tf.squeeze(conv(tf.concat([g1, inputs], axis=-1), 1, bias=False, name="start_pointer"), -1)

