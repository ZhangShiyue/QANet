import tensorflow as tf
from layers import conv, highway, trilinear, mask_logits, optimized_trilinear_for_attention
from qanet import BasicModel


class BiDAFModel(BasicModel):
    def __init__(self, context, context_mask, context_char, question, question_mask, ques_char,
                 y1, y2, word_mat, char_mat, dropout, batch_size, para_limit,
                 ques_limit, ans_limit, char_limit, hidden, char_dim, word_dim):
        BasicModel.__init__(self, context, context_mask, context_char, question, question_mask, ques_char,
                            y1, y2, word_mat, char_mat, dropout, batch_size, para_limit, ques_limit, ans_limit,
                            char_limit, hidden, char_dim, word_dim)

        self.c_len = tf.reduce_sum(tf.cast(self.c_mask, tf.int32), axis=-1)
        self.q_len = tf.reduce_sum(tf.cast(self.q_mask, tf.int32), axis=-1)

        self.cells = []
        for i in range(8):
            self.cells.append(tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(hidden),
                                                            input_keep_prob=1.0 - self.dropout))

    def build_model(self, global_step):
        # word, character embedding
        c_emb, q_emb = self.input_embedding()
        c, q = self.input_encoder(c_emb, q_emb)
        attention_outputs = self.optimized_bidaf_attention(c, q)
        self.logits1, self.logits2 = self.model_encoder(attention_outputs)
        losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits1, labels=tf.reshape(self.y1, [self.N, -1]))
        losses2 = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits2, labels=tf.reshape(self.y2, [self.N, -1]))
        batch_loss = losses + losses2
        return tf.reduce_mean(batch_loss), batch_loss

    def sample(self, beam_size):
        outer = tf.matmul(tf.expand_dims(tf.nn.softmax(self.logits1), axis=2),
                          tf.expand_dims(tf.nn.softmax(self.logits2), axis=1))
        outer = tf.matrix_band_part(outer, 0, self.AL)
        bprobs, bindex = tf.nn.top_k(tf.reshape(outer, [-1, self.PL * self.PL]), k=beam_size)
        byp1 = bindex // self.PL
        byp2 = bindex % self.PL
        bprobs = -tf.log(bprobs)
        return byp1, byp2, bprobs

    def input_encoder(self, c_emb, q_emb):
        with tf.variable_scope("Input_Encoder_Layer"):
            (ch_fw, ch_bw), _ = tf.nn.bidirectional_dynamic_rnn(self.cells[0], self.cells[1], c_emb,
                                                                sequence_length=self.c_len,
                                                                dtype='float', scope='input_encoder')
            c = tf.concat([ch_fw, ch_bw], axis=-1)
            (qh_fw, qh_bw), _ = tf.nn.bidirectional_dynamic_rnn(self.cells[0], self.cells[1], q_emb,
                                                                sequence_length=self.q_len,
                                                                dtype='float', scope='input_encoder')
            q = tf.concat([qh_fw, qh_bw], axis=-1)
            return c, q

    def model_encoder(self, attention_outputs, num_layers=3):
        with tf.variable_scope("Model_Encoder_Layer"):
            p0 = tf.reshape(tf.concat(attention_outputs, axis=-1), [self.N, self.PL, -1])
            (g0h_fw, g0h_bw), _ = tf.nn.bidirectional_dynamic_rnn(self.cells[2], self.cells[3], p0,
                                                                  sequence_length=self.c_len,
                                                                  dtype='float', scope="g0")
            g0 = tf.concat([g0h_fw, g0h_bw], axis=-1)
            (g1h_fw, g1h_bw), _ = tf.nn.bidirectional_dynamic_rnn(self.cells[4], self.cells[5], g0,
                                                                  sequence_length=self.c_len,
                                                                  dtype='float', scope='g1')
            g1 = tf.concat([g1h_fw, g1h_bw], axis=-1)
            logits1 = tf.squeeze(conv(tf.nn.dropout(tf.concat([g1, p0], axis=-1), keep_prob=1.0 - self.dropout),
                                      1, bias=False, name="start_pointer"), -1)
            logits1 = mask_logits(tf.reshape(logits1, [self.N, -1]), tf.reshape(self.c_mask, [self.N, -1]))

            ali = tf.reduce_sum(tf.expand_dims(tf.nn.softmax(logits1), -1) * g1, [1])
            ali = tf.tile(tf.expand_dims(ali, 1), [1, self.PL, 1])

            (g2h_fw, g2h_bw), _ = tf.nn.bidirectional_dynamic_rnn(self.cells[6], self.cells[7],
                                                                  tf.concat([p0, g1, ali, g1 * ali], axis=-1),
                                                                  self.c_len, dtype='float', scope='g2')
            g2 = tf.concat([g2h_fw, g2h_bw], axis=-1)
            logits2 = tf.squeeze(conv(tf.nn.dropout(tf.concat([g2, p0], axis=-1), keep_prob=1.0 - self.dropout),
                                      1, bias=False, name="end_pointer"), -1)
            logits2 = mask_logits(tf.reshape(logits2, [self.N, -1]), tf.reshape(self.c_mask, [self.N, -1]))
            return logits1, logits2
