import tensorflow as tf
from layers import initializer, transformer_block, highway, conv, mask_logits, trilinear, total_params, \
    optimized_trilinear_for_attention, _linear, multihead_attention


class TransformerModel(object):
    def __init__(self, context, context_mask, context_char, question, question_mask, ques_char,
                 y1, y2, word_mat, char_mat, dropout, batch_size, para_limit, ques_limit,
                 char_limit, hidden, char_dim, word_dim, num_head):
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

        self.dropout = dropout
        self.N = batch_size
        self.PL = para_limit
        self.QL = ques_limit
        self.CL = char_limit
        self.d = hidden
        self.dc = char_dim
        self.dw = word_dim
        self.nh = num_head

    def build_model(self, global_step):
        # word, character embedding
        c_emb, q_emb = self.input_embedding()
        # input_encoder
        c, q = self.input_encoder(c_emb, q_emb)
        # bidaf_attention
        attention_outputs = self.optimized_bidaf_attention(c, q)
        # model_encoder
        self.model_encoder(attention_outputs)
        # span start and end prediction
        self.output()
        # compute loss
        losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits1, labels=self.y1)
        losses2 = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits2, labels=self.y2)
        return tf.reduce_mean(losses + losses2)

    def sample(self, beam_size):
        outer = tf.matmul(tf.expand_dims(tf.nn.softmax(self.logits1), axis=2),
                          tf.expand_dims(tf.nn.softmax(self.logits2), axis=1))
        outer = tf.matrix_band_part(outer, 0, self.QL)
        bprobs, bindex = tf.nn.top_k(tf.reshape(outer, [-1, self.PL * self.PL]), k=beam_size)
        byp1 = bindex // self.PL
        byp2 = bindex % self.PL
        bprobs = -tf.log(bprobs)
        return byp1, byp2, bprobs

    def input_embedding(self):
        with tf.variable_scope("Input_Embedding_Layer"):
            ch_emb = tf.reshape(tf.nn.embedding_lookup(
                    self.char_mat, self.ch), [self.N * self.PL, self.CL, self.dc])
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

            ch_emb = tf.reshape(ch_emb, [self.N, self.PL, ch_emb.shape[-1]])
            qh_emb = tf.reshape(qh_emb, [self.N, self.QL, qh_emb.shape[-1]])

            c_emb = tf.nn.dropout(tf.nn.embedding_lookup(self.word_mat, self.c), 1.0 - self.dropout)
            q_emb = tf.nn.dropout(tf.nn.embedding_lookup(self.word_mat, self.q), 1.0 - self.dropout)

            c_emb = tf.concat([c_emb, ch_emb], axis=2)
            q_emb = tf.concat([q_emb, qh_emb], axis=2)

            c_emb = highway(c_emb, scope="highway", dropout=self.dropout, reuse=None)
            q_emb = highway(q_emb, scope="highway", dropout=self.dropout, reuse=True)

            return c_emb, q_emb

    def input_encoder(self, c_emb, q_emb, num_blocks=4, reuse=True):
        with tf.variable_scope("Input_Encoder_Layer"):
            c = transformer_block(c_emb,
                               num_blocks=num_blocks,
                               mask=self.c_mask,
                               num_filters=self.d,
                               num_heads=self.nh,
                               scope="Input_Encoder_Block",
                               bias=False,
                               dropout=self.dropout,
                               input_projection=True)
            q = transformer_block(q_emb,
                               num_blocks=num_blocks,
                               mask=self.q_mask,
                               num_filters=self.d,
                               num_heads=self.nh,
                               scope="Input_Encoder_Block",
                               reuse=reuse,
                               bias=False,
                               dropout=self.dropout,
                               input_projection=True)
            return c, q

    def bidaf_attention(self, c, q):
        with tf.variable_scope("BiDAF"):
            # BiDAF
            C = tf.tile(tf.expand_dims(c, 2), [1, 1, self.QL, 1])
            Q = tf.tile(tf.expand_dims(q, 1), [1, self.PL, 1, 1])
            S = trilinear([C, Q, C * Q], input_keep_prob=1.0 - self.dropout)
            mask_q = tf.expand_dims(self.q_mask, 1)
            S_ = tf.nn.softmax(mask_logits(S, mask=mask_q))
            mask_c = tf.expand_dims(self.c_mask, 2)
            S_T = tf.transpose(tf.nn.softmax(mask_logits(S, mask=mask_c), dim=1), (0, 2, 1))
            self.c2q = tf.matmul(S_, q)
            self.q2c = tf.matmul(tf.matmul(S_, S_T), c)
            attention_outputs = [c, self.c2q, c * self.c2q, c * self.q2c]
            return attention_outputs

    def optimized_bidaf_attention(self, c, q):
        with tf.variable_scope("BiDAF"):
            S = optimized_trilinear_for_attention([c, q], self.PL, self.QL, input_keep_prob=1.0 - self.dropout)
            mask_q = tf.expand_dims(self.q_mask, 1)
            S_ = tf.nn.softmax(mask_logits(S, mask=mask_q))
            mask_c = tf.expand_dims(self.c_mask, 2)
            S_T = tf.transpose(tf.nn.softmax(mask_logits(S, mask=mask_c), dim=1), (0, 2, 1))
            self.c2q = tf.matmul(S_, q)
            self.q2c = tf.matmul(tf.matmul(S_, S_T), c)
            attention_outputs = [c, self.c2q, c * self.c2q, c * self.q2c]
            return attention_outputs

    def model_encoder(self, attention_outputs, num_layers=3, num_blocks=7):
        with tf.variable_scope("Model_Encoder_Layer"):
            inputs = tf.concat(attention_outputs, axis=-1)
            self.enc = [conv(inputs, self.d, name="input_projection")]
            for i in range(num_layers):
                if i % 2 == 0:  # dropout every 2 blocks
                    self.enc[i] = tf.nn.dropout(self.enc[i], 1.0 - self.dropout)
                self.enc.append(transformer_block(self.enc[i],
                                               num_blocks=num_blocks,
                                               mask=self.c_mask,
                                               num_filters=self.d,
                                               num_heads=self.nh,
                                               scope="Model_Encoder",
                                               bias=False,
                                               reuse=True if i > 0 else None,
                                               dropout=self.dropout))

    def output(self):
        with tf.variable_scope("Output_Layer"):
            start_logits = tf.squeeze(
                    conv(tf.concat([self.enc[1], self.enc[2]], axis=-1), 1, bias=False, name="start_pointer"), -1)
            end_logits = tf.squeeze(
                    conv(tf.concat([self.enc[1], self.enc[3]], axis=-1), 1, bias=False, name="end_pointer"), -1)
            self.logits = [mask_logits(start_logits, mask=self.c_mask),
                           mask_logits(end_logits, mask=self.c_mask)]
            self.logits1, self.logits2 = [l for l in self.logits]