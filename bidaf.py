import tensorflow as tf
from layers import conv, mask_logits, _linear, multihead_attention, vanilla_attention
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


class BiDAFGenerator(BiDAFModel):
    def __init__(self, context, context_mask, context_char, question, question_mask, ques_char, answer, answer_mask,
                 ans_char, y1, y2, word_mat, char_mat, dropout, batch_size, para_limit, ques_limit, ans_limit,
                 char_limit, hidden, char_dim, word_dim, num_words, use_pointer, attention_type):
        BiDAFModel.__init__(self, context, context_mask, context_char, question, question_mask, ques_char,
                            y1, y2, word_mat, char_mat, dropout, batch_size, para_limit, ques_limit, ans_limit,
                            char_limit, hidden, char_dim, word_dim)

        self.a = answer
        self.a_mask = answer_mask
        self.NV = num_words
        self.NVP = num_words + self.PL
        self.loop_function = self._loop_function_nopointer if not use_pointer else self._loop_function
        self.use_pointer = use_pointer
        self.attention_function = multihead_attention if attention_type == "dot" else vanilla_attention

    def build_model(self, global_step):
        # word, character embedding
        c_emb, q_emb = self.input_embedding()
        # input_encoder
        c, q = self.input_encoder(c_emb, q_emb)
        # bidaf_attention
        attention_outputs = self.optimized_bidaf_attention(c, q)
        # model_encoder
        self.enc = self.model_encoder(attention_outputs)
        # answer generator
        outputs, oups, attn_ws, p_gens = self.decode(self.a)
        # compute loss
        batch_loss = self._compute_loss(outputs, oups, attn_ws, p_gens, global_step, use_pointer=self.use_pointer)
        loss = tf.reduce_mean(batch_loss)
        return loss, batch_loss

    def model_encoder(self, attention_outputs, num_layers=3):
        with tf.variable_scope("Model_Encoder_Layer"):
            p0 = tf.reshape(tf.concat(attention_outputs, axis=-1), [self.N, self.PL, -1])
            (g0h_fw, g0h_bw), _ = tf.nn.bidirectional_dynamic_rnn(self.cells[2], self.cells[3], p0,
                                                                  sequence_length=self.c_len,
                                                                  dtype='float', scope="g0")
            g0 = tf.concat([g0h_fw, g0h_bw], axis=-1)
            return g0

    def decode(self, a, reuse=None):
        with tf.variable_scope("Decoder_Layer", reuse=reuse):
            memory = self.enc
            oups = tf.split(a, [1] * self.AL, 1)
            h = tf.tanh(_linear(tf.reduce_mean(memory, axis=1), output_size=self.d, bias=False, scope="h_initial"))
            c = tf.tanh(_linear(tf.reduce_mean(memory, axis=1), output_size=self.d, bias=False, scope="c_initial"))
            state = (c, h)
            attn_ws = []
            p_gens = []
            outputs = []
            for i, inp in enumerate(oups):
                einp = tf.reshape(tf.nn.embedding_lookup(self.word_mat, inp), [self.N, self.dw])
                if i > 0:
                    tf.get_variable_scope().reuse_variables()

                attn, attn_w = self.attention_function(tf.expand_dims(h, 1), units=self.d, num_heads=1, memory=memory,
                                                   mask=self.c_mask, bias=False, return_weights=True)

                attn_w = tf.reshape(attn_w, [-1, self.PL])
                attn_ws.append(attn_w)
                # update cell state
                attn = tf.reshape(attn, [-1, self.d])
                cinp = tf.concat([einp, attn], 1)
                h, state = self.cells[4](cinp, state)

                with tf.variable_scope("AttnOutputProjection"):
                    # generation prob
                    p_gen = tf.sigmoid(_linear([h] + [cinp], output_size=1, bias=True, scope="gen_prob"))
                    p_gens.append(p_gen)
                    # generation
                    output = _linear([h] + [cinp], output_size=self.dw * 2, bias=False, scope="output")
                    output = tf.reshape(output, [-1, self.dw, 2])
                    output = tf.reduce_max(output, 2)  # maxout
                    outputs.append(output)

            return outputs, oups, attn_ws, p_gens

    def sample(self, beam_size):
        with tf.variable_scope("Decoder_Layer", reuse=True):
            memory = self.enc
            oups = tf.split(self.a, [1] * self.AL, 1)
            h = tf.tanh(_linear(tf.reduce_mean(memory, axis=1), output_size=self.d, bias=False, scope="h_initial"))
            c = tf.tanh(_linear(tf.reduce_mean(memory, axis=1), output_size=self.d, bias=False, scope="c_initial"))
            state = (c, h)
            prev, attn_w, p_gen = None, None, None
            prev_probs = tf.zeros((self.N, 1))
            symbols = []
            attn_ws = []
            p_gens = []
            for i, inp in enumerate(oups):
                einp = tf.nn.embedding_lookup(self.word_mat, inp)
                if prev is not None:
                    with tf.variable_scope("loop_function", reuse=True):
                        einp, prev_probs, index, prev_symbol = self.loop_function(beam_size, prev, attn_w, p_gen,
                                                                                  prev_probs, i)
                        h = tf.gather_nd(h, index)  # update prev state
                        state = tuple(tf.gather_nd(s, index) for s in state)  # update prev state
                        for j, symbol in enumerate(symbols):
                            symbols[j] = tf.gather_nd(symbol, index)  # update prev symbols
                        for j, attn_w in enumerate(attn_ws):
                            attn_ws[j] = tf.gather_nd(attn_w, index)  # update prev attn_ws
                        for j, p_gen in enumerate(p_gens):
                            p_gens[j] = tf.gather_nd(p_gen, index)  # update prev p_gens
                        symbols.append(prev_symbol)

                attn, attn_w = self.attention_function(tf.expand_dims(h, 1) if i == 0 else h, units=self.d, num_heads=1,
                                                   memory=memory, mask=self.c_mask, bias=False, return_weights=True)
                attn_w = tf.reshape(attn_w, [self.N, -1, self.PL])
                attn_ws.append(attn_w)

                # update cell state
                cinp = tf.concat([einp, attn], -1)
                h, state = self.cells[4](tf.reshape(cinp, [-1, self.dw + self.d]),
                                         tuple(tf.reshape(s, [-1, self.d]) for s in state))
                h = tf.reshape(h, [self.N, -1, self.d])
                state = tuple(tf.reshape(s, [self.N, -1, self.d]) for s in state)

                with tf.variable_scope("AttnOutputProjection"):
                    oinp = tf.reshape(tf.concat([h, cinp], -1), [-1, self.d * 2 + self.dw])
                    # generation prob
                    p_gen = tf.sigmoid(_linear([oinp], output_size=1, bias=True, scope="gen_prob"))
                    p_gen = tf.reshape(p_gen, [self.N, -1, 1])
                    p_gens.append(p_gen)
                    # generation
                    output = _linear([oinp], output_size=self.dw * 2, bias=False, scope="output")
                    output = tf.reshape(output, [-1, self.dw, 2])
                    output = tf.reduce_max(output, 2)  # maxout
                    output = tf.reshape(output, [self.N, -1, self.dw])

                prev = output

            # process the last symbol
            einp, prev_probs, index, prev_symbol = self.loop_function(beam_size, prev, attn_w, p_gen, prev_probs, i)
            for j, symbol in enumerate(symbols):
                symbols[j] = tf.gather_nd(symbol, index)  # update prev symbols
            symbols.append(prev_symbol)

            # output the final best result of beam search
            index = tf.stack([tf.range(self.N), tf.zeros(self.N, dtype=tf.int32)], axis=-1)
            for k, symbol in enumerate(symbols):
                symbols[k] = tf.gather_nd(symbol, index)

            return symbols

    def _loop_function_nopointer(self, beam_size, prev, attn_w, p_gen, prev_probs, i):
        dim = 1 if i == 1 else beam_size
        # scatter attention probs
        logit = tf.matmul(tf.reshape(prev, [-1, self.dw]), self.word_mat, transpose_b=True)
        logit = tf.reshape(logit, [self.N, dim, -1])
        dist_g = tf.nn.softmax(logit)
        final_dist = tf.log(dist_g)
        # beam search
        prev_probs = tf.expand_dims(prev_probs, -1)
        prev = final_dist + prev_probs  # batch_size * dim * NVP
        prev = tf.reshape(prev, [self.N, -1])  # batch_size * (dim * NVP)
        probs, prev_symbolb = tf.nn.top_k(prev, beam_size)  # batch_size * beam_size
        index = prev_symbolb // self.NV
        bindex = tf.tile(tf.expand_dims(tf.range(self.N), -1), [1, beam_size])
        index = tf.stack((bindex, index), axis=2)
        prev_symbol = prev_symbolb % self.NV

        # embedding_lookup
        emb_prev = tf.nn.embedding_lookup(self.word_mat, prev_symbol)

        return emb_prev, probs, index, prev_symbol

    def _loop_function(self, beam_size, prev, attn_w, p_gen, prev_probs, i):
        dim = 1 if i == 1 else beam_size
        # scatter attention probs
        bc = tf.tile(tf.expand_dims(self.c, 1), [1, dim, 1])  # batch_size * beam_size * PL
        batch_nums_c = tf.tile(tf.reshape(tf.range(self.N), [self.N, 1, 1]), [1, dim, self.PL])
        beam_size_c = tf.tile(tf.reshape(tf.range(dim), [1, dim, 1]), [self.N, 1, self.PL])
        indices_c = tf.stack((batch_nums_c, beam_size_c, bc), axis=3)
        dist_c = tf.scatter_nd(indices_c, attn_w, [self.N, dim, self.NVP])
        # combine generation probs and copy probs
        logit = tf.matmul(tf.reshape(prev, [-1, self.dw]), self.word_mat, transpose_b=True)
        logit = tf.reshape(logit, [self.N, dim, -1])
        plus_logit = tf.zeros([self.N, dim, self.PL]) - 1e30
        logit = tf.concat([logit, plus_logit], axis=-1)
        dist_g = tf.nn.softmax(logit)
        final_dist = tf.log(p_gen * dist_g + (1 - p_gen) * dist_c)
        # beam search
        prev_probs = tf.expand_dims(prev_probs, -1)
        prev = final_dist + prev_probs  # batch_size * dim * NVP
        prev = tf.reshape(prev, [self.N, -1])  # batch_size * (dim * NVP)
        probs, prev_symbolb = tf.nn.top_k(prev, beam_size)  # batch_size * beam_size
        index = prev_symbolb // self.NVP
        bindex = tf.tile(tf.expand_dims(tf.range(self.N), -1), [1, beam_size])
        index = tf.stack((bindex, index), axis=2)
        prev_symbol = prev_symbolb % self.NVP

        # embedding_lookup
        plus_word_mat = tf.tile(tf.nn.embedding_lookup(self.word_mat, [1]), [self.PL, 1])
        emb_prev = tf.nn.embedding_lookup(tf.concat([self.word_mat, plus_word_mat], axis=0), prev_symbol)
        # emb_prev = tf.nn.embedding_lookup(self.word_mat, prev_symbol)

        return emb_prev, probs, index, prev_symbol

    def _compute_loss(self, ouputs, oups, attn_ws, p_gens, global_step, use_pointer):
        batch_nums_c = tf.tile(tf.expand_dims(tf.range(self.N), 1), [1, self.PL])
        indices_c = tf.stack((batch_nums_c, self.c), axis=2)
        batch_nums = tf.expand_dims(tf.range(self.N), 1)
        weights = []
        crossents = []
        for output, oup, attn_w, p_gen in zip(ouputs[:-1], oups[1:], attn_ws[:-1], p_gens[:-1]):
            # combine copy and generation probs
            logit = tf.matmul(output, self.word_mat, transpose_b=True)
            target = tf.reshape(oup, [-1])
            if use_pointer:
                dist_c = tf.scatter_nd(indices_c, attn_w, [self.N, self.NVP])
                plus_logit = tf.zeros([self.N, self.PL]) - 1e30
                logit = tf.concat([logit, plus_logit], axis=-1)
                dist_g = tf.nn.softmax(logit)
                final_dist = p_gen * dist_g + (1 - p_gen) * dist_c
                # get loss
                indices = tf.concat((batch_nums, oup), axis=1)
                gold_probs = tf.gather_nd(final_dist, indices)
                crossent = tf.cond(global_step < 10000,
                                   lambda: tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=target),
                                   lambda: -tf.log(tf.clip_by_value(gold_probs, 1e-10, 1.0)))
                weight_add = tf.cast(target < self.NV, tf.float32)
                weight = tf.cond(global_step < 10000,
                                 lambda: tf.cast(tf.cast(target, tf.bool), tf.float32) * weight_add,
                                 lambda: tf.cast(tf.cast(target, tf.bool), tf.float32))
            else:
                crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=target)
                weight = tf.cast(tf.cast(target, tf.bool), tf.float32)
            weights.append(weight)
            crossents.append(crossent * weight)
        log_perps = tf.add_n(crossents) / (tf.add_n(weights) + 1e-12)
        return log_perps


class BiDAFRLGenerator(BiDAFGenerator):
    def __init__(self, context, context_mask, context_char, question, question_mask, ques_char, answer, answer_mask,
                 ans_char, y1, y2, word_mat, char_mat, dropout, batch_size, para_limit, ques_limit, ans_limit,
                 char_limit, hidden, char_dim, word_dim, num_words, use_pointer, attention_type,
                 reward, sampled_answer, mixing_ratio, pre_step):
        BiDAFGenerator.__init__(self, context, context_mask, context_char, question, question_mask, ques_char, answer,
                                answer_mask, ans_char, y1, y2, word_mat, char_mat, dropout, batch_size, para_limit,
                                ques_limit, ans_limit, char_limit, hidden, char_dim, word_dim, num_words,
                                use_pointer, attention_type)
        self.reward = reward
        self.sa = sampled_answer
        self.lamda = mixing_ratio
        self.pre_step = pre_step

    def build_model(self, global_step):
        # word, character embedding
        c_emb, q_emb = self.input_embedding()
        # input_encoder
        c, q = self.input_encoder(c_emb, q_emb)
        # bidaf_attention
        attention_outputs = self.optimized_bidaf_attention(c, q)
        # model_encoder
        self.enc = self.model_encoder(attention_outputs)
        # compute loss
        # ml
        outputs, oups, attn_ws, p_gens = self.decode(self.a)
        batch_loss_ml = self._compute_loss(outputs, oups, attn_ws, p_gens, global_step, use_pointer=self.use_pointer)
        loss_ml = tf.reduce_mean(batch_loss_ml)
        # rl
        outputs, oups, attn_ws, p_gens = self.decode(self.sa, reuse=True)
        batch_loss_rl = self._compute_loss(outputs, oups, attn_ws, p_gens, global_step, use_pointer=self.use_pointer)
        loss_rl = tf.reduce_mean(batch_loss_rl * self.reward)
        loss = tf.cond(global_step < self.pre_step, lambda: loss_ml, lambda: (1 - self.lamda) * loss_ml + self.lamda * loss_rl)
        return loss, loss_ml, loss_rl

    def sample_rl(self):
        with tf.variable_scope("Decoder_Layer", reuse=True):
            memory = self.enc
            oups = tf.split(self.a, [1] * self.AL, 1)
            h = tf.tanh(_linear(tf.reduce_mean(memory, axis=1), output_size=self.d, bias=False, scope="h_initial"))
            c = tf.tanh(_linear(tf.reduce_mean(memory, axis=1), output_size=self.d, bias=False, scope="c_initial"))
            state = (c, h)
            prev, attn_w, p_gen = None, None, None
            symbols = []
            attn_ws = []
            p_gens = []
            for i, inp in enumerate(oups):
                einp = tf.reshape(tf.nn.embedding_lookup(self.word_mat, inp), [self.N, self.dw])
                if prev is not None:
                    with tf.variable_scope("loop_function", reuse=True):
                        einp, prev_symbol = self._loop_function_rl(prev, attn_w, p_gen, i)
                        symbols.append(prev_symbol)

                attn, attn_w = self.attention_function(tf.expand_dims(h, 1), units=self.d, num_heads=1,
                                                   memory=memory, mask=self.c_mask, bias=False, return_weights=True)
                attn_w = tf.reshape(attn_w, [-1, self.PL])
                attn = tf.reshape(attn, [-1, self.d])
                attn_ws.append(attn_w)

                # update cell state
                cinp = tf.concat([einp, attn], -1)
                h, state = self.cells[4](cinp, state)

                with tf.variable_scope("AttnOutputProjection"):
                    # generation prob
                    p_gen = tf.sigmoid(_linear([h] + [cinp], output_size=1, bias=True, scope="gen_prob"))
                    p_gens.append(p_gen)
                    # generation
                    output = _linear([h] + [cinp], output_size=self.dw * 2, bias=False, scope="output")
                    output = tf.reshape(output, [-1, self.dw, 2])
                    output = tf.reduce_max(output, 2)  # maxout
                prev = output

            # process the last symbol
            einp, prev_symbol = self._loop_function_rl(prev, attn_w, p_gen, i)
            symbols.append(prev_symbol)

            return symbols

    def _loop_function_rl(self, prev, attn_w, p_gen, i):
        # scatter attention probs
        batch_nums_c = tf.tile(tf.expand_dims(tf.range(self.N), 1), [1, self.PL])
        indices_c = tf.stack((batch_nums_c, self.c), axis=2)
        dist_c = tf.scatter_nd(indices_c, attn_w, [self.N, self.NVP])

        # combined probs
        logit = tf.matmul(prev, self.word_mat, transpose_b=True)
        plus_logit = tf.zeros([self.N, self.PL]) - 1e30
        logit = tf.concat([logit, plus_logit], axis=-1)
        dist_g = tf.nn.softmax(logit)
        final_dist = p_gen * dist_g + (1 - p_gen) * dist_c

        # multinomial sample
        dist = tf.distributions.Categorical(probs=final_dist)
        prev_symbol = dist.sample()
        # emb_prev = tf.nn.embedding_lookup(self.word_mat, prev_symbol)
        plus_word_mat = tf.tile(tf.nn.embedding_lookup(self.word_mat, [1]), [self.PL, 1])
        emb_prev = tf.nn.embedding_lookup(tf.concat([self.word_mat, plus_word_mat], axis=0), prev_symbol)

        return emb_prev, prev_symbol