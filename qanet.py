import tensorflow as tf
from layers import initializer, regularizer, residual_block, highway, conv, mask_logits, trilinear, total_params, \
    optimized_trilinear_for_attention, _linear, multihead_attention


class QANetModel(object):
    def __init__(self, context, context_mask, context_char, question, question_mask, ques_char,
                 y1, y2, word_mat, char_mat, dropout, batch_size, para_limit, ques_limit,
                 char_limit, hidden, char_dim, word_dim, num_head, model_encoder_layers,
                 model_encoder_blocks, model_encoder_convs, input_encoder_convs):
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

        self.model_encoder_layers = model_encoder_layers
        self.model_encoder_blocks = model_encoder_blocks
        self.model_encoder_convs = model_encoder_convs
        self.input_encoder_convs = input_encoder_convs


    def build_model(self, global_step):
        # word, character embedding
        c_emb, q_emb = self.input_embedding()
        # input_encoder
        c, q = self.input_encoder(c_emb, q_emb, num_conv_layers=self.input_encoder_convs)
        # bidaf_attention
        attention_outputs = self.optimized_bidaf_attention(c, q)
        # model_encoder
        self.model_encoder(attention_outputs, num_layers=self.model_encoder_layers,
                           num_conv_layers=self.model_encoder_convs, num_blocks=self.model_encoder_blocks)
        # span start and end prediction
        self.output()
        # compute loss
        losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits1, labels=self.y1)
        losses2 = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits2, labels=self.y2)
        batch_loss = losses + losses2
        return tf.reduce_mean(batch_loss), batch_loss

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

    def input_encoder(self, c_emb, q_emb, num_blocks=1, num_conv_layers=4, kernel_size=7, reuse=True):
        with tf.variable_scope("Input_Encoder_Layer"):
            c = residual_block(c_emb,
                               num_blocks=num_blocks,
                               num_conv_layers=num_conv_layers,
                               kernel_size=kernel_size,
                               mask=self.c_mask,
                               num_filters=self.d,
                               num_heads=self.nh,
                               scope="Input_Encoder_Block",
                               bias=False,
                               dropout=self.dropout,
                               input_projection=True)
            q = residual_block(q_emb,
                               num_blocks=num_blocks,
                               num_conv_layers=num_conv_layers,
                               kernel_size=kernel_size,
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

    def model_encoder(self, attention_outputs, num_layers=3, num_blocks=7, num_conv_layers=2, kernel_size=5):
        with tf.variable_scope("Model_Encoder_Layer"):
            inputs = tf.concat(attention_outputs, axis=-1)
            self.enc = [conv(inputs, self.d, name="input_projection")]
            for i in range(num_layers):
                if i % 2 == 0:  # dropout every 2 blocks
                    self.enc[i] = tf.nn.dropout(self.enc[i], 1.0 - self.dropout)
                self.enc.append(residual_block(self.enc[i],
                                               num_blocks=num_blocks,
                                               num_conv_layers=num_conv_layers,
                                               kernel_size=kernel_size,
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


class QANetGenerator(QANetModel):
    def __init__(self, context, context_mask, context_char, question, question_mask, ques_char,
                 answer, answer_mask, ans_char, y1, y2, word_mat, char_mat, num_words, dropout, batch_size,
                 para_limit, ques_limit, ans_limit, char_limit, hidden, char_dim, word_dim, num_head,
                 model_encoder_layers, model_encoder_blocks, model_encoder_convs, input_encoder_convs):
        QANetModel.__init__(self, context, context_mask, context_char, question, question_mask, ques_char,
                            y1, y2, word_mat, char_mat, dropout, batch_size, para_limit, ques_limit, char_limit, hidden,
                            char_dim, word_dim, num_head, model_encoder_layers, model_encoder_blocks,
                            model_encoder_convs, input_encoder_convs)
        self.a = answer
        self.a_mask = answer_mask
        self.ah = ans_char
        self.cell = tf.nn.rnn_cell.LSTMCell(hidden)
        self.AL = ans_limit
        self.NV = num_words
        self.NVP = self.NV + self.PL

    def build_model(self, global_step):
        # word, character embedding
        c_emb, q_emb = self.input_embedding()
        # input_encoder
        c, q = self.input_encoder(c_emb, q_emb, num_conv_layers=self.input_encoder_convs)
        # bidaf_attention
        attention_outputs = self.optimized_bidaf_attention(c, q)
        # model_encoder
        self.model_encoder(attention_outputs, num_layers=self.model_encoder_layers,
                           num_conv_layers=self.model_encoder_convs, num_blocks=self.model_encoder_blocks)
        # answer generator
        outputs, oups, attn_ws, p_gens = self.decode(self.a)
        # compute loss
        batch_loss = self._compute_loss(outputs, oups, attn_ws, p_gens, global_step)
        loss = tf.reduce_mean(batch_loss)
        return loss

    def decode(self, a, reuse=None):
        with tf.variable_scope("Decoder_Layer", reuse=reuse):
            memory = tf.concat(self.enc[1:], axis=-1)
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

                attn, attn_w = multihead_attention(tf.expand_dims(h, 1), units=self.d, num_heads=1, memory=memory,
                                                   mask=self.c_mask, bias=False, return_weights=True)

                attn_w = tf.reshape(attn_w, [-1, self.PL])
                attn_ws.append(attn_w)
                # update cell state
                attn = tf.reshape(attn, [-1, self.d])
                cinp = tf.concat([einp, attn], 1)
                h, state = self.cell(cinp, state)

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
            memory = tf.concat(self.enc[1:], axis=-1)
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
                        einp, prev_probs, index, prev_symbol = self._loop_function(beam_size, prev, attn_w, p_gen,
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

                attn, attn_w = multihead_attention(tf.expand_dims(h, 1) if i == 0 else h, units=self.d, num_heads=1,
                                                   memory=memory, mask=self.c_mask, bias=False, return_weights=True)
                attn_w = tf.reshape(attn_w, [self.N, -1, self.PL])
                attn_ws.append(attn_w)

                # update cell state
                cinp = tf.concat([einp, attn], -1)
                h, state = self.cell(tf.reshape(cinp, [-1, self.dw + self.d]),
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
            einp, prev_probs, index, prev_symbol = self._loop_function(beam_size, prev, attn_w, p_gen, prev_probs, i)
            for j, symbol in enumerate(symbols):
                symbols[j] = tf.gather_nd(symbol, index)  # update prev symbols
            symbols.append(prev_symbol)

            # output the final best result of beam search
            index = tf.stack([tf.range(self.N), tf.zeros(self.N, dtype=tf.int32)], axis=-1)
            for k, symbol in enumerate(symbols):
                symbols[k] = tf.gather_nd(symbol, index)

            return symbols

    def _loop_function(self, beam_size, prev, attn_w, p_gen, prev_probs, i):
        dim = 1 if i == 1 else beam_size
        # scatter attention probs
        bc = tf.tile(tf.expand_dims(self.c, 1), [1, dim, 1])   # batch_size * beam_size * PL
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

    def _compute_loss(self, ouputs, oups, attn_ws, p_gens, global_step):
        batch_nums_c = tf.tile(tf.expand_dims(tf.range(self.N), 1), [1, self.PL])
        indices_c = tf.stack((batch_nums_c, self.c), axis=2)
        batch_nums = tf.expand_dims(tf.range(self.N), 1)
        weights = []
        crossents = []
        for output, oup, attn_w, p_gen in zip(ouputs[:-1], oups[1:], attn_ws[:-1], p_gens[:-1]):
            # combine copy and generation probs
            dist_c = tf.scatter_nd(indices_c, attn_w, [self.N, self.NVP])
            logit = tf.matmul(output, self.word_mat, transpose_b=True)
            plus_logit = tf.zeros([self.N, self.PL]) - 1e30
            logit = tf.concat([logit, plus_logit], axis=-1)
            dist_g = tf.nn.softmax(logit)
            final_dist = p_gen * dist_g + (1 - p_gen) * dist_c
            # get loss
            indices = tf.concat((batch_nums, oup), axis=1)
            gold_probs = tf.gather_nd(final_dist, indices)
            target = tf.reshape(oup, [-1])
            crossent = tf.cond(global_step < 10000,
                               lambda: tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=target),
                               lambda: -tf.log(tf.clip_by_value(gold_probs, 1e-10, 1.0)))
            # crossent = -tf.log(tf.clip_by_value(gold_probs, 1e-10, 1.0))
            weight_add = tf.cast(target < self.NV, tf.float32)
            weight = tf.cond(global_step < 10000,
                             lambda: tf.cast(tf.cast(target, tf.bool), tf.float32) * weight_add,
                             lambda: tf.cast(tf.cast(target, tf.bool), tf.float32))
            # weight = tf.cast(tf.cast(target, tf.bool), tf.float32)
            weights.append(weight)
            crossents.append(crossent * weight)
        log_perps = tf.add_n(crossents) / (tf.add_n(weights) + 1e-12)
        return log_perps


class QANetRLGenerator(QANetGenerator):
    def __init__(self, context, context_mask, context_char, question, question_mask, ques_char, answer, answer_mask,
                 ans_char, y1, y2, word_mat, char_mat, num_words, dropout, batch_size, para_limit, ques_limit, ans_limit,
                 char_limit, hidden, char_dim, word_dim, num_head, model_encoder_layers, model_encoder_blocks,
                 model_encoder_convs, input_encoder_convs, reward, sampled_answer, mixing_ratio, pre_step):
        QANetGenerator.__init__(self, context, context_mask, context_char, question, question_mask, ques_char,
                                answer, answer_mask, ans_char, y1, y2, word_mat, char_mat, num_words, dropout, batch_size,
                                para_limit, ques_limit, ans_limit, char_limit, hidden, char_dim, word_dim, num_head,
                                model_encoder_layers, model_encoder_blocks, model_encoder_convs, input_encoder_convs)
        self.reward = reward
        self.sa = sampled_answer
        self.lamda = mixing_ratio
        self.pre_step = pre_step

    def build_model(self, global_step):
        # word, character embedding
        c_emb, q_emb = self.input_embedding()
        # input_encoder
        c, q = self.input_encoder(c_emb, q_emb, num_conv_layers=self.input_encoder_convs)
        # bidaf_attention
        attention_outputs = self.optimized_bidaf_attention(c, q)
        # model_encoder
        self.model_encoder(attention_outputs, num_layers=self.model_encoder_layers,
                           num_conv_layers=self.model_encoder_convs, num_blocks=self.model_encoder_blocks)
        # compute loss
        # ml
        outputs, oups, attn_ws, p_gens = self.decode(self.a)
        loss_ml = tf.reduce_mean(self._compute_loss(outputs, oups, attn_ws, p_gens, global_step))
        # rl
        outputs, oups, attn_ws, p_gens = self.decode(self.sa, reuse=True)
        loss_rl = tf.reduce_mean(self._compute_loss(outputs, oups, attn_ws, p_gens, global_step) * self.reward)
        loss = tf.cond(global_step < self.pre_step, lambda: loss_ml, lambda: (1 - self.lamda) * loss_ml + self.lamda * loss_rl)
        return loss, loss_ml, loss_rl

    def sample_rl(self):
        with tf.variable_scope("Decoder_Layer", reuse=True):
            memory = tf.concat(self.enc[1:], axis=-1)
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

                attn, attn_w = multihead_attention(tf.expand_dims(h, 1), units=self.d, num_heads=1,
                                                   memory=memory, mask=self.c_mask, bias=False, return_weights=True)
                attn_w = tf.reshape(attn_w, [-1, self.PL])
                attn = tf.reshape(attn, [-1, self.d])
                attn_ws.append(attn_w)

                # update cell state
                cinp = tf.concat([einp, attn], -1)
                h, state = self.cell(cinp, state)

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