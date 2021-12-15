import numpy as np
import tensorflow as tf
from tensorflow_addons.text import crf

class MaskedCRF(object):
    def __init__(self, num_output, use_mask, label2idx_map):
        self.num_output = num_output
        self.label2idx_map = label2idx_map

        self.mask_tran_matrix = None
        if use_mask:
            self.mask_tran_matrix = self.get_mask_trans()

    def decode(self, logits, label_ids, lengths):
        
        initializer = tf.random_uniform_initializer(minval=-0.1, maxval=0.1, seed=None)
        trans = tf.Variable(
            name="crf_transitions",
            initial_value=initializer(shape=[self.num_output, self.num_output])
        )

        if self.mask_tran_matrix is not None:
            trans = tf.minimum(trans, self.mask_tran_matrix)

        log_probs, trans = crf.crf_log_likelihood(
            inputs=logits,
            tag_indices=label_ids,
            transition_params=trans,
            sequence_lengths=lengths
        )
        per_example_loss = -log_probs
        loss = tf.reduce_mean(per_example_loss)
        label_pred, score = crf.crf_decode(potentials=logits, transition_params=trans, sequence_length=lengths)

        return loss, per_example_loss, label_pred

    def get_mask_trans(self):
        size = len(self.label2idx_map)
        tag_lst = self.label2idx_map.keys()

        mask_mat = np.ones(shape=(size, size), dtype=np.float32)
        # analysis tag schema，BIO or BIOES
        is_scheme_bioes = False
        flag_e = False
        flag_s = False
        for tag in tag_lst:
            if tag.startswith("E-"):
                flag_e = True

            if tag.startswith("S-"):
                flag_s = True

        if flag_e and flag_s:
            is_scheme_bioes = True

        for col_tag, col_index in self.label2idx_map.items():
            if col_tag.startswith("I-"):
                slot_name = col_tag.replace("I-", "")
                begin_slot = "B-" + slot_name
                for row_tag, row_index in self.label2idx_map.items():
                    # I-city must follow B-city or I-city
                    if row_tag != begin_slot and row_tag != col_tag:
                        mask_mat[row_index, col_index] = -1.0

            if is_scheme_bioes:
                if col_tag.startswith("E-"):
                    slot_name = col_tag.replace("E-", "")
                    intermediate_slot = "I-" + slot_name
                    begin_slot = "B-" + slot_name
                    for row_tag, row_index in self.label2idx_map.items():
                        # E-city must follow I-city or B-city
                        if row_tag != intermediate_slot and row_tag != begin_slot:
                            mask_mat[row_index, col_index] = -1.0

                if col_tag.startswith("S-") or col_tag.startswith("B-"):
                    for row_tag, row_index in self.label2idx_map.items():
                        # S-city must not follow B-slot or I-slot
                        if row_tag.startswith("B-") or row_tag.startswith("I-"):
                            mask_mat[row_index, col_index] = -1.0

        return 100 * mask_mat
