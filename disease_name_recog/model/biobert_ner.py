import tensorflow as tf
from transformers import BertConfig, TFBertModel

# configuration = BertConfig.from_json_file('biobert_v1.1_pubmed/bert_config.json')
# biobert_model = TFBertModel.from_pretrained("biobert_v1.1_pubmed/pytorch_model.bin", from_pt=True,
#                                             config=configuration)


configuration = BertConfig.from_json_file('disease_name_recog/biobert_v1.1_pubmed/bert_config.json')
biobert_model = TFBertModel.from_pretrained("disease_name_recog/biobert_v1.1_pubmed/pytorch_model.bin", from_pt=True,
                                            config=configuration)


def shape_list(tensor: tf.Tensor):
    """
    Deal with dynamic shape in tensorflow cleanly.
    Args:
        tensor (:obj:`tf.Tensor`): The tensor we want the shape of.
    Returns:
        :obj:`List[int]`: The shape of the tensor as a list.
    """
    dynamic = tf.shape(tensor)

    if tensor.shape == tf.TensorShape(None):
        return dynamic

    static = tensor.shape.as_list()

    return [dynamic[i] if s is None else s for i, s in enumerate(static)]


def sparse_categorical_accuracy_masked(y_true, y_pred):
    mask_value = 3
    active_loss = tf.reshape(y_true, (-1,)) != mask_value
    reduced_logits = tf.boolean_mask(tf.reshape(y_pred, (-1, shape_list(y_pred)[2])), active_loss)
    y_true = tf.boolean_mask(tf.reshape(y_true, (-1,)), active_loss)
    reduced_logits = tf.cast(tf.argmax(reduced_logits, axis=-1), tf.keras.backend.floatx())
    equality = tf.equal(y_true, reduced_logits)
    return tf.reduce_mean(tf.cast(equality, tf.keras.backend.floatx()))


def sparse_crossentropy_masked(y_true, y_pred):
    mask_value = 3
    y_true_masked = tf.boolean_mask(y_true, tf.not_equal(y_true, mask_value))
    y_pred_masked = tf.boolean_mask(y_pred, tf.not_equal(y_true, mask_value))
    return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y_true_masked, y_pred_masked))


def create_model(num_tags, max_length):
    input_ids = tf.keras.layers.Input(shape=(max_length,), name='input_ids', dtype=tf.int32)
    attention_mask = tf.keras.layers.Input(shape=(max_length,), name='attention_mask', dtype=tf.int32)
    embedding = biobert_model(input_ids, attention_mask=attention_mask)[0]
    embedding = tf.keras.layers.Dropout(0.3)(embedding)
    output = tf.keras.layers.Dense(num_tags, activation='softmax')(embedding)

    model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output)
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
    model.compile(optimizer=optimizer, loss=sparse_crossentropy_masked, metrics=[sparse_categorical_accuracy_masked])
    return model