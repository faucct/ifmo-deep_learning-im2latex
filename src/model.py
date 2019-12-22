import keras
import keras_preprocessing.image
import keras_transformer.position
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd


def images_seq(image):
    return keras.Sequential([
        keras.layers.Conv2D(64, 3, strides=1, padding='SAME', activation='relu'),
        keras.layers.MaxPooling2D(2, 2, 'SAME'),
        keras.layers.Conv2D(128, 3, strides=1, padding='SAME', activation='relu'),
        keras.layers.MaxPooling2D(2, 2, 'SAME'),
        keras.layers.Conv2D(256, 3, strides=1, padding='SAME', activation='relu'),
        keras.layers.Conv2D(256, 3, strides=1, padding='SAME', activation='relu'),
        keras.layers.MaxPooling2D((2, 1), (2, 1), 'SAME'),
        keras.layers.Conv2D(512, 3, strides=1, padding='SAME', activation='relu'),
        keras.layers.MaxPooling2D((1, 2), (1, 2), 'SAME'),
        keras.layers.Conv2D(512, 3, strides=1, padding='VALID', activation='relu'),
        keras_transformer.position.AddPositionalEncoding(),
        keras.layers.Reshape((-1, 512)),
    ])(image)


class AttentionCell(keras.layers.Layer):
    def __init__(self, cell, seq, vocabulary_size, **kwargs):
        self.cell = cell
        self.state_size = list(cell.state_size)
        self.seq = seq
        self.W1_e = keras.layers.Dense(512, use_bias=False)(seq)
        self.W2_h = keras.layers.Dense(512, use_bias=False)
        self.beta = keras.layers.Dense(1, use_bias=False)
        self.W3_o = keras.layers.Dense(512, use_bias=False, activation='tanh')
        self.W4 = keras.layers.Dense(vocabulary_size, use_bias=False)
        self.vocabulary_size = vocabulary_size
        super(AttentionCell, self).__init__(**kwargs)

    def build(self, input_shape):
        self.cell.build(input_shape)

    def call(self, inputs, attention_state):
        """
        Args:
            inputs: shape = (batch_size, dim_embeddings) embeddings from previous time step
            attention_state: (AttentionState) state from previous time step
        """
        state, o = attention_state
        # compute h
        h, new_state = self.cell.call(keras.layers.Concatenate(axis=-1)([inputs, o]), state)
        # apply previous logic
        c = keras.layers.Lambda(
            keras.backend.sum,
            arguments=dict(axis=1),
            output_shape=lambda shape: (shape[0], shape[2]),
        )(keras.layers.multiply([
            keras.layers.Activation('softmax')(self.beta(keras.layers.Activation('tanh')(keras.layers.add([
                self.W1_e,
                self.W2_h(h),
            ])))),
            self.seq,
        ]))
        new_o = self.W3_o(keras.layers.Concatenate(axis=-1)([h, c]))
        logits = self.W4(new_o)
        return logits, [new_state, new_o]


class TrainingSequence(keras.utils.Sequence):
    def __init__(self, image_id_and_sequence, token_encoder):
        self.data_frame_iterator = keras_preprocessing.image.dataframe_iterator.DataFrameIterator(
            pd.DataFrame(
                [
                    [image_id, list(token_encoder.transform(['SOS'] + sequence))]
                    for image_id, sequence in image_id_and_sequence
                ],
                columns=('filename', 'class'),
            ),
            directory='tmp/formula_images/',
            image_data_generator=ImageDataGenerator(
                rescale=1. / 255,
            ),
            class_mode='raw',
            dtype='uint8',
            color_mode='grayscale',
        )
        self.max_sequence_length = max(len(sequence) for image_id, sequence in image_id_and_sequence)
        self.token_encoder = token_encoder

    def __len__(self):
        return len(self.data_frame_iterator)

    def __getitem__(self, index):
        images, sequences = self.data_frame_iterator[index]
        padded = keras.preprocessing.sequence.pad_sequences(
            sequences,
            padding='post',
            maxlen=self.max_sequence_length + 1,
        )
        return (
            [images, padded[:, :-1]],
            keras.utils.to_categorical(padded[:, 1:], len(self.token_encoder.classes_)),
        )
