import keras
import keras_preprocessing.image
import keras_transformer.position
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd


def images_seq(image):
    return keras.Sequential([
        keras.layers.Lambda(keras.backend.cast, arguments=dict(dtype='float32'), output_shape=lambda x: x),
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
    ], name='images')(image)


class AttentionCell(keras.layers.Layer):
    def __init__(self, cell, seq, vocabulary_size, **kwargs):
        self.cell = cell
        self.state_size = list(cell.state_size) + [512]
        self.seq = seq
        self.W1_e = keras.layers.Dense(512, use_bias=False, name='W1_e')(seq)
        self.W2_h = keras.layers.Dense(512, use_bias=False, name='W2_h')
        self.beta = keras.layers.Dense(1, use_bias=False, name='beta')
        self.W3_o = keras.layers.Dense(512, use_bias=False, activation='tanh', name='W3_o')
        self.W4 = keras.layers.Dense(vocabulary_size, use_bias=False, name='W4')
        self.output_size = vocabulary_size
        super(AttentionCell, self).__init__(**kwargs)

    def build(self, input_shape):
        self.cell.build((input_shape[0], input_shape[1] + 512))

    def get_config(self):
        return dict(cell=self.cell, seq=self.seq, vocabulary_size=self.output_size)

    def call(self, inputs, attention_state):
        """
        Args:
            inputs: shape = (batch_size, dim_embeddings) embeddings from previous time step
            attention_state: (AttentionState) state from previous time step
        """
        *state, o = attention_state
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
        return logits, [*new_state, new_o]


class TrainingSequence(keras.utils.Sequence):
    def __init__(self, image_path_and_sequence, token_encoder):
        self.data_frame_iterator = keras_preprocessing.image.dataframe_iterator.DataFrameIterator(
            pd.DataFrame(
                [
                    [image_path, list(token_encoder.transform(['SOS'] + sequence))]
                    for image_path, sequence in image_path_and_sequence
                ],
                columns=('filename', 'class'),
            ),
            image_data_generator=ImageDataGenerator(
                rescale=1. / 255,
            ),
            class_mode='raw',
            dtype='uint8',
            color_mode='grayscale',
        )
        self.max_sequence_length = max(len(sequence) for image_path, sequence in image_path_and_sequence)
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
            padded[:, 1:],
        )


class Model:
    def __init__(self, vocab_size):
        images_input = keras.layers.Input(shape=(256, 256, 1), dtype='uint8')
        seq = images_seq(images_input)
        self.seq_mean = keras.layers.Lambda(
            keras.backend.mean,
            arguments={'axis': 1},
            output_shape=lambda x: (x[0], x[2]),
        )(seq)
        # 1. get token embeddings
        dim = 80
        sequences_input = keras.Input((None,), dtype='int32')
        tok_embeddings = keras.layers.Embedding(vocab_size, dim, dtype='float32', name='embedding')(sequences_input)

        # 2. add the special <sos> token embedding at the beggining of every formula

        # 3. decode
        state_inputs = [
            keras.layers.Input((512,)),
            keras.layers.Input((512,)),
            keras.layers.Input((512,)),
        ]
        attention_cell = AttentionCell(keras.layers.LSTMCell(512, name='lstm'), seq, vocab_size)
        initial_state = [
            keras.layers.Dense(512, activation='tanh', name='h0')(self.seq_mean),
            keras.layers.Dense(512, activation='tanh', name='c0')(self.seq_mean),
            keras.layers.Dense(512, activation='tanh', name='o0')(self.seq_mean),
        ]
        self.training = keras.models.Model(
            [images_input, sequences_input],
            keras.layers.RNN(attention_cell, return_sequences=True)(tok_embeddings, initial_state=initial_state),
        )
        self.evaluation0 = keras.models.Model(
            [images_input, sequences_input],
            keras.layers.RNN(attention_cell, return_state=True)(tok_embeddings, initial_state=initial_state),
        )
        self.evaluation = keras.models.Model(
            [images_input, sequences_input] + state_inputs,
            keras.layers.RNN(attention_cell, return_state=True)(tok_embeddings, initial_state=state_inputs),
        )

    def save_weights(self, path):
        self.training.save_weights(path)

    def load_weights(self, path):
        self.training.load_weights(path, by_name=True)
        self.evaluation0.load_weights(path, by_name=True)
        self.evaluation.load_weights(path, by_name=True)

    def predict(self, images):
        sequences = np.ones((len(images), 1))
        done = np.full(len(images), False)
        logits, *states = self.evaluation0.predict([images, sequences])
        while True:
            classes = np.argmax(logits, axis=-1)
            sequences = np.concatenate([sequences, classes.reshape(-1, 1)], axis=1)
            done |= classes == 0
            if np.all(done):
                return sequences
            logits, *states = self.evaluation.predict([images, sequences, *states])
