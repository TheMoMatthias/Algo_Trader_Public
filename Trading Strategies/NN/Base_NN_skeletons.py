import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization, LeakyReLU, Conv1D, MaxPooling1D, Flatten, Reshape, LayerNormalization,MultiHeadAttention, Add, Concatenate, RepeatVector
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers.legacy import Adam
# from tf_keras.optimizers.legacy import Adam
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import BinaryCrossentropy, SparseCategoricalCrossentropy, CategoricalCrossentropy
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.metrics import Precision, Recall, AUC
from tensorflow.keras.saving import register_keras_serializable
from sklearn.metrics import precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split, ParameterGrid
import numpy as np
from autokeras import StructuredDataRegressor, StructuredDataClassifier, ImageRegressor

@register_keras_serializable(name="SpectralNormalization")
class SpectralNormalization(tf.keras.layers.Layer):
    def __init__(self, layer, **kwargs):
        self.layer = layer
        super(SpectralNormalization, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.layer.build(input_shape)
        self.u = self.add_weight(shape=(1, self.layer.kernel.shape[-1]), initializer='random_normal', trainable=False, name='sn_u')
        super(SpectralNormalization, self).build(input_shape)
    
    def call(self, inputs, training=None):
        w = self.layer.kernel
        w_shape = w.shape.as_list()
        w = tf.reshape(w, [-1, w_shape[-1]])
        u_hat = self.u
        v_hat = tf.linalg.l2_normalize(tf.matmul(u_hat, w, transpose_b=True))
        u_hat = tf.linalg.l2_normalize(tf.matmul(v_hat, w))
        sigma = tf.matmul(tf.matmul(v_hat, w), u_hat, transpose_b=True)
        self.u.assign(u_hat)
        w_norm = w / sigma
        w_norm = tf.reshape(w_norm, w_shape)
        self.layer.kernel.assign(w_norm)
        return self.layer(inputs)


def spectral_norm(layer):
    return SpectralNormalization(layer)

@register_keras_serializable(name="PositionalEncoding")
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, seq_len, d_model, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.seq_len = seq_len
        self.d_model = d_model
        self.positional_encoding = self.create_positional_encoding(seq_len, d_model)

    def create_positional_encoding(self, seq_len, d_model):
        position = np.arange(seq_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pos_encoding = np.zeros((seq_len, d_model))
        pos_encoding[:, 0::2] = np.sin(position * div_term)
        pos_encoding[:, 1::2] = np.cos(position * div_term)
        return tf.constant(pos_encoding[np.newaxis, ...], dtype=tf.float32)

    def call(self, inputs):
        return inputs + self.positional_encoding[:, :tf.shape(inputs)[1], :]

    def get_config(self):
        config = super(PositionalEncoding, self).get_config()
        config.update({"seq_len": self.seq_len, "d_model": self.d_model})
        return config


@register_keras_serializable(name="LearnablePositionalEncoding")
class LearnablePositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, seq_len, d_model, **kwargs):
        super(LearnablePositionalEncoding, self).__init__(**kwargs)
        self.seq_len = seq_len
        self.d_model = d_model
        self.pos_embedding = self.add_weight(
            "pos_embedding", shape=[seq_len, d_model], initializer="random_normal", trainable=True)

    def call(self, inputs):
        return inputs + self.pos_embedding

    def get_config(self):
        config = super(LearnablePositionalEncoding, self).get_config()
        config.update({"seq_len": self.seq_len, "d_model": self.d_model})
        return config

# Feedforward layer typically used in transformer blocks
def feed_forward_network(d_model, d_ff):
    return tf.keras.Sequential([
        Dense(d_ff, activation='relu'),  # Intermediate layer (e.g., 2048 units)
        Dense(d_model)  # Back to d_model dimension
    ])

######################################################     GAN REGRSSION MODELS     ######################################################


class convolutional_lstm_gan_regression():
    def build_generator(input_shape, dropout_rate, kernel_regularizer):
        initializer = HeNormal()
        inputs = Input(shape=input_shape)

        # Convolutional layers for feature extraction
        x = Conv1D(64, kernel_size=3, padding='same', activation='relu', kernel_initializer=initializer)(inputs)
        x = Dropout(dropout_rate)(x)

        x = Conv1D(128, kernel_size=3, padding='same', activation='relu', kernel_initializer=initializer)(x)
        x = Dropout(dropout_rate)(x)

        x = Conv1D(256, kernel_size=3, padding='same', activation='relu', kernel_initializer=initializer)(x)
        x = Dropout(dropout_rate)(x)
        
        # LSTM layers for capturing temporal dependencies
        x = LSTM(128, return_sequences=True, kernel_regularizer=l2(kernel_regularizer))(x)
        x = Dropout(dropout_rate)(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = LSTM(128, return_sequences=True, kernel_regularizer=l2(kernel_regularizer))(x)
        x = Dropout(dropout_rate)(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = LSTM(128, return_sequences=False, kernel_regularizer=l2(kernel_regularizer))(x)
        x = Dropout(dropout_rate)(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = Dense(64, kernel_initializer=initializer, kernel_regularizer=l2(kernel_regularizer))(x)
        x = Dropout(dropout_rate)(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = Dense(1, kernel_initializer=initializer)(x)  # Output a single continuous value for regression
        generator = Model(inputs, x)
        return generator

    def build_discriminator(input_shape, dropout_rate, kernel_regularizer):
        initializer = HeNormal()
        inputs = Input(shape=input_shape)
        
        # Convolutional layers for feature extraction
        x = Conv1D(64, kernel_size=3, padding='same', activation='relu', kernel_initializer=initializer)(inputs)
        x = Dropout(dropout_rate)(x)
        
        x = Conv1D(64, kernel_size=3, padding='same', activation='relu', kernel_initializer=initializer)(x)
        x = Dropout(dropout_rate)(x)
        
        # LSTM layers for capturing temporal dependencies
        x = LSTM(32, return_sequences=True, kernel_regularizer=l2(kernel_regularizer))(x)
        x = Dropout(dropout_rate)(x)
        x = LeakyReLU(alpha=0.2)(x)
        
        x = LSTM(32, return_sequences=False, kernel_regularizer=l2(kernel_regularizer))(x)
        x = Dropout(dropout_rate)(x)
        x = LeakyReLU(alpha=0.2)(x)
        
        x = Dense(64, kernel_initializer=initializer, kernel_regularizer=l2(kernel_regularizer))(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(dropout_rate)(x)
        
        x = Dense(1, kernel_initializer=initializer)(x)  # Output a single continuous value for regression
        discriminator = Model(inputs, x)
        return discriminator

    def build_critic(input_shape, dropout_rate, kernel_regularizer):
        initializer = HeNormal()
        inputs = Input(shape=input_shape)

        # Convolutional layers for feature extraction
        x = Conv1D(64, kernel_size=3, padding='same', activation='relu', kernel_initializer=initializer)(inputs)
        x = Dropout(dropout_rate)(x)

        x = Conv1D(64, kernel_size=3, padding='same', activation='relu', kernel_initializer=initializer)(x)
        x = Dropout(dropout_rate)(x)

        # LSTM layers for capturing temporal dependencies
        x = LSTM(32, return_sequences=True, kernel_regularizer=l2(kernel_regularizer))(x)
        x = Dropout(dropout_rate)(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = LSTM(32, return_sequences=False, kernel_regularizer=l2(kernel_regularizer))(x)
        x = Dropout(dropout_rate)(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = Dense(64, kernel_initializer=initializer, kernel_regularizer=l2(kernel_regularizer))(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(dropout_rate)(x)

        x = Dense(1, kernel_initializer=initializer)(x)  # Output a single continuous value for regression
        critic = Model(inputs, x)
        return critic

class convolutional_lstm_gan_regression_with_noise():
    def build_generator(input_shape, noise_dim, dropout_rate, kernel_regularizer):
        initializer = HeNormal()
        inputs = Input(shape=input_shape)
        noise = Input(shape=(noise_dim,))  # Adding noise input
        
        # Concatenate the noise with the input data
        concatenated = Concatenate()([inputs, RepeatVector(input_shape[0])(noise)])

        # Convolutional layers for feature extraction
        x = Conv1D(32, kernel_size=3, padding='same', activation='relu', kernel_initializer=initializer)(concatenated)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)

        x = Conv1D(64, kernel_size=3, padding='same', activation='relu', kernel_initializer=initializer)(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)

        # LSTM layers for capturing temporal dependencies
        x = LSTM(64, return_sequences=True, kernel_regularizer=l2(kernel_regularizer))(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = LSTM(64, return_sequences=False, kernel_regularizer=l2(kernel_regularizer))(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = Dense(32, kernel_initializer=initializer, kernel_regularizer=l2(kernel_regularizer))(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = Dense(1, kernel_initializer=initializer)(x)  # Output a single continuous value for regression
        generator = Model([inputs, noise], x)
        return generator

    def build_discriminator(input_shape, dropout_rate, kernel_regularizer):
        initializer = HeNormal()
        inputs = Input(shape=input_shape)
        
        # LSTM layers for capturing temporal dependencies
        x = LSTM(32, return_sequences=True, kernel_regularizer=l2(kernel_regularizer))(inputs)
        x = Dropout(dropout_rate)(x)
        x = LeakyReLU(alpha=0.2)(x)
        
        x = LSTM(32, return_sequences=False, kernel_regularizer=l2(kernel_regularizer))(x)
        x = Dropout(dropout_rate)(x)
        x = LeakyReLU(alpha=0.2)(x)
        
        x = Dense(32, kernel_initializer=initializer, kernel_regularizer=l2(kernel_regularizer))(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(dropout_rate)(x)
        
        x = Dense(1, kernel_initializer=initializer)(x)  # Output a single continuous value for regression
        discriminator = Model(inputs, x)
        return discriminator
    
class lstm_gan_regression():
    def build_generator(input_shape,dropout_rate, kernel_regularizer):
        initializer = HeNormal()
        inputs = Input(shape=input_shape)
        x = LSTM(256, return_sequences=True, kernel_regularizer=l2(kernel_regularizer))(inputs)
        x = Dropout(dropout_rate)(x)
        x = LSTM(128, return_sequences=True, kernel_regularizer=l2(kernel_regularizer))(x)
        x = Dropout(dropout_rate)(x)
        x = LSTM(64, return_sequences=True, kernel_regularizer=l2(kernel_regularizer))(x)
        x = Dropout(dropout_rate)(x)
        x = Dense(1, kernel_initializer=initializer)(x)
        generator = Model(inputs, x)
        return generator

    def build_discriminator(input_shape, dropout_rate, kernel_regularizer):
        initializer = HeNormal()
        inputs = Input(shape=input_shape)
        x = LSTM(64, return_sequences=True, kernel_regularizer=l2(kernel_regularizer))(inputs)
        x = Dropout(dropout_rate)(x)
        x = LSTM(32, return_sequences=False, kernel_regularizer=l2(kernel_regularizer))(x)
        x = Dropout(dropout_rate)(x)
        x = Flatten()(x)
        x = Dense(32, kernel_initializer=initializer, kernel_regularizer=l2(kernel_regularizer))(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dense(1)  
        discriminator = Model(inputs, x)
        return discriminator

class convolutional_lstm_gan_classification():

    def build_generator(input_shape, dropout_rate, kernel_regularizer):
        initializer = HeNormal()
        inputs = Input(shape=input_shape)

        # LSTM layers for capturing temporal dependencies
        x = LSTM(64, return_sequences=True, kernel_regularizer=l2(kernel_regularizer))(inputs)
        x = Dropout(dropout_rate)(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = LSTM(64, return_sequences=True, kernel_regularizer=l2(kernel_regularizer))(x)
        x = Dropout(dropout_rate)(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = LSTM(32, return_sequences=True, kernel_regularizer=l2(kernel_regularizer))(x)
        x = Dropout(dropout_rate)(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = LSTM(32, return_sequences=True, kernel_regularizer=l2(kernel_regularizer))(x)
        x = Dropout(dropout_rate)(x)
        x = LeakyReLU(alpha=0.2)(x)

        # Final Dense layer for the output
        x = Dense(3, activation='softmax', kernel_initializer=initializer)(x)
        
        generator = Model(inputs, x)
        return generator
    
    def build_discriminator(input_shape, dropout_rate, kernel_regularizer):
        initializer = HeNormal()
        inputs = Input(shape=input_shape)
                
        # LSTM layers for capturing temporal dependencies
        x = LSTM(16, return_sequences=True, kernel_regularizer=l2(kernel_regularizer))(inputs)
        x = Dropout(dropout_rate)(x)
        x = LeakyReLU(alpha=0.2)(x)
        
        x = LSTM(16, return_sequences=True, kernel_regularizer=l2(kernel_regularizer))(x)
        x = Dropout(dropout_rate)(x)
        x = LeakyReLU(alpha=0.2)(x)
        
        x = LSTM(16, return_sequences=False, kernel_regularizer=l2(kernel_regularizer))(x)
        x = Dropout(dropout_rate)(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = Dense(1, kernel_initializer=initializer)(x)
        discriminator = Model(inputs, x)
        return discriminator
    
    def build_critic(input_shape, dropout_rate, kernel_regularizer):
        initializer = HeNormal()
        inputs = Input(shape=input_shape)

        # LSTM layers for capturing temporal dependencies
        x = LSTM(16, return_sequences=True, kernel_regularizer=l2(kernel_regularizer))(inputs)
        x = Dropout(dropout_rate)(x)
        x = LeakyReLU(alpha=0.2)(x)
        
        x = LSTM(16, return_sequences=True, kernel_regularizer=l2(kernel_regularizer))(x)
        x = Dropout(dropout_rate)(x)
        x = LeakyReLU(alpha=0.2)(x)
        
        x = LSTM(16, return_sequences=False, kernel_regularizer=l2(kernel_regularizer))(x)
        x = Dropout(dropout_rate)(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = Dense(1, kernel_initializer=initializer)(x)
        critic = Model(inputs, x)
        return critic

class convolutional_lstm_gan_classification_with_noise:
    


    def build_generator(input_shape, noise_dim, dropout_rate, kernel_regularizer):
        initializer = HeNormal()
        real_input = Input(shape=input_shape)
        noise_input = Input(shape=(noise_dim,))

        # Concatenate noise with real input data
        noise_expanded = tf.keras.layers.Reshape((1, noise_dim))(noise_input)
        noise_expanded = tf.keras.layers.Concatenate(axis=1)([noise_expanded] * input_shape[0])
        x = tf.concat([real_input, noise_expanded], axis=-1)

        # LSTM layers for capturing temporal dependencies
        x = LSTM(64, return_sequences=True, kernel_regularizer=l2(kernel_regularizer))(x)
        x = Dropout(dropout_rate)(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = LSTM(32, return_sequences=True, kernel_regularizer=l2(kernel_regularizer))(x)
        x = Dropout(dropout_rate)(x)
        x = LeakyReLU(alpha=0.2)(x)

        # x = LSTM(32, return_sequences=True, kernel_regularizer=l2(kernel_regularizer))(x)
        # x = Dropout(dropout_rate)(x)
        # x = LeakyReLU(alpha=0.2)(x)

        # x = LSTM(32, return_sequences=True, kernel_regularizer=l2(kernel_regularizer))(x)
        # x = Dropout(dropout_rate)(x)
        # x = LeakyReLU(alpha=0.2)(x)

        # Dense layers for the final output
        x = Dense(16, kernel_initializer=initializer, kernel_regularizer=l2(kernel_regularizer))(x)
        x = Dropout(dropout_rate)(x)
        x = LeakyReLU(alpha=0.2)(x)

        # Final output layer
        x = Dense(3, activation='softmax', kernel_initializer=initializer)(x)
        
        generator = Model([real_input, noise_input], x)
        return generator


    def build_critic(input_shape, dropout_rate, kernel_regularizer):
        initializer = HeNormal()
        inputs = Input(shape=input_shape)

        # # Convolutional layers for feature extraction
        # x = Conv1D(64, kernel_size=3, padding='same', activation='relu', kernel_initializer=initializer)(inputs)
        # x = Dropout(dropout_rate)(x)

        # x = Conv1D(64, kernel_size=3, padding='same', activation='relu', kernel_initializer=initializer)(x)
        # x = Dropout(dropout_rate)(x)

        # LSTM layers for capturing temporal dependencies
        x = LSTM(32, return_sequences=True, kernel_regularizer=l2(kernel_regularizer))(inputs)
        x = Dropout(dropout_rate)(x)
        x = LeakyReLU(alpha=0.2)(x)
        
        x = LSTM(32, return_sequences=True, kernel_regularizer=l2(kernel_regularizer))(x)
        x = Dropout(dropout_rate)(x)
        x = LeakyReLU(alpha=0.2)(x)
        
        # x = LSTM(32, return_sequences=True, kernel_regularizer=l2(kernel_regularizer))(x)
        # x = Dropout(dropout_rate)(x)
        # x = LeakyReLU(alpha=0.2)(x)

        # x = LSTM(32, return_sequences=False, kernel_regularizer=l2(kernel_regularizer))(x)
        # x = Dropout(dropout_rate)(x)
        # x = LeakyReLU(alpha=0.2)(x)

        x = Dense(16, kernel_initializer=initializer, kernel_regularizer=l2(kernel_regularizer))(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(dropout_rate)(x)

        x = Dense(1, kernel_initializer=initializer)(x)
        critic = Model(inputs, x)
        return critic

class lstm_gan_classification():
    def build_generator(input_shape,dropout_rate, kernel_regularizer):
        initializer = HeNormal()
        inputs = Input(shape=input_shape)
        x = LSTM(64, return_sequences=True, kernel_regularizer=l2(kernel_regularizer))(inputs)
        x = Dropout(dropout_rate)(x)
        x = LSTM(64, return_sequences=True, kernel_regularizer=l2(kernel_regularizer))(x)
        x = Dropout(dropout_rate)(x)
        x = LSTM(32, return_sequences=True, kernel_regularizer=l2(kernel_regularizer))(x)
        x = Dropout(dropout_rate)(x)
        x = Dense(3, activation='softmax', kernel_initializer=initializer)(x)
        generator = Model(inputs, x)
        return generator

    def build_discriminator(input_shape, dropout_rate, kernel_regularizer):
        initializer = HeNormal()
        inputs = Input(shape=input_shape)
        x = LSTM(16, return_sequences=True, kernel_regularizer=l2(kernel_regularizer))(inputs)
        x = Dropout(dropout_rate)(x)
        x = LSTM(16, return_sequences=False, kernel_regularizer=l2(kernel_regularizer))(x)
        x = Dropout(dropout_rate)(x)
        x = Flatten()(x)
        x = Dense(16, kernel_initializer=initializer, kernel_regularizer=l2(kernel_regularizer))(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dense(1, activation='sigmoid', kernel_initializer=initializer)(x)  # Output three logits
        discriminator = Model(inputs, x)
        return discriminator

######################################################     LSTM TRANSFORMER REGRSSION MODELS     ######################################################
#
#
#
################################################################################################################################################


class lstm_transformer():
    def build_lstm_transformer_model_pca(input_shape, lstm_units, dropout_rate, kernel_regularizer):
        inputs = Input(shape=input_shape)
        x = LSTM(lstm_units, return_sequences=True)(inputs)
        x = Dropout(dropout_rate)(x)
        x = LSTM(lstm_units, return_sequences=True)(x)
        x = Dropout(dropout_rate)(x)
        
        # Transformer part
        attn_output = MultiHeadAttention(num_heads=2, key_dim=2)(x, x)
        attn_output = Add()([x, attn_output])
        x = LayerNormalization(epsilon=1e-6)(attn_output)
        
        # Final output
        x = Dense(64, activation='relu')(x)
        x = Dropout(dropout_rate)(x)
        outputs = Dense(1)(x)  # Predicting the return as a single continuous value

        model = Model(inputs, outputs)
        return model
    
    # def build_lstm_transformer_model(input_shape, lstm_units, dropout_rate, kernel_regularizer):
    #     inputs = Input(shape=input_shape)
    #     x = LSTM(64, return_sequences=True, kernel_regularizer=l2(kernel_regularizer))(inputs)
    #     x = Dropout(dropout_rate)(x)
    #     x = LSTM(64, return_sequences=True, kernel_regularizer=l2(kernel_regularizer))(x)
    #     x = Dropout(dropout_rate)(x)
    #     x = LSTM(32, return_sequences=True, kernel_regularizer=l2(kernel_regularizer))(x)
    #     x = Dropout(dropout_rate)(x)
    #     x = LSTM(32, return_sequences=True, kernel_regularizer=l2(kernel_regularizer))(x)
    #     x = Dropout(dropout_rate)(x)
    #     x = LSTM(16, return_sequences=True, kernel_regularizer=l2(kernel_regularizer))(x)
    #     x = Dropout(dropout_rate)(x)
    #     x = LSTM(8, return_sequences=True, kernel_regularizer=l2(kernel_regularizer))(x)
    #     x = Dropout(dropout_rate)(x)
        
    #     # Transformer part
    #     attn_output = MultiHeadAttention(num_heads=2, key_dim=2)(x, x)
    #     attn_output = Add()([x, attn_output])
    #     x = LayerNormalization(epsilon=1e-6)(attn_output)
        
    #     # Final output
    #     x = Dense(64, activation='relu')(x)
    #     x = Dropout(dropout_rate)(x)
    #     outputs = Dense(1)(x)  # Predicting the return as a single continuous value

    #     model = Model(inputs, outputs)
    #     return model
    
    def build_lstm_transformer_model(input_shape, lstm_units, dropout_rate, num_classes):
        inputs = Input(shape=input_shape)
        
        # LSTM layers
        x = LSTM(lstm_units, return_sequences=True)(inputs)
        x = Dropout(dropout_rate)(x)
        
        # Add Positional Encoding
        positional_encoding_layer = PositionalEncoding(seq_len=input_shape[0], d_model=lstm_units)
        x = positional_encoding_layer(x)
        
        # Transformer part
        attn_output = MultiHeadAttention(num_heads=4, key_dim=4)(x, x)
        attn_output = Add()([x, attn_output])
        x = LayerNormalization(epsilon=1e-6)(attn_output)
        
        # Dense layers and classification head
        x = Dense(64, activation='relu')(x)
        x = Dropout(dropout_rate)(x)
        outputs = Dense(num_classes, activation='softmax')(x)  # Multi-class classification
        
        model = Model(inputs, outputs)
        
        # Compile the model for classification
        model.compile(optimizer=Adam(learning_rate=1e-4), 
                    loss=SparseCategoricalCrossentropy(from_logits=False), 
                    metrics=['accuracy'])
        
        return model
    
class lstm_autotransformer():
    def build_lstm_autoencoder(input_shape, lstm_units, dropout_rate, kernel_regularizer):
        inputs = Input(shape=input_shape)
        x = LSTM(64, return_sequences=True, kernel_regularizer=l2(kernel_regularizer))(inputs)
        x = Dropout(dropout_rate)(x)
        x = LSTM(64, return_sequences=True, kernel_regularizer=l2(kernel_regularizer))(x)
        x = Dropout(dropout_rate)(x)
        x = LSTM(64, return_sequences=True, kernel_regularizer=l2(kernel_regularizer))(x)
        x = Dropout(dropout_rate)(x)
        x = LSTM(32, return_sequences=True, kernel_regularizer=l2(kernel_regularizer))(x)
        x = Dropout(dropout_rate)(x)
        x = LSTM(32, return_sequences=True, kernel_regularizer=l2(kernel_regularizer))(x)
        x = Dropout(dropout_rate)(x)
        
        # Transformer part
        attn_output = MultiHeadAttention(num_heads=2, key_dim=2)(x, x)
        attn_output = Add()([x, attn_output])
        x = LayerNormalization(epsilon=1e-6)(attn_output)
        
        # Final output
        x = Dense(32, activation='relu')(x)
        x = Dropout(dropout_rate)(x)
        outputs = Dense(1)(x)  # Predicting the return as a single continuous value

        model = Model(inputs, outputs)
        return model
    
    
class lstm_autotransformer_classification():
    @staticmethod
    def build_lstm_autotransformer(input_shape, lstm_units, dropout_rate, kernel_regularizer):
        inputs = Input(shape=input_shape)
        encoded = LSTM(lstm_units, activation='relu', kernel_regularizer=l2(kernel_regularizer), return_sequences=True)(inputs)
        encoded = Dropout(dropout_rate)(encoded)
        encoded = LSTM(lstm_units // 2, activation='relu', kernel_regularizer=l2(kernel_regularizer), return_sequences=False)(encoded)
        encoded = Dropout(dropout_rate)(encoded)
        
        repeated = RepeatVector(input_shape[0])(encoded)
        decoded = LSTM(lstm_units // 2, activation='relu', kernel_regularizer=l2(kernel_regularizer), return_sequences=True)(repeated)
        decoded = Dropout(dropout_rate)(decoded)
        decoded = LSTM(lstm_units, activation='relu', kernel_regularizer=l2(kernel_regularizer), return_sequences=True)(decoded)
        decoded = Dropout(dropout_rate)(decoded)
        
        # Transformer part
        attn_output = MultiHeadAttention(num_heads=2, key_dim=2)(decoded, decoded)
        attn_output = Add()([decoded, attn_output])
        decoded = LayerNormalization(epsilon=1e-6)(attn_output)
        
        outputs = Dense(input_shape[-1], activation='linear')(decoded)
        
        model = Model(inputs, outputs)
        return model
    
    # LSTM + Transformer Model for Multi-Class Classification
    def build_lstm_transformer_model(input_shape, lstm_units, dropout_rate, num_classes):
        inputs = Input(shape=input_shape)

        # LSTM layers (for time dependency learning)
        x = LSTM(lstm_units, return_sequences=True)(inputs)
        x = Dropout(dropout_rate)(x)
        
        # Add Positional Encoding (to capture temporal ordering in Transformer)
        positional_encoding_layer = PositionalEncoding(seq_len=input_shape[0], d_model=lstm_units)
        x = positional_encoding_layer(x)

        # Transformer Block
        attn_output = MultiHeadAttention(num_heads=4, key_dim=4)(x, x)  # Increased heads/key_dim
        attn_output = Add()([x, attn_output])
        x = LayerNormalization(epsilon=1e-6)(attn_output)

        # Dense layers
        x = Dense(64, activation='relu')(x)
        x = Dropout(dropout_rate)(x)

        # Output layer for multi-class classification
        outputs = Dense(num_classes, activation='softmax')(x)

        # Compile the model for classification
        model = Model(inputs, outputs)
        
        
    # LSTM + Transformer Model for Multi-Class Classification
    def build_sophisticated_lstm_transformer_model(input_shape, dropout_rate_lstm, dropout_rate_dense, dropout_rate_multihead, 
                                                   regularization_rate, num_heads, size_heads, num_classes, d_ff=256):
        inputs = Input(shape=input_shape)
        last_lstm_units = 64
        d_ff = d_ff if d_ff > last_lstm_units*4 else last_lstm_units*4
        # Stack of LSTM layers (for deeper time dependency learning)
        x = LSTM(64, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(regularization_rate))(inputs)  # First LSTM layer
        x = Dropout(dropout_rate_lstm)(x)
        x = LSTM(64, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(regularization_rate))(x)  # Second LSTM layer
        x = Dropout(dropout_rate_lstm)(x)
        x = LSTM(64, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(regularization_rate))(x)  # Third LSTM layer (optional for deeper architectures)
        x = Dropout(dropout_rate_lstm)(x)
        # x = LSTM(64, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(regularization_rate))(x)  # Third LSTM layer (optional for deeper architectures)
        # x = Dropout(dropout_rate)(x)
        x = LSTM(last_lstm_units, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(regularization_rate))(x)  # Third LSTM layer (optional for deeper architectures)
        x = Dropout(dropout_rate_lstm)(x)

        # Add Learnable Positional Encoding (to capture temporal ordering in Transformer)
        positional_encoding_layer = LearnablePositionalEncoding(seq_len=input_shape[0], d_model=last_lstm_units)   #using 64 neurons 
        x = positional_encoding_layer(x)

        # First Transformer Block
        attn_output1 = MultiHeadAttention(num_heads=num_heads, key_dim=size_heads)(x, x)  # Self-attention mechanism
        attn_output1 = Dropout(dropout_rate_multihead)(attn_output1)
        attn_output1 = Add()([x, attn_output1])  # Residual Connection
        attn_output1 = LayerNormalization(epsilon=1e-6)(attn_output1)

        ffn_output1 = feed_forward_network(d_model=last_lstm_units, d_ff=d_ff)(attn_output1)   #using 64 neurons 
        ffn_output1 = Dropout(dropout_rate_dense)(ffn_output1)
        ffn_output1 = Add()([attn_output1, ffn_output1])  # Residual Connection
        x = LayerNormalization(epsilon=1e-6)(ffn_output1)

        # Second Transformer Block
        attn_output2 = MultiHeadAttention(num_heads=num_heads, key_dim=size_heads)(x, x)  # Self-attention mechanism
        attn_output2 = Dropout(dropout_rate_multihead)(attn_output2)
        attn_output2 = Add()([x, attn_output2])  # Residual Connection
        attn_output2 = LayerNormalization(epsilon=1e-6)(attn_output2)

        ffn_output2 = feed_forward_network(d_model=last_lstm_units, d_ff=d_ff)(attn_output2)
        ffn_output2 = Dropout(dropout_rate_dense)(ffn_output2)
        ffn_output2 = Add()([attn_output2, ffn_output2])  # Residual Connection
        x = LayerNormalization(epsilon=1e-6)(ffn_output2)
        
        # Third Transformer Block
        attn_output3 = MultiHeadAttention(num_heads=num_heads, key_dim=size_heads)(x, x)  # Self-attention mechanism
        # attn_output2 = Dropout(dropout_rate_multihead)(attn_output2)
        attn_output3 = Add()([x, attn_output3])  # Residual Connection
        attn_output3 = LayerNormalization(epsilon=1e-6)(attn_output3)

        ffn_output3 = feed_forward_network(d_model=last_lstm_units, d_ff=d_ff)(attn_output3)
        # ffn_output2 = Dropout(dropout_rate_dense)(ffn_output2)
        ffn_output3 = Add()([attn_output3, ffn_output3])  # Residual Connection
        x = LayerNormalization(epsilon=1e-6)(ffn_output3)
        
        # Dense layers for classification
        x = Dense(last_lstm_units, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(regularization_rate),activity_regularizer=tf.keras.regularizers.l1(1e-5))(x)
        x = Dropout(dropout_rate_dense)(x)

        # Output layer for multi-class classification
        outputs = Dense(num_classes, activation='softmax')(x)

        # Create and return the model (without compilation)
        model = Model(inputs, outputs)

        model.summary()
        return model