import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Layer, Dense, Dropout
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, GlobalAveragePooling2D, Conv1D, Activation, Add


class ExtractPatches(Layer):
    """
    Custom layer to extract patches from input tensor.

    Args:
        **kwargs: Additional keyword arguments passed to the parent class.
    """
    def __init__(self, **kwargs):
        super(ExtractPatches, self).__init__(**kwargs)

    def call(self, inputs):
        """
        Perform the patch extraction operation.

        Args:
            inputs (tf.Tensor): Input tensor.

        Returns:
            tf.Tensor: Extracted patches tensor.
        """
        shape = tf.shape(inputs)
        reshaped_data = tf.reshape(inputs, (-1, shape[1], shape[2], 1))
        patches = tf.image.extract_patches(reshaped_data, sizes=[1, 16, 16, 1], strides=[1, 8, 8, 1], rates=[1,1,1,1], padding='SAME')
        return patches

def embed(data):
    """
    Embedding function to process input data.

    Args:
        data (tf.Tensor): Input data tensor.

    Returns:
        tf.Tensor: Embedded data tensor.
    """
    patches = ExtractPatches()(data)
    dense = Dense(128, activation='relu')(patches)
    return dense

class ConcatTensor(Layer):
    """
    Custom layer to concatenate input tensor with conditioning tensor.

    Args:
        **kwargs: Additional keyword arguments passed to the parent class.
    """
    def __init__(self, **kwargs):
        super(ConcatTensor, self).__init__(**kwargs)

    def call(self, input_tensor, conditioning_tensor):
        """
        Perform tensor concatenation.

        Args:
            input_tensor (tf.Tensor): Input tensor.
            conditioning_tensor (tf.Tensor): Conditioning tensor.

        Returns:
            tf.Tensor: Concatenated tensor.
        """

        # Reshape conditioning tensor to match the shape of input tensor along the appropriate axis
        conditioning_tensor = tf.expand_dims(conditioning_tensor, axis=-1)
        conditioning_tensor = tf.tile(conditioning_tensor, [1, 1, input_tensor.shape[2]])
        # Concatenate input tensor and reshaped conditioning tensor along the last axis
        concatenated_tensor = tf.concat([input_tensor, conditioning_tensor], axis=-1)

        return concatenated_tensor

class ReshapeLayer(Layer):
    """
    Custom layer to reshape input tensor.

    Args:
        **kwargs: Additional keyword arguments passed to the parent class.
    """
    def __init__(self, **kwargs):
        super(ReshapeLayer, self).__init__(**kwargs)

    def call(self, inputs):
        """
        Perform tensor reshaping.

        Args:
            inputs (tf.Tensor): Input tensor.

        Returns:
            tf.Tensor: Reshaped tensor.
        """
        reshaped_data = tf.reshape(inputs, (-1, 1))
        return reshaped_data

class ReshapeLayer2(Layer):
    """
    Custom layer to reshape input tensor based on another tensor's shape.

    Args:
        **kwargs: Additional keyword arguments passed to the parent class.
    """

    def __init__(self, **kwargs):
        super(ReshapeLayer2, self).__init__(**kwargs)

    def call(self, inputs, x):
        """
        Perform tensor reshaping based on another tensor's shape.

        Args:
            inputs (tf.Tensor): Input tensor.
            x (tf.Tensor): Tensor whose shape is used for reshaping.

        Returns:
            tf.Tensor: Reshaped tensor.
        """
        input_shape = tf.shape(inputs)
        new_shape = [-1, tf.shape(x)[1], tf.shape(x)[2]]  # Match the shape of x
        input_tensor = tf.reshape(inputs, new_shape)
        return input_tensor

# Build the Transformer Block

def transformer_encoder(inputs, head_size, num_heads, ff_units, dropout=0):
    """
    Perform transformer encoder operations.

    Args:
        inputs (tf.Tensor): Input tensor.
        head_size (int): Size of each attention head.
        num_heads (int): Number of attention heads.
        ff_units (int): Number of units in feed-forward network.
        dropout (float): Dropout rate.

    Returns:
        tf.Tensor: Output tensor after transformer encoding.
    """
    # Normalization and attention mechanism for capturing global dependencies
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = Dropout(dropout)(x)
    res = x + inputs  

    # Feed-forward network for further feature transformation
    x = LayerNormalization(epsilon=1e-6)(res)
    x = Dense(ff_units, activation="relu")(x)
    x = Dropout(dropout)(x)
    return x + res 

def diffusion_step(input_tensor, conditioning_tensor, num_filters):
    """
    Perform diffusion step to enhance audio.

    Args:
        input_tensor (tf.Tensor): Input tensor.
        conditioning_tensor (tf.Tensor): Conditioning tensor.
        num_filters (int): Number of filters for convolutional layers.

    Returns:
        tf.Tensor: Enhanced output tensor.
    """
    x = ConcatTensor()(input_tensor, conditioning_tensor)

    # Convolutional layers are for learning noise patterns and then their cancellation. Notice how we are convolving over une dimension and not in 2 as we did with images
    #x = Reshape((-1, 129, 1))(x)  # Add a singleton channel dimension
    x = Conv1D(num_filters, kernel_size=3, padding='same')(x)
    x = Activation('relu')(x)
    x = Conv1D(num_filters, kernel_size=1)(x)
    #input_tensor = ReshapeLayer2()(input_tensor, x)
    #input_tensor = ReshapeLayer2()(input_tensor, x)
    #convolve input_tensor to 
    return Add()([x, input_tensor]) 
# Assemble full model

def create_audio_model(input_shape, num_classes, num_transformer_blocks=2, num_diffusion_steps=10):
    """
    Create an audio model combining transformer and diffusion-like components.

    Args:
        input_shape (tuple): Shape of the input audio data.
        num_classes (int): Number of output classes.
        num_transformer_blocks (int): Number of transformer blocks.
        num_diffusion_steps (int): Number of diffusion steps.

    Returns:
        tf.keras.Model: Audio model.
    """

    audio_input = Input(shape = input_shape)
    #embedding_layer = embed(audio_input)
    #print(embedding_layer.shape)
    x = embed(audio_input)#Dense(64, activation='relu')(audio_input)  # Initial dense layer for dimensionality expansion
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size=128, num_heads=2, ff_units=128)
    classification_features = GlobalAveragePooling2D()(x)  # This will be your conditional tensor
    #x =  Dropout(0.1)(x)# add droput layer
    classification_output = Dense(num_classes, activation='softmax', name='classification_output')(classification_features)
     # This is the Diffusion-like model to enhancement audio that ises the transformer features for conditioning (conditional tensor)
    conditioning_input = classification_features  # Using classification features as a conditional tensor
    y = audio_input#ReshapeLayer()(audio_input)# Raw audio for enhancement
    for _ in range(num_diffusion_steps):
        y = diffusion_step(y, conditioning_input, num_filters=238)
    enhancement_output = Conv1D(1, kernel_size=1, activation='linear', name='enhancement_output')(y)

    model = Model(inputs=audio_input, outputs=[classification_output, enhancement_output]) # define inputs and outputs
    model.compile(optimizer='adam',
                  loss={'classification_output': 'sparse_categorical_crossentropy',
                        'enhancement_output': 'mean_squared_error'},
                  metrics={'classification_output': 'accuracy',
                           'enhancement_output': 'mean_squared_error'})
    return model

def main():
    return create_audio_model((128,238), 10)

    
if __name__ == "__main__":
    main()