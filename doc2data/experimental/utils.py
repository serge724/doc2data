import numpy as np
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
from doc2data.utils import denormalize_bbox

def draw_bounding_boxes(image, bounding_boxes, bbox_color = 'red'):
    """Draws bounding boxes."""

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    image_copy = image.copy()

    # check if only one bounding box was provided
    if isinstance(bounding_boxes[0], float):
        bounding_boxes = [bounding_boxes]

    bounding_boxes = np.array(bounding_boxes)

    for box in bounding_boxes:
        print(box)
        if all(box <= 1):
            box = denormalize_bbox(box, image.width, image.height)
            # maybe -1 on the end_x & end_y coordinate

        draw = ImageDraw.Draw(image_copy)
        draw.rectangle(box, outline = bbox_color)

    return image_copy

def plot_keras_history(history):
    """Generates plot from a Keras history object."""

    fig, (ax1, ax2) = plt.subplots(1, 2, sharex = True)
    fig.suptitle('Training progress')
    ax1.plot(history.history['loss'])
    ax1.plot(history.history['val_loss'])
    ax1.set(title = 'loss')
    ax1.legend(['training', 'validation'], loc='upper right')
    if 'mae' in history.history.keys():
        ax2.plot(history.history['mae'])
        ax2.plot(history.history['val_mae'])
        ax2.set(title = 'mae')
    if 'accuracy' in history.history.keys():
        ax2.plot(history.history['accuracy'])
        ax2.plot(history.history['val_accuracy'])
        ax2.set(title = 'accuracy')
    fig.tight_layout()
    fig.set_size_inches(w = 15, h = 5)
    return fig

# copied from:
# https://github.com/keras-team/keras/blob/v2.9.0/keras/utils/np_utils.py#L21-L74
def to_categorical(y, num_classes=None, dtype='float32'):
  """Converts a class vector (integers) to binary class matrix.
  E.g. for use with `categorical_crossentropy`.
  Args:
      y: Array-like with class values to be converted into a matrix
          (integers from 0 to `num_classes - 1`).
      num_classes: Total number of classes. If `None`, this would be inferred
        as `max(y) + 1`.
      dtype: The data type expected by the input. Default: `'float32'`.
  Returns:
      A binary matrix representation of the input. The class axis is placed
      last.
  Example:
  >>> a = tf.keras.utils.to_categorical([0, 1, 2, 3], num_classes=4)
  >>> a = tf.constant(a, shape=[4, 4])
  >>> print(a)
  tf.Tensor(
    [[1. 0. 0. 0.]
     [0. 1. 0. 0.]
     [0. 0. 1. 0.]
     [0. 0. 0. 1.]], shape=(4, 4), dtype=float32)
  >>> b = tf.constant([.9, .04, .03, .03,
  ...                  .3, .45, .15, .13,
  ...                  .04, .01, .94, .05,
  ...                  .12, .21, .5, .17],
  ...                 shape=[4, 4])
  >>> loss = tf.keras.backend.categorical_crossentropy(a, b)
  >>> print(np.around(loss, 5))
  [0.10536 0.82807 0.1011  1.77196]
  >>> loss = tf.keras.backend.categorical_crossentropy(a, a)
  >>> print(np.around(loss, 5))
  [0. 0. 0. 0.]
  """
  y = np.array(y, dtype='int')
  input_shape = y.shape
  if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
    input_shape = tuple(input_shape[:-1])
  y = y.ravel()
  if not num_classes:
    num_classes = np.max(y) + 1
  n = y.shape[0]
  categorical = np.zeros((n, num_classes), dtype=dtype)
  categorical[np.arange(n), y] = 1
  output_shape = input_shape + (num_classes,)
  categorical = np.reshape(categorical, output_shape)
  return categorical
