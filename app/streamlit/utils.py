import tensorflow as tf

# takes values from a list of tensors and creates a combined tensor at the specified dimension
def pack_features_vector(features, labels):
  """Pack the features into a single array."""
  features = tf.stack(list(features.values()), axis=1)
  return features, labels