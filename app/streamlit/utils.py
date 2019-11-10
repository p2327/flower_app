import tensorflow as tf


# takes values from a list of tensors and creates
# a combined tensor at the specified dimension
def pack_features_vector(features, labels):
    """
    Pack the features into a single array.
    """
    features = tf.stack(list(features.values()), axis=1)
    return features, labels


def loss(model, x, y):
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True)
    # predictions
    y_ = model(x)
    # return cross-entropy (actual and predicted)
    return loss_object(y_true=y, y_pred=y_)


def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        # internal call to loss
        loss_value = loss(model, inputs, targets)
    # returns loss and gradients
    return loss_value, tape.gradient(loss_value, model.trainable_variables)
