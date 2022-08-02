import tensorflow_transform as tft

from preprocessing.features import NUM_OOV_TOKENS


def get_vocabulary_size(
    transform_output: tft.TFTransformOutput, vocab_filename: str
) -> int:
    """
    Returns size of a vocabulary given by name.
    """
    vocab_size: int = transform_output.vocabulary_size_by_name(vocab_filename)
    return vocab_size + NUM_OOV_TOKENS
