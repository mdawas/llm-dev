from random import sample
from llm_dev.ch2.data_loader import create_dataloader_v1
import numpy as np


def test_create_dataloader():
    sample_text = """
    "It's the last he painted, you know,"
    Mrs. Gisburn said with pardonable pride.
    """
    loader = create_dataloader_v1(
        sample_text, batch_size=2, max_length=4, stride=2, shuffle=False
    )
    data_iter = iter(loader)
    first_batch = next(data_iter)
    # test input and output length, should be a batch of two samples
    assert first_batch[0].shape[0] == 2
    assert first_batch[1].shape[0] == 2

    # test input and output size, should be 4
    assert first_batch[0].shape[1] == 4
    assert first_batch[1].shape[1] == 4

    # test the stride is computed correctly
    # The beginning of the second sample must be the end of the first sample.
    assert np.array_equal(first_batch[0][0][-2:], first_batch[0][1][:2])

    # Across batches, the end of the last sample in the first batch must be
    # the same as the beginning of the first sample in the second batch.
    second_batch = next(data_iter)
    assert np.array_equal(first_batch[0][1][-2:], second_batch[0][0][:2])
