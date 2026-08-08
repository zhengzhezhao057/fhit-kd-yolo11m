import random

import numpy as np
import torch

from src.train_teacher import capture_rng_state, restore_rng_state


def test_teacher_rng_resume_replays_the_next_random_values():
    random.seed(7); np.random.seed(7); torch.manual_seed(7)
    generator = torch.Generator().manual_seed(7)
    _ = (random.random(), np.random.rand(), torch.rand(1), torch.randperm(8, generator=generator))
    state = capture_rng_state(generator)
    expected = (random.random(), np.random.rand(), torch.rand(1), torch.randperm(8, generator=generator))
    restore_rng_state(state, generator)
    actual = (random.random(), np.random.rand(), torch.rand(1), torch.randperm(8, generator=generator))
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])
    assert torch.equal(actual[3], expected[3])
