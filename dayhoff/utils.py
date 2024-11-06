import numpy as np


def cosine_anneal_with_warmup(n_warmup_steps, n_anneal_steps, final_ratio=0.0):
    # Linear warmup, then anneal from max lr to 0 over n_anneal_steps
    def get_lr(step):
        step += 1
        if step <= n_warmup_steps:
            return step / n_warmup_steps
        else:
            return final_ratio + 0.5 * (1 - final_ratio) * (1 + np.cos((step - n_warmup_steps) * np.pi / n_anneal_steps))
    return get_lr