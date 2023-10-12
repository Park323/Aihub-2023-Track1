import os
import logging

import torch

try:
    import nova
except:
    pass

from inference import inference


def bind_model(model, optimizer=None):
    def save(path, *args, **kwargs):
        logging.info(f"Saving model to `{path}`")
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        torch.save(state, os.path.join(path, 'model.pt'))
        print('Model saved')

    def load(path, *args, **kwargs):
        state = torch.load(os.path.join(path, 'model.pt'))
        model.load_state_dict(state['model'])
        if 'optimizer' in state and optimizer:
            optimizer.load_state_dict(state['optimizer'])
        print('Model loaded')

    # 추론
    def infer(path, **kwargs):
        return inference(path, model)

    try:
        nova.bind(save=save, load=load, infer=infer)  # 'nova.bind' function must be called at the end.
        logging.info("NOVA successfully binded the model")
    except:
        print("No NOVA module.")
    