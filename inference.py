import os
from glob import glob

import pydub
import numpy as np
import torch
from torch import Tensor

from espnet2.bin.asr_inference import Speech2Text
from espnet2.text.korean_separator import grp2char


def load_audio(audio_path: str, extension: str = 'pcm') -> np.ndarray:
    """
    Load audio file (PCM) to sound. if del_silence is True, Eliminate all sounds below 30dB.
    If exception occurs in numpy.memmap(), return None.
    """
    try:
        if extension == 'pcm':
            signal = np.memmap(audio_path, dtype='h', mode='r').astype('float32')

            if sum(abs(signal)) <= 80:
                raise ValueError('[WARN] Silence file in {0}'.format(audio_path))

            return signal / 32767  # normalize audio

        elif extension == 'wav':
            aud = pydub.AudioSegment.from_wav(audio_path)
            aud = aud.set_frame_rate(16000)
            signal = np.array(aud.get_array_of_samples()).astype('float32')

            if sum(abs(signal)) <= 80:
                raise ValueError('[WARN] Silence file in {0}'.format(audio_path))

            return signal/32767

    except ValueError:
        # print('ValueError in {0}'.format(audio_path))
        return None
    except RuntimeError:
        print('RuntimeError in {0}'.format(audio_path))
        return None
    except IOError:
        print('IOError in {0}'.format(audio_path))
        return None

def inference(path, model, **kwargs):
    stt = Speech2Text(asr_model=model, ctc_weight=0.1, device="cuda")

    results = []
    for i in glob(os.path.join(path, '*')):
        results.append(
            {
                'filename': i.split('/')[-1],
                'text': single_infer(stt, i)
            }
        )
    return sorted(results, key=lambda x: x['filename'])

def single_infer(stt, path):
    signal = load_audio(path)
    text, token, token_int, hypothesis = stt(signal)[0]
    if stt.asr_model.token_normalize:
        text = grp2char(text)
    return text
    