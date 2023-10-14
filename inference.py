import os
import time
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

# def inference(path, model, **kwargs):
#     print(time.strftime("%Y-%m-%d/%H:%M", time.localtime()), "Start inference", path)
    
#     device = "cuda"
#     model = model.to(device)
#     stt = Speech2Text(asr_model=model, ctc_weight=0.1, device="cuda")

#     results = []
#     for i in glob(os.path.join(path, '*')):
#         results.append(
#             {
#                 'filename': i.split('/')[-1],
#                 'text': single_infer(stt, i)
#             }
#         )
#     print(time.strftime("%Y-%m-%d/%H:%M", time.localtime()), "Inference finished")
#     return sorted(results, key=lambda x: x['filename'])

def inference_subprocess(job_id:int, paths, model, return_list):
    device = "cuda"
    model = model.to(device)
    stt = Speech2Text(asr_model=model, ctc_weight=0.1, device="cuda")

    for i in paths:
        return_list[i.split('/')[-1]] = single_infer(stt, i)

def inference(path, model, **kwargs):
    """Inference Multiprocessing"""
    import torch.multiprocessing as mp
    print(time.strftime("%Y-%m-%d/%H:%M", time.localtime()), "Start inference", path)
    
    torch.multiprocessing.set_start_method("spawn")

    inference_nj = 8
    test_paths = glob(os.path.join(path, '*'))
    
    manager = mp.Manager()
    return_dict = manager.dict()
    jobs = []
    for i in range(inference_nj):
        p = mp.Process(
            target=inference_subprocess, 
            args=(i, test_paths[i::inference_nj], model, return_dict))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()
    
    results = []
    for key, value in return_dict.items():
        results.append({
            'filename': key,
            'text': value})

    print(time.strftime("%Y-%m-%d/%H:%M", time.localtime()), "Inference finished")
    return sorted(results, key=lambda x: x['filename'])


def single_infer(stt, path):
    signal = load_audio(path)
    text, token, token_int, hypothesis = stt(signal)[0]
    if stt.asr_model.token_normalize:
        text = grp2char(text)
    print("\t".join([os.path.basename(path), text]))
    return text
    