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

def _inference(path, model, **kwargs):
    print(time.strftime("%Y-%m-%d/%H:%M", time.localtime()), "Start inference", path)
    
    device = "cuda"
    model = model.to(device)
    stt = Speech2Text(asr_model=model, ctc_weight=0.1, device="cuda")

    results = []
    for i in glob(os.path.join(path, '*')):
        results.append(
            {
                'filename': i.split('/')[-1],
                'text': single_infer(stt, i)
            }
        )
    print(time.strftime("%Y-%m-%d/%H:%M", time.localtime()), "Inference finished")
    return sorted(results, key=lambda x: x['filename'])

def inference_subprocess(gpu_id:int, job_id:int, paths, model, return_list, debug=False):
    if debug:
        print(time.strftime("%Y-%m-%d/%H:%M:%S", time.localtime()), job_id, "started on pid:", os.getpid())
    device = f"cuda:{gpu_id}"
    model = model.to(device)
    stt = Speech2Text(
        asr_model=model, device=device,
        ctc_weight=0.0, beam_size=3) ################# Debug config ###############

    for i in paths:
        return_list[i.split('/')[-1]] = single_infer(stt, i, debug=debug)

def inference(path, model, debug=False, **kwargs):
    """Inference Multiprocessing"""
    import torch.multiprocessing as mp
    print(time.strftime("%Y-%m-%d/%H:%M:%S", time.localtime()), "Start inference", path)
    
    torch.multiprocessing.set_start_method("spawn")

    n_gpu = torch.cuda.device_count()            ################# Debug environ. ###############
    inference_nj = n_gpu * 8
    test_paths = glob(os.path.join(path, '*'))
    if debug:
        test_paths = sorted(test_paths)[:5000]
        print(time.strftime("%Y-%m-%d/%H:%M:%S", time.localtime()), f"{len(test_paths)} of Inference data collected by {inference_nj} jobs with {n_gpu}", path)
    
    manager = mp.Manager()
    return_dict = manager.dict()
    jobs = []
    for i in range(inference_nj):
        gpu_id = int(i // (inference_nj / n_gpu))
        p = mp.Process(
            target=inference_subprocess, 
            args=(gpu_id, i, test_paths[i::inference_nj], model, return_dict, debug))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()
    
    results = []
    for key, value in return_dict.items():
        results.append({
            'filename': key,
            'text': value})

    print(time.strftime("%Y-%m-%d/%H:%M:%S", time.localtime()), "Inference finished")
    return sorted(results, key=lambda x: x['filename'])


def single_infer(stt, path, debug=False):
    signal = load_audio(path)
    text, token, token_int, hypothesis = stt(signal)[0]
    if stt.asr_model.token_normalize:
        text = grp2char(text)
    if debug:
        print(time.strftime("%Y-%m-%d/%H:%M:%S", time.localtime()), "\t".join([os.path.basename(path), text]))
    return text
