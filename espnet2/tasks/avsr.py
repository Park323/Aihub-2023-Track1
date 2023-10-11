import argparse
import logging
from typing import Callable, Collection, Dict, List, Optional, Tuple

import numpy as np
import torch
from typeguard import check_argument_types, check_return_type

from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr.decoder.hugging_face_transformers_decoder import (  # noqa: H301
    HuggingFaceTransformersDecoder,
)
from espnet2.asr.decoder.mlm_decoder import MLMDecoder
from espnet2.asr.decoder.rnn_decoder import RNNDecoder
from espnet2.asr.decoder.s4_decoder import S4Decoder
from espnet2.asr.decoder.transducer_decoder import TransducerDecoder
from espnet2.asr.decoder.transformer_decoder import (
    DynamicConvolution2DTransformerDecoder,
    DynamicConvolutionTransformerDecoder,
    LightweightConvolution2DTransformerDecoder,
    LightweightConvolutionTransformerDecoder,
    TransformerDecoder,
)
from espnet2.asr.decoder.whisper_decoder import OpenAIWhisperDecoder
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.encoder.branchformer_encoder import BranchformerEncoder
from espnet2.asr.encoder.conformer_encoder import ConformerEncoder
from espnet2.asr.encoder.contextual_block_conformer_encoder import (
    ContextualBlockConformerEncoder,
)
from espnet2.asr.encoder.contextual_block_transformer_encoder import (
    ContextualBlockTransformerEncoder,
)
from espnet2.asr.encoder.e_branchformer_encoder import EBranchformerEncoder
from espnet2.asr.encoder.hubert_encoder import (
    FairseqHubertEncoder,
    FairseqHubertPretrainEncoder,
    TorchAudioHuBERTPretrainEncoder,
)
from espnet2.asr.encoder.longformer_encoder import LongformerEncoder
from espnet2.asr.encoder.rnn_encoder import RNNEncoder
from espnet2.asr.encoder.transformer_encoder import TransformerEncoder
from espnet2.asr.encoder.transformer_encoder_multispkr import (
    TransformerEncoder as TransformerEncoderMultiSpkr,
)
from espnet2.asr.encoder.vgg_rnn_encoder import VGGRNNEncoder
from espnet2.asr.encoder.wav2vec2_encoder import FairSeqWav2Vec2Encoder
from espnet2.asr.encoder.whisper_encoder import OpenAIWhisperEncoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend as ASRFrontend
from espnet2.asr.frontend.default import DefaultFrontend
from espnet2.asr.frontend.fused import FusedFrontends
from espnet2.asr.frontend.resnet import ResNet18 as ResNet18_1D
from espnet2.asr.frontend.s3prl import S3prlFrontend
from espnet2.asr.frontend.whisper import WhisperFrontend
from espnet2.asr.frontend.windowing import SlidingWindow
from espnet2.asr.postencoder.abs_postencoder import AbsPostEncoder
from espnet2.asr.postencoder.hugging_face_transformers_postencoder import (
    HuggingFaceTransformersPostEncoder,
)
from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from espnet2.asr.preencoder.linear import LinearProjection
from espnet2.asr.preencoder.sinc import LightweightSincConvs
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.asr.specaug.specaug import SpecAug
from espnet2.asr_transducer.joint_network import JointNetwork
from espnet2.avsr.encoder import (
    AbsFusionEncoder,
    LateFusionEncoder,
    EarlyFusionEncoder
)
from espnet2.avsr.iip_avsr_model import ESPnetAVSRModel
from espnet2.avsr.postencoder.timecrop import FirstT
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.layers.global_mvn import GlobalMVN
from espnet2.layers.utterance_mvn import UtteranceMVN
from espnet2.tasks.abs_task import AbsTask
from espnet2.text.phoneme_tokenizer import g2p_choices
from espnet2.torch_utils.initialize import initialize
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet2.train.class_choices import ClassChoices
from espnet2.train.collate_fn import CommonCollateFn
from espnet2.train.distributed_utils import DistributedOption
from espnet2.train.preprocessor import (
    AbsPreprocessor,
    AudioVisualPreprocessor,
)
from espnet2.train.trainer import Trainer
from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet2.utils.nested_dict_action import NestedDictAction
from espnet2.utils.types import float_or_none, int_or_none, str2bool, str_or_none
from espnet2.vsr.frontend.resnet import ResNet18 as ResNet18_2D
from espnet2.vsr.frontend.abs_frontend import AbsFrontend as VSRFrontend
from espnet2.vsr.frontend.raw import RawFrontend
from espnet2.vsr.aug.aug import DefaultAug
from espnet2.vsr.aug.corrupt import VisualCorruptModel


# ======================
# Audio Modules
# ======================
audio_frontend_choices = ClassChoices(
    name="audio_frontend",
    classes=dict(
        default=DefaultFrontend,
        resnet=ResNet18_1D,
        sliding_window=SlidingWindow,
        s3prl=S3prlFrontend,
        fused=FusedFrontends,
        whisper=WhisperFrontend,
    ),
    type_check=ASRFrontend,
    default=None,
    optional=True,
)
audio_specaug_choices = ClassChoices(
    name="audio_specaug",
    classes=dict(
        specaug=SpecAug,
    ),
    type_check=AbsSpecAug,
    default=None,
    optional=True,
)
audio_normalize_choices = ClassChoices(
    "audio_normalize",
    classes=dict(
        global_mvn=GlobalMVN,
        utterance_mvn=UtteranceMVN,
    ),
    type_check=AbsNormalize,
    default="utterance_mvn",
    optional=True,
)
audio_preencoder_choices = ClassChoices(
    name="audio_preencoder",
    classes=dict(
        sinc=LightweightSincConvs,
        linear=LinearProjection,
    ),
    type_check=AbsPreEncoder,
    default=None,
    optional=True,
)
audio_encoder_choices = ClassChoices(
    "audio_encoder",
    classes=dict(
        conformer=ConformerEncoder,
        transformer=TransformerEncoder,
        transformer_multispkr=TransformerEncoderMultiSpkr,
        contextual_block_transformer=ContextualBlockTransformerEncoder,
        contextual_block_conformer=ContextualBlockConformerEncoder,
        vgg_rnn=VGGRNNEncoder,
        rnn=RNNEncoder,
        wav2vec2=FairSeqWav2Vec2Encoder,
        hubert=FairseqHubertEncoder,
        hubert_pretrain=FairseqHubertPretrainEncoder,
        torchaudiohubert=TorchAudioHuBERTPretrainEncoder,
        longformer=LongformerEncoder,
        branchformer=BranchformerEncoder,
        whisper=OpenAIWhisperEncoder,
        e_branchformer=EBranchformerEncoder,
    ),
    type_check=AbsEncoder,
    default=None,
    optional=True,
)
audio_postencoder_choices = ClassChoices(
    name="audio_postencoder",
    classes=dict(
    ),
    type_check=AbsPostEncoder,
    default=None,
    optional=True,
)
# ==============================
# Visual Modules
# ==============================
visual_frontend_choices = ClassChoices(
    name="visual_frontend",
    classes=dict(
        raw=RawFrontend,
        resnet=ResNet18_2D,
    ),
    type_check=VSRFrontend,
    default=None,
    optional=True,
)
visual_aug_choices = ClassChoices(
    name="visual_aug",
    classes=dict(
        default=DefaultAug,
        corrupt=VisualCorruptModel,
    ),
    default=None,
    optional=True,
)
visual_normalize_choices = ClassChoices(
    "visual_normalize",
    classes=dict(
        global_mvn=GlobalMVN,
        utterance_mvn=UtteranceMVN,
    ),
    type_check=AbsNormalize,
    default="utterance_mvn",
    optional=True,
)
visual_preencoder_choices = ClassChoices(
    name="visual_preencoder",
    classes=dict(),
    type_check=AbsPreEncoder,
    default=None,
    optional=True,
)
visual_encoder_choices = ClassChoices(
    "visual_encoder",
    classes=dict(
        conformer=ConformerEncoder,
        transformer=TransformerEncoder,
        rnn=RNNEncoder,
        contextual_block_transformer=ContextualBlockTransformerEncoder,
        contextual_block_conformer=ContextualBlockConformerEncoder,
    ),
    type_check=AbsEncoder,
    default=None,
    optional=True,
)
visual_postencoder_choices = ClassChoices(
    name="visual_postencoder",
    classes=dict(
    ),
    type_check=AbsPostEncoder,
    default=None,
    optional=True,
)
# ========================
# AudioVisual module
# ! This is used only for early fusion
# ========================
av_encoder_choices = ClassChoices(
    "av_encoder",
    classes=dict(
        conformer=ConformerEncoder,
        transformer=TransformerEncoder,
        transformer_multispkr=TransformerEncoderMultiSpkr,
        contextual_block_transformer=ContextualBlockTransformerEncoder,
        contextual_block_conformer=ContextualBlockConformerEncoder,
        vgg_rnn=VGGRNNEncoder,
        rnn=RNNEncoder,
        wav2vec2=FairSeqWav2Vec2Encoder,
        hubert=FairseqHubertEncoder,
        hubert_pretrain=FairseqHubertPretrainEncoder,
        torchaudiohubert=TorchAudioHuBERTPretrainEncoder,
        longformer=LongformerEncoder,
        branchformer=BranchformerEncoder,
        whisper=OpenAIWhisperEncoder,
        e_branchformer=EBranchformerEncoder,
    ),
    type_check=AbsEncoder,
    default=None,
    optional=True,
)
av_postencoder_choices = ClassChoices(
    name="av_postencoder",
    classes=dict(
        first=FirstT
    ),
    type_check=AbsPostEncoder,
    default=None,
    optional=True,
)
# ========================
# Common
# ========================
model_choices = ClassChoices(
    "model",
    classes=dict(
        espnet=ESPnetAVSRModel,
    ),
    type_check=AbsESPnetModel,
    default="espnet",
)
encoder_choices = ClassChoices(
    "encoder",
    classes=dict(
        late=LateFusionEncoder,
        early=EarlyFusionEncoder,
    ),
    type_check=AbsFusionEncoder,
    default='late',
)
decoder_choices = ClassChoices(
    "decoder",
    classes=dict(
        transformer=TransformerDecoder,
        lightweight_conv=LightweightConvolutionTransformerDecoder,
        lightweight_conv2d=LightweightConvolution2DTransformerDecoder,
        dynamic_conv=DynamicConvolutionTransformerDecoder,
        dynamic_conv2d=DynamicConvolution2DTransformerDecoder,
        rnn=RNNDecoder,
        transducer=TransducerDecoder,
        mlm=MLMDecoder,
        whisper=OpenAIWhisperDecoder,
        hugging_face_transformers=HuggingFaceTransformersDecoder,
        s4=S4Decoder,
    ),
    type_check=AbsDecoder,
    default=None,
    optional=True,
)
preprocessor_choices = ClassChoices(
    "preprocessor",
    classes=dict(
        default=AudioVisualPreprocessor,
        # multi=AudioVisualPreprocessor_multi,
    ),
    type_check=AbsPreprocessor,
    default="default",
)


class AVSRTask(AbsTask):
    # If you need more than one optimizers, change this value
    num_optimizers: int = 1

    # Add variable objects configurations
    class_choices_list = [
        # --audio_frontend and --audio_frontend_conf
        audio_frontend_choices,
        # --audio_specaug and --audio_specaug_conf
        audio_specaug_choices,
        # --audio_normalize and --audio_normalize_conf
        audio_normalize_choices,
        # --audio_preencoder and --audio_preencoder_conf
        audio_preencoder_choices,
        # --audio_encoder and --audio_encoder_conf
        audio_encoder_choices,
        # --audio_postencoder and --audio_postencoder_conf
        audio_postencoder_choices,
        # --visual_frontend and --visual_frontend_conf
        visual_frontend_choices,
        # --visual_aug and --visual_aug_conf
        visual_aug_choices,
        # --visual_normalize and --visual_normalize_conf
        visual_normalize_choices,
        # --visual_preencoder and --visual_preencoder_conf
        visual_preencoder_choices,
        # --visual_encoder and --visual_encoder_conf
        visual_encoder_choices,
        # --visual_postencoder and --visual_postencoder_conf
        visual_postencoder_choices,
        # --av_encoder and --av_encoder_conf
        av_encoder_choices,
        # --av_postencoder and --av_postencoder_conf
        av_postencoder_choices,
        # --model and --model_conf
        model_choices,
        # --encoder and --encoder_conf
        encoder_choices,
        # --decoder and --decoder_conf
        decoder_choices,
        # --preprocessor and --preprocessor_conf
        preprocessor_choices,
    ]

    # If you need to modify train() or eval() procedures, change Trainer class here
    trainer = Trainer

    @classmethod
    def add_task_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(description="Task related")

        # NOTE(kamo): add_arguments(..., required=True) can't be used
        # to provide --print_config mode. Instead of it, do as
        required = parser.get_default("required")
        required += ["token_list"]

        group.add_argument(
            "--token_list",
            type=str_or_none,
            default=None,
            help="A text mapping int-id to token",
        )
        group.add_argument(
            "--init",
            type=lambda x: str_or_none(x.lower()),
            default=None,
            help="The initialization method",
            choices=[
                "chainer",
                "xavier_uniform",
                "xavier_normal",
                "kaiming_uniform",
                "kaiming_normal",
                None,
            ],
        )

        group.add_argument(
            "--curriculum_epochs",
            default=None,
            help="The number of epochs for each curriculum step",
        )

        group.add_argument(
            "--curriculum_max_len",
            default=None,
            help="The maximum length of audio inputs for each curriculum step, paired with `curriculum_epochs`",
        )

        group.add_argument(
            "--sync_batchnorm",
            default=False,
            help="Sync batchnormalization for multiple GPUs",
        )

        group.add_argument(
            "--audio_input_size",
            type=int_or_none,
            default=None,
            help="The number of input dimension of the feature",
        )

        group.add_argument(
            "--visual_input_size",
            type=int_or_none,
            default=None,
            help="The number of input dimension of the feature",
        )

        group.add_argument(
            "--ctc_conf",
            action=NestedDictAction,
            default=get_default_kwargs(CTC),
            help="The keyword arguments for CTC class.",
        )
        group.add_argument(
            "--joint_net_conf",
            action=NestedDictAction,
            default=None,
            help="The keyword arguments for joint network class.",
        )

        group = parser.add_argument_group(description="Preprocess related")
        group.add_argument(
            "--use_preprocessor",
            type=str2bool,
            default=True,
            help="Apply preprocessing to data or not",
        )
        group.add_argument(
            "--token_type",
            type=str,
            default="bpe",
            choices=[
                "bpe",
                "char",
                "word",
                "phn",
                "hugging_face",
                "whisper_en",
                "whisper_multilingual",
            ],
            help="The text will be tokenized " "in the specified level token",
        )
        group.add_argument(
            "--bpemodel",
            type=str_or_none,
            default=None,
            help="The model file of sentencepiece",
        )
        parser.add_argument(
            "--non_linguistic_symbols",
            type=str_or_none,
            help="non_linguistic_symbols file path",
        )
        group.add_argument(
            "--cleaner",
            type=str_or_none,
            choices=[
                None,
                "tacotron",
                "jaconv",
                "vietnamese",
                "whisper_en",
                "whisper_basic",
            ],
            default=None,
            help="Apply text cleaning",
        )
        group.add_argument(
            "--g2p",
            type=str_or_none,
            choices=g2p_choices,
            default=None,
            help="Specify g2p method if --token_type=phn",
        )
        group.add_argument(
            "--speech_volume_normalize",
            type=float_or_none,
            default=None,
            help="Scale the maximum amplitude to the given value.",
        )
        group.add_argument(
            "--rir_scp",
            type=str_or_none,
            default=None,
            help="The file path of rir scp file.",
        )
        group.add_argument(
            "--rir_apply_prob",
            type=float,
            default=1.0,
            help="THe probability for applying RIR convolution.",
        )
        group.add_argument(
            "--noise_scp",
            type=str_or_none,
            default=None,
            help="The file path of noise scp file.",
        )
        group.add_argument(
            "--noise_apply_prob",
            type=float,
            default=1.0,
            help="The probability applying Noise adding.",
        )
        group.add_argument(
            "--noise_db_range",
            type=str,
            default="13_15",
            help="The range of noise decibel level.",
        )
        group.add_argument(
            "--short_noise_thres",
            type=float,
            default=0.5,
            help="If len(noise) / len(speech) is smaller than this threshold during "
            "dynamic mixing, a warning will be displayed.",
        )
        group.add_argument(
            "--aux_ctc_tasks",
            type=str,
            nargs="+",
            default=[],
            help="Auxillary tasks to train on using CTC loss. ",
        )

        for class_choices in cls.class_choices_list:
            # Append --<name> and --<name>_conf.
            # e.g. --encoder and --encoder_conf
            class_choices.add_arguments(group)

    @classmethod
    def build_collate_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Callable[
        [Collection[Tuple[str, Dict[str, np.ndarray]]]],
        Tuple[List[str], Dict[str, torch.Tensor]],
    ]:
        assert check_argument_types()
        # NOTE(kamo): int value = 0 is reserved by CTC-blank symbol
        return CommonCollateFn(float_pad_value=0.0, int_pad_value=-1)

    @classmethod
    def build_preprocess_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Optional[Callable[[str, Dict[str, np.array]], Dict[str, np.ndarray]]]:
        assert check_argument_types()
        if args.use_preprocessor:
            try:
                _ = getattr(args, "preprocessor")
            except AttributeError:
                setattr(args, "preprocessor", "default")
                setattr(args, "preprocessor_conf", dict())
            except Exception as e:
                raise e

            preprocessor_class = preprocessor_choices.get_class(args.preprocessor)
            retval = preprocessor_class(
                train=train,
                token_type=args.token_type,
                token_list=args.token_list,
                bpemodel=args.bpemodel,
                non_linguistic_symbols=args.non_linguistic_symbols,
                text_cleaner=args.cleaner,
                g2p_type=args.g2p,
                # NOTE(kamo): Check attribute existence for backward compatibility
                rir_scp=args.rir_scp if hasattr(args, "rir_scp") else None,
                rir_apply_prob=args.rir_apply_prob
                if hasattr(args, "rir_apply_prob")
                else 1.0,
                noise_scp=args.noise_scp if hasattr(args, "noise_scp") else None,
                noise_apply_prob=args.noise_apply_prob
                if hasattr(args, "noise_apply_prob")
                else 1.0,
                noise_db_range=args.noise_db_range
                if hasattr(args, "noise_db_range")
                else "13_15",
                short_noise_thres=args.short_noise_thres
                if hasattr(args, "short_noise_thres")
                else 0.5,
                speech_volume_normalize=args.speech_volume_normalize
                if hasattr(args, "rir_scp")
                else None,
                aux_task_names=args.aux_ctc_tasks
                if hasattr(args, "aux_ctc_tasks")
                else None,
                **args.preprocessor_conf,
            )
        else:
            retval = None
        assert check_return_type(retval)
        return retval

    @classmethod
    def build_iter_options(
        cls,
        args: argparse.Namespace,
        distributed_option: DistributedOption,
        mode: str,
    ):
        args.train_shape_file = [args.train_shape_file[0]]
        args.valid_shape_file = [args.valid_shape_file[0]]
        
        return super().build_iter_options(
            args, distributed_option=distributed_option, mode=mode
        )

    @classmethod
    def required_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        if not inference:
            retval = ("speech", "visual", "text")
        else:
            # Recognition mode
            retval = ("speech", "visual")
        return retval

    @classmethod
    def optional_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        MAX_REFERENCE_NUM = 4

        retval = ["text_spk{}".format(n) for n in range(2, MAX_REFERENCE_NUM + 1)]
        retval = tuple(retval)

        logging.info(f"Optional Data Names: {retval }")
        assert check_return_type(retval)
        return retval

    @classmethod
    def build_model(cls, args: argparse.Namespace) -> ESPnetAVSRModel:
        assert check_argument_types()
        if isinstance(args.token_list, str):
            with open(args.token_list, encoding="utf-8") as f:
                token_list = [line.rstrip() for line in f]

            # Overwriting token_list to keep it as "portable".
            args.token_list = list(token_list)
        elif isinstance(args.token_list, (tuple, list)):
            token_list = list(args.token_list)
        else:
            raise RuntimeError("token_list must be str or list")

        # If use multi-blank transducer criterion,
        # big blank symbols are added just before the standard blank
        if args.model_conf.get("transducer_multi_blank_durations", None) is not None:
            sym_blank = args.model_conf.get("sym_blank", "<blank>")
            blank_idx = token_list.index(sym_blank)
            for dur in args.model_conf.get("transducer_multi_blank_durations"):
                if f"<blank{dur}>" not in token_list:  # avoid this during inference
                    token_list.insert(blank_idx, f"<blank{dur}>")
            args.token_list = token_list

        vocab_size = len(token_list)
        logging.info(f"Vocabulary size: {vocab_size }")

        # 0. Initialize a dictionary for Optional Modules
        optional_modules = dict()

        # 1. frontend
        # 1.1 Audio Frontend
        if args.audio_frontend is not None:
            # Extract features in the model
            audio_frontend_class = audio_frontend_choices.get_class(args.audio_frontend)
            audio_frontend = audio_frontend_class(**args.audio_frontend_conf)
            audio_input_size = audio_frontend.output_size()
        else:
            # Give features from data-loader
            args.audio_frontend = None
            args.audio_frontend_conf = {}
            audio_frontend = None
            audio_input_size = args.audio_input_size
        # 1.2 Visual Frontend
        if args.audio_frontend is not None:
            # Extract features in the model
            visual_frontend_class = visual_frontend_choices.get_class(args.visual_frontend)
            visual_frontend = visual_frontend_class(**args.visual_frontend_conf)
            visual_input_size = visual_frontend.output_size()
        else:
            # Give features from data-loader
            args.visual_frontend = None
            args.visual_frontend_conf = {}
            visual_frontend = None
            visual_input_size = args.visual_input_size

        # 2. Data augmentation
        # 2.1 Data augmentation for audio specaug
        if args.audio_specaug is not None:
            audio_specaug_class = audio_specaug_choices.get_class(args.audio_specaug)
            audio_specaug = audio_specaug_class(**args.audio_specaug_conf)
        else:
            audio_specaug = None
        # 2.2 Data augmentation for image sequences
        if args.visual_aug is not None:
            visual_aug_class = visual_aug_choices.get_class(args.visual_aug)
            visual_aug = visual_aug_class(**args.visual_aug_conf)
        else:
            visual_aug = None

        # 3. Normalization layer
        # 3.1 Audio Normalization
        if args.audio_normalize is None or (args.audio_normalize=='global_mvn' and args.collect_stats):
            audio_normalize = None
        else:
            audio_normalize_class = audio_normalize_choices.get_class(args.audio_normalize)
            audio_normalize = audio_normalize_class(**args.audio_normalize_conf)
        # 3.2 Visual Normalization
        if args.visual_normalize is None or (args.visual_normalize=='global_mvn' and args.collect_stats):
            visual_normalize = None
        else:
            visual_normalize_class = visual_normalize_choices.get_class(args.visual_normalize)
            visual_normalize = visual_normalize_class(**args.visual_normalize_conf)
        
        # 4. Pre-encoder input block
        # NOTE(kan-bayashi): Use getattr to keep the compatibility
        # 4.1 Audio Pre-encoder
        if getattr(args, "audio_preencoder", None) is not None:
            audio_preencoder_class = audio_preencoder_choices.get_class(args.audio_preencoder)
            audio_preencoder = audio_preencoder_class(**args.audio_preencoder_conf)
            audio_input_size = audio_preencoder.output_size()
        else:
            audio_preencoder = None
        # 4.2 Visual Pre-encoder
        if getattr(args, "visual_preencoder", None) is not None:
            visual_preencoder_class = visual_preencoder_choices.get_class(args.visual_preencoder)
            visual_preencoder = visual_preencoder_class(**args.visual_preencoder_conf)
            visual_input_size = visual_preencoder.output_size()
        else:
            visual_preencoder = None

        # 6. Encoder
        if getattr(args, "visual_encoder", None) is not None and getattr(args, "audio_encoder", None) is not None:
            # 6.1 Audio Encoder
            audio_encoder_class = audio_encoder_choices.get_class(args.audio_encoder)
            audio_encoder = audio_encoder_class(input_size=audio_input_size, **args.audio_encoder_conf)
            # 6.2 Visual Encoder
            visual_encoder_class = visual_encoder_choices.get_class(args.visual_encoder)
            visual_encoder = visual_encoder_class(input_size=visual_input_size, **args.visual_encoder_conf)

            optional_modules['audioEncoder'] = audio_encoder
            optional_modules['visualEncoder'] = visual_encoder
            av_encoder = None
        elif getattr(args, "av_encoder", None) is not None:
            # 6.1 AV Encoder
            av_encoder_class = av_encoder_choices.get_class(args.av_encoder)
            av_encoder = av_encoder_class(**args.av_encoder_conf)
            
            optional_modules['avEncoder'] = av_encoder
            audio_encoder = None
            visual_encoder = None
        else:
            raise BaseException(msg="'Audio encoder and visual encoder' or 'AudioVisual early encoder' should be defined!!")

        # 7. Post-encoder block
        # NOTE(kan-bayashi): Use getattr to keep the compatibility
        if optional_modules.get('audioEncoder', None) is not None:
            # 7.1 Audio Post-encoder block
            audio_encoder_output_size = audio_encoder.output_size()
            if getattr(args, "audio_postencoder", None) is not None:
                audio_postencoder_class = audio_postencoder_choices.get_class(args.audio_postencoder)
                audio_postencoder = audio_postencoder_class(
                    input_size=audio_encoder_output_size, **args.audio_postencoder_conf
                )
                audio_encoder_output_size = audio_postencoder.output_size()
            else:
                audio_postencoder = None
            # 7.2 Visual Post-encoder block
            visual_encoder_output_size = visual_encoder.output_size()
            if getattr(args, "visual_postencoder", None) is not None:
                visual_postencoder_class = visual_postencoder_choices.get_class(args.visual_postencoder)
                visual_postencoder = visual_postencoder_class(
                    input_size=visual_encoder_output_size, **args.visual_postencoder_conf
                )
                visual_encoder_output_size = visual_postencoder.output_size()
            else:
                visual_postencoder = None
            optional_modules['audioPostencoder'] = audio_postencoder
            optional_modules['visualPostencoder'] = visual_postencoder
        else:
            # 7.3 AV Post-encoder block
            av_encoder_output_size = av_encoder.output_size()
            if getattr(args, "av_postencoder", None) is not None:
                av_postencoder_class = av_postencoder_choices.get_class(args.av_postencoder)
                av_postencoder = av_postencoder_class(
                    input_size=av_encoder_output_size, **args.av_postencoder_conf
                )
                av_encoder_output_size = av_postencoder.output_size()
            else:
                av_postencoder = None
            optional_modules['avPostencoder'] = av_postencoder

        # 8. Build Encoder ( Contains whole upper modules )
        encoder_class = encoder_choices.get_class(args.encoder)
        encoder = encoder_class(
            vocab_size=vocab_size,
            audioFrontend=audio_frontend,
            audioSpecaug=audio_specaug,
            audioNormalize=audio_normalize,
            audioPreencoder=audio_preencoder,
            visualFrontend=visual_frontend,
            visualAug=visual_aug,
            visualNormalize=visual_normalize,
            visualPreencoder=visual_preencoder,
            **args.encoder_conf,
            **optional_modules,
        )
        encoder_output_size = encoder.output_size()

        # 9. Decoder
        if getattr(args, "decoder", None) is not None:
            decoder_class = decoder_choices.get_class(args.decoder)

            if args.decoder == "transducer":
                decoder = decoder_class(
                    vocab_size,
                    embed_pad=0,
                    **args.decoder_conf,
                )

                joint_network = JointNetwork(
                    vocab_size,
                    encoder.output_size(),
                    decoder.dunits,
                    **args.joint_net_conf,
                )
            else:
                decoder = decoder_class(
                    vocab_size=vocab_size,
                    encoder_output_size=encoder_output_size,
                    **args.decoder_conf,
                )
                joint_network = None
        else:
            decoder = None
            joint_network = None

        # 10. CTC
        ctc = CTC(
            odim=vocab_size, encoder_output_size=encoder_output_size, **args.ctc_conf
        )

        # 11. Build model
        try:
            model_class = model_choices.get_class(args.model)
        except AttributeError:
            model_class = model_choices.get_class("espnet")
        model = model_class(
            vocab_size=vocab_size,
            encoder=encoder,
            decoder=decoder,
            ctc=ctc,
            joint_network=joint_network,
            token_list=token_list,
            **args.model_conf,
        )

        # FIXME(kamo): Should be done in model?
        # 12. Initialize
        if args.init is not None:
            initialize(model, args.init)

        # 13. DDP Synchronize the model
        if args.ngpu > 1 and args.get("sync_batchnorm", False):
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        elif args.ngpu > 1:
            logging.warning("BatchNorm modules are not synchronized among the GPU devices")

        assert check_return_type(model)
        return model
