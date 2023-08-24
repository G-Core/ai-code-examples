import logging
import requests
import io
import imghdr
import os
from typing import Tuple
import tempfile

from ssf.application import SSFApplicationInterface, SSFApplicationTestInterface
from ssf.utils import get_ipu_count
from ssf.results import *
from datasets import Dataset, disable_caching, load_dataset
import audiosegment
import numpy as np
import poptorch
import torch
from dataclasses import dataclass, field
from optimum.graphcore import IPUConfig, IPUTrainer, IPUTrainingArguments
from optimum.graphcore.modeling_utils import to_pipelined
#from optimum.utils import logging
from pydub import AudioSegment
from pydub.silence import split_on_silence
from tqdm import tqdm
from transformers import (WhisperConfig, WhisperForConditionalGeneration,
                          WhisperProcessor)
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from werkzeug.utils import secure_filename
from string import ascii_lowercase as alc
import time

logger = logging.getLogger("Whisper")


def check_available_ipus(req_ipus: int) -> bool:
    return get_ipu_count() >= req_ipus


def compile_or_load_model_exe(pipe, processor):
    try:
        logger.warning(f"Run compile_or_load_model_exe")
        import os
        cwd = os.getcwd()
        logger.warning(f"Run {cwd}")
        input_file = "./news.mp3"
        sound_file = audiosegment.from_file(input_file).resample(sample_rate_Hz=16000)
        audio_chunks = split_on_silence(sound_file, min_silence_len=500, silence_thresh=-40)

        for i, chunk in enumerate(audio_chunks):
            q, mod = divmod(i,26)
            out_file = "./start" + "/chunk{0}{1}.mp3".format(alc[q],alc[mod])
            chunk.export(out_file, format="mp3")
        ds = load_dataset("./start/", split="train", num_proc=1)
        input_features = []
        for d in ds:
            input_features.append(processor(d["audio"]["array"],
                                   return_tensors="pt",
                                   sampling_rate=d['audio']['sampling_rate']).input_features.half())

        chank_number = 0
        transcription_sum = ""
        headers = {'Content-Type': 'text/text; charset=utf-8'}
        for i_f in input_features:
            chank_number += 1
            sample_output = pipe.generate(i_f, max_length=448, min_length=3)
            transcription = processor.batch_decode(sample_output, skip_special_tokens=True)[0]
            transcription_sum +=transcription + ". "

    except Exception as e:
        logger.error(f"Model compile/load step failed with {e}.")
        return RESULT_FAIL

    return RESULT_OK

@dataclass
class IPUWhisperConf:
        """A data class to collect IPU-related config parameters"""
        model_spec: str
        ipus_per_replica: int
        pod_type: str
        use_cache: True

class Whisper2API(SSFApplicationInterface):
    def __init__(self):

        self._pipe = None
        self._processor = None
        max_ipus = 4
        self.req_ipus = max_ipus
        model_path = './whisper-small_gpu_83k-iters'
        ipu_whisper = {
            "tiny": IPUWhisperConf(model_spec='openai/whisper-tiny.en', ipus_per_replica=2, pod_type="pod4", use_cache=True),
            "small": IPUWhisperConf(model_spec=model_path, ipus_per_replica=2, pod_type="pod4", use_cache=True),
            "large": IPUWhisperConf(model_spec='openai/whisper-large-v2', ipus_per_replica=16, pod_type="pod16",use_cache=True)
        }
        model_size = "small"
        iwc = ipu_whisper[model_size]
        self._processor = WhisperProcessor.from_pretrained(iwc.model_spec)
        model = WhisperForConditionalGeneration.from_pretrained(iwc.model_spec)
        model.config.forced_decoder_ids = None


        pod_type = os.getenv("GRAPHCORE_POD_TYPE", iwc.pod_type)
        ipu_config = IPUConfig(executable_cache_dir='/tmp/whisper_exe_cache/', ipus_per_replica=iwc.ipus_per_replica)
        pipelined_model = to_pipelined(model, ipu_config)
        self._pipe = pipelined_model.parallelize(for_generation=True, use_cache=True).half()

    def build(self) -> int:
        logger.info(f"Whisper API build - compile to generate executable")

        if not check_available_ipus(self.req_ipus):
            logger.error(f"{self.req_ipus} IPUs not available.")
            return RESULT_SKIPPED

        logger.info(f"{self.req_ipus} IPUs available for model. Building...")
        return compile_or_load_model_exe(self._pipe, self._processor)

    def startup(self) -> int:
        logger.info("Whisper API startup - trigger load executable")

        if not check_available_ipus(self.req_ipus):
            logger.error(f"{self.req_ipus} IPUs not available.")
            return RESULT_SKIPPED

        logger.info(f"{self.req_ipus} IPUs available for model. Loading...")
        return compile_or_load_model_exe(self._pipe, self._processor)

    def request(self, params: dict, meta: dict) -> dict:
        input_file = params["file"]

        uniq_id = str(time.time())
        sound_file = audiosegment.from_file(input_file).resample(sample_rate_Hz=16000)
        audio_chunks = split_on_silence(sound_file, min_silence_len=500, silence_thresh=-40)


        i = 0
        if not os.path.exists('/tmp/dataset/'):
            os.makedirs('/tmp/dataset/')
        os.mkdir("/tmp/dataset/" + uniq_id)
        for i, chunk in enumerate(audio_chunks):
            q, mod = divmod(i,26)
            out_file = "/tmp/dataset/" + uniq_id+"/chunk{0}{1}.mp3".format(alc[q],alc[mod])
            chunk.export(out_file, format="mp3")
        disable_caching()
        ds = load_dataset("/tmp/dataset/" + uniq_id, split="train", num_proc=1)
        input_features = []
        for d in ds:
            input_features.append(self._processor(d["audio"]["array"],
                                   return_tensors="pt",
                                   sampling_rate=d['audio']['sampling_rate']).input_features.half())

        chank_number = 0
        transcription_sum = ""
        headers = {'Content-Type': 'text/text; charset=utf-8'}
        for i_f in input_features:
            chank_number += 1
            sample_output = self._pipe.generate(i_f, max_length=448, min_length=3)
            transcription = self._processor.batch_decode(sample_output, skip_special_tokens=True)[0]
            transcription_sum +=transcription + ". "
                

        result_dict = {"result": transcription_sum}
        logger.info(f"{transcription_sum}")
        return result_dict

    def shutdown(self) -> int:
        logger.info("Whisper API shutdown")
        return RESULT_OK

    def is_healthy(self) -> bool:
        logger.info("Whisper API check health")
        return True


# NOTE:
# This can be called multiple times (with separate processes)
# if running with multiple workers (replicas). Be careful that
# your application can handle multiple parallel worker processes.
# (for example, that there is no conflict for file I/O).
def create_ssf_application_instance() -> SSFApplicationInterface:
    logger.info("Create Whisper API instance")
    return Whisper2API()


class MyApplicationTest(SSFApplicationTestInterface):
    def begin(self, session, ipaddr: str) -> int:
        logger.info("MyApp test begin")
        return RESULT_OK

    def subtest(self, session, ipaddr: str, index: int) -> Tuple[bool, str, bool]:

        # Check for IPUs available for test, this affects the generated output image.
        # Default set to 8 here to keep consistency with application.
        req_ipus = 8
        if not check_available_ipus(req_ipus):
            req_ipus = 4
            if not check_available_ipus(req_ipus):
                return (False, f"Test failure: {req_ipus} IPUs not available.", False)

        logger.info(f"Running Whisper test with {req_ipus} IPUs version.")

        version = 1
        subtests = 1
        last_index = subtests - 1
        endpoint_name = "txt2img_512"

        test_input = {
            "prompt": "A large bottle of shiny blue juice",
            "guidance_scale": 9,
            "num_inference_steps": 25,
            "negative_prompt": "red",
            "random_seed": 5555,
        }

        logger.debug(
            f"MyApp test index={index} ver={version} test_input={test_input} test_expected= 'png' type image"
        )

        try:
            url = f"{ipaddr}/v{version}/{endpoint_name}"
            params = test_input

            logger.info(f"POST with params={params}")

            # Needs a long timeout as first request downloads model + loads executable.
            response = session.post(
                url, json=params, headers={"accept": "*/*"}, timeout=2700
            )

            MAGIC1 = 200
            MAGIC2 = "png"

            # Check staus code
            ok_1 = response.status_code == MAGIC1
            if not ok_1:
                logger.error(
                    f"Failed {url} with {params} : {response.status_code} v expected {MAGIC1}"
                )
            else:
                logger.info(
                    f"SD2 passed subtest check 1: expected {MAGIC1} v received {response.status_code}"
                )

            # Verify output binary value image type
            out_type = None
            try:
                # Write response to an undefined binary filetype
                with tempfile.NamedTemporaryFile(mode="wb+") as tmp:
                    tmp.write(response.content)

                    # Use Python built-in imghdr module to check type, expect 'png'
                    out_type = imghdr.what(tmp.name)

                    # Verify type
                    ok_2 = out_type == MAGIC2

            except Exception as e:
                ok_2 = False
                logger.error(f"Error with test temporary file creation/deletion: {e}")

            if not ok_2:
                logger.error(
                    f"Failed {url} with {params} : {response.content[:100]} is of type {out_type} v expected type PNG"
                )
            else:
                logger.info(
                    f"SD2 passed subtest check 2: expected {MAGIC2} v received '{out_type}'"
                )

            return (
                ok_1 == ok_2 == True,
                f"v{version} {test_input} => image type {out_type}",
                index < last_index,
            )

        except requests.exceptions.RequestException as e:
            logger.info(f"Exception {e}")
            return (False, e, False)

    def end(self, session, ipaddr: str) -> int:
        logger.info("MyApp test end")
        return RESULT_OK


def create_ssf_application_test_instance(ssf_config) -> SSFApplicationTestInterface:
    logger.info("Create MyApplication test instance")
    return MyApplicationTest()
