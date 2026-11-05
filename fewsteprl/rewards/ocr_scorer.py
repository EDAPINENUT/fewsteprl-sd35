"""
OCR-based text rendering accuracy scorer.

Uses PaddleOCR to recognize text in generated images and compares against
the target text extracted from the prompt (text between quotes).

Adapted from DiffusionNFT's OCR reward.

PaddleOCR 3.x (PaddleX API)
---------------------------
This module targets **paddleocr 3.x** with PP-OCRv5 models:
  - models load offline from local PaddleX inference dirs
    (``<root>/<model_name>/inference.*``, see :func:`_resolve_ocr_model_root`)
    instead of a cache root that auto-downloads;
  - inference is ``.predict(img)`` returning per-image dicts with a
    ``rec_texts`` list (see :func:`_ocr_texts`);
  - the auto-download sub-modules (doc orientation / unwarping / textline
    orientation) must be disabled explicitly or construction tries to fetch
    them.

Reward per image: spaces are removed and both texts lowercased; the distance
is 0 if the target is a substring of the recognized text, otherwise their
Levenshtein distance clamped to ``len(target)``; reward =
``1 - distance / len(target)``.
"""

import os

# Disable Paddle's oneDNN backend and cap MKL threads before importing
# paddleocr. The default oneDNN ElementwiseKernel + CPUAllocator path is not
# thread-safe under high contention (several ranks x paddle threads x
# dataloader workers sharing the address space) and can SIGSEGV
# non-deterministically in AnalysisPredictor::ZeroCopyRun during long runs.
#
# FLAGS_use_mkldnn is set unconditionally (not setdefault) so it always wins.
# MKL_NUM_THREADS uses setdefault and OMP_NUM_THREADS is not touched, so the
# launcher's thread settings are respected: overriding them here would
# throttle the whole training process, not just OCR.
os.environ["FLAGS_use_mkldnn"] = "false"
os.environ.setdefault("MKL_NUM_THREADS", "1")

import time
import fcntl
import logging
import signal
import threading
from pathlib import Path
from contextlib import contextmanager
import torch
import numpy as np
from typing import List, Union
from PIL import Image
from Levenshtein import distance
from fewsteprl.rewards import reward_ckpt_path

logger = logging.getLogger(__name__)

# Recognition model. Only the English rec model is needed (prompts are the
# quoted-text OCR set).
_REC_MODEL_NAME = "en_PP-OCRv5_mobile_rec"

# Detection model: prefer mobile (~5 MB, fast on CPU). The server variant
# (~85 MB) is ~10x slower per image on CPU; with a full eval sweep per
# interval that alone can push a rank past the NCCL collective timeout.
_DET_MODEL_CANDIDATES = ("PP-OCRv5_mobile_det", "PP-OCRv5_server_det")

# Per-image OCR timeout in seconds (FEWSTEPRL_OCR_TIMEOUT_S; 0 disables it).
# A single pathological early-training image can keep paddle's det/rec busy
# for minutes; on a multi-rank job that stalls one rank past the NCCL
# collective timeout and SIGABRTs the whole run. Each image is hard-capped via
# SIGALRM; on timeout it takes the max penalty (same as an OCR failure) so
# training keeps moving and ranks stay in lock-step.
_OCR_TIMEOUT_S = int(os.environ.get("FEWSTEPRL_OCR_TIMEOUT_S", "20"))


class _OcrTimeout(Exception):
    pass


def _alarm_handler(signum, frame):  # noqa: ARG001 - signal handler signature
    raise _OcrTimeout()


class OcrScorer:
    def __init__(self, use_gpu: bool = False, init_retries: int = 3):
        """
        OCR reward calculator (PaddleOCR 3.x, local models, offline).

        :param use_gpu: ignored; PaddleOCR 3.x selects the device itself and
            takes no ``use_gpu`` flag.
        :param init_retries: retries for the rare concurrent-init race.
        """
        model_root = _resolve_ocr_model_root()
        lock_path = os.path.join(
            _resolve_ocr_cache_root(), ".fewsteprl_paddleocr_init.lock",
        )
        os.makedirs(os.path.dirname(lock_path), exist_ok=True)
        self.ocr = None
        self._timeout = max(0, _OCR_TIMEOUT_S)

        for attempt_idx in range(init_retries):
            try:
                # Prefer parallel init across ranks; only synchronize on rare
                # temporary-file races.
                self.ocr = _init_paddleocr(model_root=model_root)
                break
            except FileNotFoundError as e:
                if ".tmp" in str(e) and attempt_idx + 1 < init_retries:
                    with _file_lock(lock_path):
                        time.sleep(0.5)
                    time.sleep(1.5 * (attempt_idx + 1))
                    continue
                raise

        if self.ocr is None:
            raise RuntimeError("Failed to initialize PaddleOCR scorer")

    def _reward_one(self, img, prompt: str) -> float:
        """OCR one image and compute its text-match reward.

        Wrapped in a SIGALRM timeout so one slow image cannot stall this rank
        (and thereby time out NCCL on the others). A timeout or OCR error gets
        the maximum penalty (reward 0).
        """
        if isinstance(img, Image.Image):
            img = np.array(img)
        armed = False
        try:
            # signal.alarm only works on the main thread; eval may call the
            # scorer from a worker thread, in which case we skip the cap.
            if self._timeout > 0 and threading.current_thread() is threading.main_thread():
                signal.signal(signal.SIGALRM, _alarm_handler)
                signal.alarm(self._timeout)
                armed = True
            recognized_text = _ocr_texts(self.ocr, img)
            recognized_text = recognized_text.replace(" ", "").lower()
            prompt = prompt.replace(" ", "").lower()
            if prompt in recognized_text:
                dist = 0
            else:
                dist = distance(recognized_text, prompt)
            if dist > len(prompt):
                dist = len(prompt)
        except _OcrTimeout:
            print(f"OCR timeout (>{self._timeout}s), penalizing this image")
            dist = len(prompt)
        except Exception as e:  # noqa: BLE001 - a bad image must not kill eval
            print(f"OCR processing failed: {str(e)}")
            dist = len(prompt)  # Maximum penalty
        finally:
            if armed:
                signal.alarm(0)
        return 1 - dist / (len(prompt))

    @torch.no_grad()
    def __call__(self, images: Union[List[Image.Image], List[np.ndarray]], prompts: List[str]) -> list:
        """
        Calculate OCR reward.
        :param images: List of input images (PIL or numpy format)
        :param prompts: Corresponding prompts; the target text is the first
            double-quoted span of each prompt
        :return: List of reward scores
        """
        prompts = [prompt.split('"')[1] for prompt in prompts]
        assert len(images) == len(prompts), "Images and prompts must have the same length"
        return [self._reward_one(img, p) for img, p in zip(images, prompts)]


def _resolve_ocr_model_root() -> str:
    """Resolve the dir holding the PaddleOCR 3.x model dirs.

    Priority:
    1) FEWSTEPRL_PADDLEOCR_V5_DIR (explicit override)
    2) <repo_root>/models/paddleocr_v5
    3) <reward_ckpt_path.CKPT_PATH>/paddleocr_v5

    Each candidate must contain at least the recognition model dir with
    ``inference.*`` files, otherwise we keep looking. Failing loudly here is
    deliberate: PaddleOCR 3.x would otherwise try to download the models,
    which on an offline machine surfaces as a confusing network error deep
    inside PaddleX.
    """
    explicit = os.environ.get("FEWSTEPRL_PADDLEOCR_V5_DIR", "").strip()
    candidates = []
    if explicit:
        candidates.append(os.path.expanduser(explicit))
    repo_root = Path(__file__).resolve().parents[2]
    candidates.append(str(repo_root / "models" / "paddleocr_v5"))
    ckpt_root = str(getattr(reward_ckpt_path, "CKPT_PATH", "") or "").strip()
    if ckpt_root:
        candidates.append(os.path.join(os.path.expanduser(ckpt_root), "paddleocr_v5"))

    for candidate in candidates:
        if _has_inference_files(os.path.join(candidate, _REC_MODEL_NAME)):
            return candidate

    raise FileNotFoundError(
        "PaddleOCR 3.x model dirs not found. Looked for "
        f"'{_REC_MODEL_NAME}/inference.*' under: {candidates}. "
        "Point FEWSTEPRL_PADDLEOCR_V5_DIR at a dir containing the "
        f"extracted PaddleX inference dirs ({_REC_MODEL_NAME} and one of "
        f"{list(_DET_MODEL_CANDIDATES)}), or symlink them into "
        "models/paddleocr_v5/."
    )


def _has_inference_files(model_dir: str) -> bool:
    """True if ``model_dir`` looks like an extracted PaddleX inference dir."""
    p = Path(model_dir)
    if not p.is_dir():
        return False
    return any(child.name.startswith("inference.") for child in p.iterdir())


def _resolve_det_model(model_root: str) -> str:
    """Pick the detection model, preferring the fast mobile variant."""
    for name in _DET_MODEL_CANDIDATES:
        if _has_inference_files(os.path.join(model_root, name)):
            return name
    raise FileNotFoundError(
        f"No PaddleOCR detection model found under {model_root}; "
        f"expected one of {list(_DET_MODEL_CANDIDATES)}."
    )


def _ocr_texts(ocr, img) -> str:
    """Run PaddleOCR 3.x ``.predict()`` and join all recognized text.

    ``.predict()`` returns a list of OCRResult dicts (one per input image);
    each has a ``rec_texts`` list.
    """
    result = ocr.predict(img)
    if not result:
        return ""
    res0 = result[0]
    texts = res0.get("rec_texts", []) if hasattr(res0, "get") else []
    return "".join(texts)


def _resolve_ocr_cache_root() -> str:
    """Resolve a writable dir for the init lock file.

    Priority:
    1) FEWSTEPRL_PADDLEOCR_HOME
    2) <reward_ckpt_path.CKPT_PATH>/.paddleocr
    3) <repo_root>/.paddleocr (fallback)

    Only the concurrent-init lock lives here; the models come from
    :func:`_resolve_ocr_model_root`.
    """
    explicit = os.environ.get("FEWSTEPRL_PADDLEOCR_HOME", "").strip()
    if explicit:
        return os.path.expanduser(explicit)

    ckpt_root = str(getattr(reward_ckpt_path, "CKPT_PATH", "") or "").strip()
    if ckpt_root:
        return os.path.join(os.path.expanduser(ckpt_root), ".paddleocr")

    return str(Path(__file__).resolve().parents[2] / ".paddleocr")


def _init_paddleocr(model_root: str):
    """Construct a PaddleOCR 3.x pipeline from local model dirs (offline).

    Every auto-download sub-module (doc orientation / unwarping / textline
    orientation) is disabled: only detection + recognition are needed for the
    text-match reward, and leaving them on makes construction try to download
    their weights.
    """
    # PaddleOCR mutates root logger level during import/init.
    # Preserve the fewsteprl logging verbosity (INFO on rank 0).
    with _preserve_root_logger_state():
        from paddleocr import PaddleOCR

        det_name = _resolve_det_model(model_root)
        logger.info(
            "[OcrScorer] paddleocr 3.x: det=%s rec=%s root=%s",
            det_name, _REC_MODEL_NAME, model_root,
        )
        return PaddleOCR(
            lang="en",
            text_detection_model_name=det_name,
            text_detection_model_dir=os.path.join(model_root, det_name),
            text_recognition_model_name=_REC_MODEL_NAME,
            text_recognition_model_dir=os.path.join(model_root, _REC_MODEL_NAME),
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            # paddle 3.3.1's PIR executor crashes inside the oneDNN/MKLDNN CPU
            # backend on these OCR models ("ConvertPirAttribute2RuntimeAttribute
            # not support [pir::ArrayAttribute<pir::DoubleAttribute>]"); the
            # plain CPU backend runs them correctly.
            enable_mkldnn=False,
            # Bound worst-case cost on noisy early-training images: cap the det
            # input side (fewer pixels) and raise the box threshold (fewer
            # spurious boxes -> fewer rec calls). Eval images are 512px so 960
            # never upscales.
            text_det_limit_side_len=960,
            text_det_limit_type="max",
            text_det_box_thresh=0.6,
        )


@contextmanager
def _preserve_root_logger_state():
    root = logging.getLogger()
    level = root.level
    disabled = root.disabled
    handler_levels = [(handler, handler.level) for handler in list(root.handlers)]
    try:
        yield
    finally:
        root.setLevel(level)
        root.disabled = disabled
        for handler, handler_level in handler_levels:
            handler.setLevel(handler_level)


@contextmanager
def _file_lock(lock_path: str):
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o666)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)
