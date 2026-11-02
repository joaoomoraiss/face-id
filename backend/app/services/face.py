import sys
from io import BytesIO
from typing import Callable, Protocol

import numpy as np
from PIL import Image, ImageOps

from app.exceptions.face import FaceProcessingException, InvalidImageException

MAX_IMAGE_BYTES = 5 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}


class Embedder(Protocol):
    def extract_embedding(self, content: bytes, content_type: str | None) -> list[float]:
        pass


RepresentFn = Callable[..., list[dict]]


# deepface printa no console e no windows estoura se n for utf-8
def prepare_console_encoding() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        reconfigure(encoding="utf-8", errors="replace")


def deepface_represent(image: np.ndarray, *, model_name: str, detector_backend: str) -> list[dict]:
    prepare_console_encoding()
    # import aqui pra subir o app sem baixar o modelo
    from deepface import DeepFace

    return DeepFace.represent(
        img_path=image,
        model_name=model_name,
        detector_backend=detector_backend,
        enforce_detection=True,
        align=True,
    )


class FaceService:
    def __init__(
        self,
        *,
        model_name: str,
        detector_backend: str,
        represent: RepresentFn | None = None,
    ) -> None:
        self.model_name = model_name
        self.detector_backend = detector_backend
        self._represent = represent or deepface_represent

    def extract_embedding(self, content: bytes, content_type: str | None) -> list[float]:
        if not content:
            raise InvalidImageException("Arquivo de imagem ausente")
        media_type = (content_type or "").split(";", 1)[0].strip().lower()
        if media_type not in ALLOWED_CONTENT_TYPES:
            raise InvalidImageException("Tipo de imagem inválido")
        if len(content) > MAX_IMAGE_BYTES:
            raise InvalidImageException("Imagem maior que 5 MB")
        image = _to_bgr(content)  # opencv quer BGR, pillow entrega RGB
        try:
            faces = self._represent(
                image,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
            )
        except (FaceProcessingException, InvalidImageException):
            raise
        except Exception as exc:
            text = str(exc).lower()
            if "face could not be detected" in text or "no face" in text:
                raise FaceProcessingException("Nenhum rosto encontrado") from exc
            raise FaceProcessingException("Não foi possível processar o rosto") from exc
        if not faces:
            raise FaceProcessingException("Nenhum rosto encontrado")
        if len(faces) > 1:
            raise FaceProcessingException("Mais de um rosto encontrado")
        embedding = faces[0].get("embedding")
        if not embedding:
            raise FaceProcessingException("Não foi possível processar o rosto")
        return [float(value) for value in embedding]


def _to_bgr(content: bytes) -> np.ndarray:
    try:
        with Image.open(BytesIO(content)) as img:
            # celular as vezes manda a foto deitada no exif
            rgb = np.array(ImageOps.exif_transpose(img).convert("RGB"))
    except Exception as exc:
        raise FaceProcessingException("Não foi possível processar o rosto") from exc
    return rgb[:, :, ::-1].copy()
