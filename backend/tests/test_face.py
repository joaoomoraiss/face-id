import pytest

from tests.images import PNG_1X1


def _service(result):
    from app.services.face import FaceService

    def represent(image, *, model_name, detector_backend):
        if isinstance(result, Exception):
            raise result
        return result

    return FaceService(
        model_name="Facenet512",
        detector_backend="opencv",
        represent=represent,
    )


def test_console_encoding_accepts_deepface_emoji_logs():
    import io
    import sys

    from app.services.face import prepare_console_encoding

    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="cp1252", errors="strict")
    original_out, original_err = sys.stdout, sys.stderr
    sys.stdout = stream
    sys.stderr = stream
    try:
        prepare_console_encoding()
        print("\U0001f517 download")
        stream.flush()
        assert "download" in raw.getvalue().decode("utf-8")
    finally:
        sys.stdout = original_out
        sys.stderr = original_err


def test_deepface_represent_can_be_imported():
    from app.services.face import deepface_represent

    assert callable(deepface_represent)
    from deepface import DeepFace

    assert callable(DeepFace.represent)


def test_extracts_one_embedding():
    service = _service([{"embedding": [1, 0]}])
    assert service.extract_embedding(PNG_1X1, "image/png") == [1.0, 0.0]


def test_rejects_empty_file():
    from app.exceptions.face import InvalidImageException

    with pytest.raises(InvalidImageException, match="Arquivo de imagem ausente"):
        _service([]).extract_embedding(b"", "image/png")


def test_rejects_invalid_type():
    from app.exceptions.face import InvalidImageException

    with pytest.raises(InvalidImageException, match="Tipo de imagem inválido"):
        _service([]).extract_embedding(PNG_1X1, "text/plain")


def test_rejects_file_over_5mb():
    from app.exceptions.face import InvalidImageException

    content = b"a" * (5 * 1024 * 1024 + 1)
    with pytest.raises(InvalidImageException, match="Imagem maior que 5 MB"):
        _service([]).extract_embedding(content, "image/jpeg")


def test_rejects_missing_face():
    from app.exceptions.face import FaceProcessingException

    service = _service(ValueError("Face could not be detected"))
    with pytest.raises(FaceProcessingException, match="Nenhum rosto encontrado"):
        service.extract_embedding(PNG_1X1, "image/jpeg")


def test_rejects_multiple_faces():
    from app.exceptions.face import FaceProcessingException

    service = _service([{"embedding": [1.0]}, {"embedding": [0.0]}])
    with pytest.raises(FaceProcessingException, match="Mais de um rosto encontrado"):
        service.extract_embedding(PNG_1X1, "image/webp")


def test_rejects_undecodable_image():
    from app.exceptions.face import FaceProcessingException

    with pytest.raises(FaceProcessingException, match="Não foi possível processar o rosto"):
        _service([]).extract_embedding(b"not-an-image", "image/png")


def test_applies_exif_orientation_before_detection():
    from io import BytesIO

    import numpy as np
    from PIL import Image

    from app.services.face import FaceService

    source = Image.new("RGB", (4, 2), (255, 0, 0))
    exif = Image.Exif()
    exif[0x0112] = 6
    buffer = BytesIO()
    source.save(buffer, format="JPEG", exif=exif)
    seen: list[tuple[int, ...]] = []

    def represent(image, *, model_name, detector_backend):
        seen.append(np.asarray(image).shape)
        return [{"embedding": [1.0]}]

    service = FaceService(model_name="Facenet512", detector_backend="opencv", represent=represent)
    service.extract_embedding(buffer.getvalue(), "image/jpeg")
    assert seen == [(4, 2, 3)]
