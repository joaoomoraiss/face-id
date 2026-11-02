from fastapi import HTTPException, status


class InvalidImageException(HTTPException):
    def __init__(self, string: str = "Arquivo de imagem ausente") -> None:
        super().__init__(status_code=status.HTTP_400_BAD_REQUEST, detail=string)


class FaceProcessingException(HTTPException):
    def __init__(self, string: str = "Não foi possível processar o rosto") -> None:
        super().__init__(status_code=status.HTTP_400_BAD_REQUEST, detail=string)
