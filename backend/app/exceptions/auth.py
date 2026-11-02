from fastapi import HTTPException, status


# login: email desconhecido e rosto errado usam a mesma, de proposito
class InvalidCredentialsException(HTTPException):
    def __init__(self, string: str = "Credenciais inválidas") -> None:
        super().__init__(status_code=status.HTTP_401_UNAUTHORIZED, detail=string)


class UnauthenticatedException(HTTPException):
    def __init__(self, string: str = "Não autenticado") -> None:
        super().__init__(status_code=status.HTTP_401_UNAUTHORIZED, detail=string)


class ForbiddenException(HTTPException):
    def __init__(self, string: str = "Permissão insuficiente") -> None:
        super().__init__(status_code=status.HTTP_403_FORBIDDEN, detail=string)
