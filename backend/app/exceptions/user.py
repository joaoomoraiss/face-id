from fastapi import HTTPException, status


class UserNotFoundException(HTTPException):
    def __init__(self, string: str = "Usuário não encontrado") -> None:
        super().__init__(status_code=status.HTTP_404_NOT_FOUND, detail=string)


# email ja cadastrado, no register e tbm no update
class UserAlreadyExistsException(HTTPException):
    def __init__(self, string: str = "E-mail já cadastrado") -> None:
        super().__init__(status_code=status.HTTP_409_CONFLICT, detail=string)


class InvalidFieldException(HTTPException):
    def __init__(self, string: str = "Campo inválido") -> None:
        super().__init__(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=string)
