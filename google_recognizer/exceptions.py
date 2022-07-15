class RequestError(Exception):
    """base class for request errors"""


class BadRequestError(RequestError):
    status_code = 400


class NotFoundError(RequestError):
    status_code = 404


class RecognitionError(Exception):
    """Raised when google response is null"""
    status_code = 200
