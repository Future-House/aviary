import os
import secrets

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

auth_scheme = HTTPBearer()


def validate_token(
    credentials: HTTPAuthorizationCredentials = Depends(auth_scheme),  # noqa: B008
) -> str:
    # NOTE: don't use os.environ.get() to avoid possible empty string matches, and
    # to have clearer server failures if the AUTH_TOKEN env var isn't present
    if not secrets.compare_digest(credentials.credentials, os.environ["AUTH_TOKEN"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials
