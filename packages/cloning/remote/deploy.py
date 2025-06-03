import os

import modal
from annotator import app as annotator_app
from annotator import router as annotator_router
from dependencies import validate_token
from fastapi import Depends, FastAPI
from poly import app as poly_app
from poly import router as poly_router
from search import app as search_app
from search import router as search_router

# from cloning.clients import APP_NAME
APP_NAME = "cloning"

if modal.is_local():
    local_secret = modal.Secret.from_dict({
        "AUTH_TOKEN": os.environ["MODAL_DEPLOY_TOKEN"]
    })
else:
    local_secret = modal.Secret.from_dict({})


app = modal.App(APP_NAME)
app.include(annotator_app)
app.include(search_app)
app.include(poly_app)

web_app = FastAPI(dependencies=[Depends(validate_token)])
web_app.include_router(annotator_router)
web_app.include_router(search_router)
web_app.include_router(poly_router)


@app.function(secrets=[local_secret], allow_concurrent_inputs=32, cpu=0.5, keep_warm=1)
@modal.asgi_app()
def fastapi_app():
    return web_app
