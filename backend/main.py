from uvicorn import Config, Server
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import multiprocessing

from app.database.faces import cleanup_face_embeddings, create_faces_table
from app.database.images import create_image_id_mapping_table, create_images_table
from app.database.albums import create_albums_table
from app.database.yolo_mapping import create_YOLO_mappings
from app.facecluster.init_face_cluster import get_face_cluster, init_face_cluster
from app.routes.test import router as test_router
from app.routes.images import router as images_router
from app.routes.albums import router as albums_router
from app.routes.facetagging import router as tagging_router
from app.custom_logging import CustomizeLogger

@asynccontextmanager
async def lifespan(app: FastAPI):
    create_YOLO_mappings()
    create_faces_table()
    create_image_id_mapping_table()
    create_images_table()
    create_albums_table()
    cleanup_face_embeddings()
    init_face_cluster()
    
    yield  
    
    face_cluster = get_face_cluster()
    if face_cluster:
        face_cluster.save_to_db()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "PictoPy Server is up and running!"}

app.include_router(test_router, prefix="/test", tags=["Test"])
app.include_router(images_router, prefix="/images", tags=["Images"])
app.include_router(albums_router, prefix="/albums", tags=["Albums"])
app.include_router(tagging_router, prefix="/tag", tags=["Tagging"])

if __name__ == "__main__":
    multiprocessing.freeze_support()
    app.logger = CustomizeLogger.make_logger("app/logging_config.json")
    config = Config(app=app, host="0.0.0.0", port=8000, log_config=None)
    server = Server(config)
    server.run()
