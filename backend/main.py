"""
This module contains the main FastAPI application for the PictoPy Server.
It initializes the database, configures middleware, and sets up API routes.
"""

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
    """
    This function manages the application lifecycle.
    It initializes database tables and the face clustering system on startup.
    """
    create_YOLO_mappings()  # Initialize YOLO object detection mappings for face detection
    create_faces_table()  # Create a database table to store face embeddings for recognition
    create_image_id_mapping_table()  # Create a mapping table for associating image IDs with metadata
    create_images_table()  # Create a table to store image metadata, such as paths and descriptions
    create_albums_table()  # Create a table for grouping images into albums
    cleanup_face_embeddings()  # Remove outdated or invalid face embeddings from the database
    init_face_cluster()  # Initialize the face clustering system for better organization and searchability
    
    yield  # Application runs here
    
    # Cleanup on shutdown
    face_cluster = get_face_cluster()
    if face_cluster:
        face_cluster.save_to_db()  # Save face clustering data to the database for persistence

# Initialize FastAPI application
app = FastAPI(lifespan=lifespan)

# Add CORS middleware to handle cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows requests from any origin for API accessibility
    allow_credentials=True,  # Enables sending credentials (cookies, authentication tokens, etc.)
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allows all HTTP headers for flexibility in API consumption
)

@app.get("/")
async def root():
    """Root endpoint to check if the server is running properly."""
    return {"message": "PictoPy Server is up and running!"}

# Register API route modules
app.include_router(test_router, prefix="/test", tags=["Test"])  # Test routes for debugging
app.include_router(images_router, prefix="/images", tags=["Images"])  # Routes to handle image uploads and retrieval
app.include_router(albums_router, prefix="/albums", tags=["Albums"])  # Routes for organizing images into albums
app.include_router(tagging_router, prefix="/tag", tags=["Tagging"])  # Routes for tagging images with metadata

# Entry point for running the application in production
if __name__ == "__main__":
    multiprocessing.freeze_support()  # Required for Windows compatibility to avoid multiprocessing issues
    
    # Set up custom logging for better debugging and tracking
    app.logger = CustomizeLogger.make_logger("app/logging_config.json")
    
    # Configure and run the FastAPI server
    config = Config(app=app, host="0.0.0.0", port=8000, log_config=None)  # Exposes API on port 8000
    server = Server(config)
    server.run()  # Starts the server and listens for incoming API requests

