"""
This module contains the main FastAPI application.
"""

# Import necessary libraries 
from uvicorn import Config, Server  # Uvicorn for running the FastAPI application
from fastapi import FastAPI  # FastAPI for building the API
from fastapi.middleware.cors import CORSMiddleware  # Middleware for handling CORS
from contextlib import asynccontextmanager  # Context manager for lifespan events
import multiprocessing  # For handling multiprocessing (especially on Windows)

# Import database-related functions
from app.database.faces import cleanup_face_embeddings, create_faces_table  # Database operations for faces
from app.database.images import create_image_id_mapping_table, create_images_table  # Database operations for images
from app.database.albums import create_albums_table  # Database operations for albums
from app.database.yolo_mapping import create_YOLO_mappings  # Database operations for YOLO mappings

# Import face clustering functions
from app.facecluster.init_face_cluster import get_face_cluster, init_face_cluster  

# Import route handlers
from app.routes.test import router as test_router  # Router for test endpoints
from app.routes.images import router as images_router  # Router for image endpoints
from app.routes.albums import router as albums_router  # Router for album endpoints
from app.routes.facetagging import router as tagging_router  # Router for face tagging endpoints

# Import custom logging module
from app.custom_logging import CustomizeLogger  
# this comments are added to ensure code readability

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler for the FastAPI application.
    This function runs tasks during application startup and shutdown.
    """
    # Initialize necessary database tables on application startup
    create_YOLO_mappings()  # Create YOLO mappings table in the database
    create_faces_table()  # Create the faces table
    create_image_id_mapping_table()  # Create the image ID mapping table
    create_images_table()  # Create the images table
    create_albums_table()  # Create the albums table

    # Perform necessary cleanup and initialization tasks
    cleanup_face_embeddings()  # Remove existing face embeddings from the database
    init_face_cluster()  # Initialize the face clustering model

    yield  # Yield control to FastAPI to handle requests
    # and the snext step is
    # Perform cleanup on application shutdown
    face_cluster = get_face_cluster()  # Retrieve the face clustering object
    if face_cluster:
        face_cluster.save_to_db()  # Save the updated face cluster data to the database


# Create a FastAPI application instance with a lifespan handler
app = FastAPI(lifespan=lifespan)

# Configure CORS middleware to allow frontend applications to access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (use a restricted list in production)
    allow_credentials=True,  # Allows credentials (e.g., cookies, authentication headers)
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allows all HTTP headers
)


@app.get("/")
async def root():
    """
    Root endpoint of the API.
    Returns a simple message indicating that the server is running.
    """
    return {"message": "PictoPy Server is up and running!"}


# Register route handlers for different API functionalities
app.include_router(test_router, prefix="/test", tags=["Test"])  # Routes for testing API
app.include_router(images_router, prefix="/images", tags=["Images"])  # Routes for image-related operations
app.include_router(albums_router, prefix="/albums", tags=["Albums"])  # Routes for album-related operations
app.include_router(tagging_router, prefix="/tag", tags=["Tagging"])  # Routes for face tagging operations


# Run the application when executing the script directly
if __name__ == "__main__":
    multiprocessing.freeze_support()  # Required for Windows to prevent multiprocessing issues
    
    # Setup custom logging for the application
    app.logger = CustomizeLogger.make_logger("app/logging_config.json")

    # Configure and start the Uvicorn server
    config = Config(app=app, host="0.0.0.0", port=8000, log_config=None)  
    server = Server(config)  # Create a Uvicorn server instance
    server.run()  # Start the server
