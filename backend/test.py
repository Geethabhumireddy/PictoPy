import sqlite3
import json
import numpy as np
from sklearn.cluster import DBSCAN
from app.config.settings import FACES_DATABASE_PATH, IMAGES_DATABASE_PATH

def get_all_face_embeddings():
    """
    Retrieves all face embeddings stored in the SQLite database 
    and maps them to their corresponding image paths.

    Returns:
        np.array: An array of all face embeddings.
        list: A list of image paths corresponding to the embeddings.
        list: A list of image paths skipped due to having more than 10 faces.
    """
    # Connect to SQLite databases
    conn_faces = sqlite3.connect(FACES_DATABASE_PATH)
    conn_images = sqlite3.connect(IMAGES_DATABASE_PATH)
    cursor_faces = conn_faces.cursor()
    cursor_images = conn_images.cursor()

    # Retrieve image IDs and their corresponding face embeddings
    cursor_faces.execute("SELECT image_id, embeddings FROM faces")
    results = cursor_faces.fetchall()

    all_embeddings = []
    image_paths = []
    skipped_images = []

    for image_id, embeddings_json in results:
        # Fetch the image path corresponding to the image ID
        cursor_images.execute("SELECT path FROM image_id_mapping WHERE id = ?", (image_id,))
        image_path = cursor_images.fetchone()[0]

        # Convert JSON-encoded embeddings into a NumPy array
        embeddings = np.array(json.loads(embeddings_json))

        # Skip images with more than 10 faces
        if len(embeddings) > 10:
            skipped_images.append(image_path)
            continue

        # Store embeddings and corresponding image paths
        all_embeddings.extend(embeddings)
        image_paths.extend([image_path] * len(embeddings))

    # Close database connections
    conn_faces.close()
    conn_images.close()

    return np.array(all_embeddings), image_paths, skipped_images

def main():
    """
    Main function that retrieves face embeddings, performs clustering using DBSCAN,
    and outputs the clustering results.
    """
    embedding_array, image_paths, skipped_images = get_all_face_embeddings()

    print(f"Shape of embeddings: {embedding_array.shape}")
    print(f"Number of images skipped (>10 faces): {len(skipped_images)}")

    # Print skipped images
    if skipped_images:
        print("Skipped images:")
        for path in skipped_images:
            print(f"  {path}")

    # Apply DBSCAN clustering using cosine similarity
    dbscan = DBSCAN(eps=0.3, min_samples=2, metric="cosine")
    cluster_labels = dbscan.fit_predict(embedding_array)

    # Group images by cluster label
    clusters = {}
    for path, label in zip(image_paths, cluster_labels):
        clusters.setdefault(label, []).append(path.split("/")[-1])

    valid_clusters = {}
    for cluster_id, image_names in clusters.items():
        # Only include clusters with more than one unique image
        if len(set(image_names)) > 1:
            valid_clusters[cluster_id] = image_names

            # Print cluster details
            print(f"\nCluster {cluster_id}:" if cluster_id != -1 else "\nOutliers:")
            for image_name in image_names:
                print(f"  {image_name}")

    # Count valid clusters excluding outliers (-1)
    n_clusters = len(valid_clusters) - (1 if -1 in valid_clusters else 0)
    print(f"\nNumber of valid clusters found: {n_clusters}")

if __name__ == "__main__":
    main()
