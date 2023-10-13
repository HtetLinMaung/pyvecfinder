import tensorflow as tf
import tensorflow_hub as hub
import os
from flask import Flask, request, jsonify
import faiss
import numpy as np
import logging
import sys


app = Flask(__name__)
app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.setLevel(logging.DEBUG)

# Load existing index or create a new one if it doesn't exist
index_file_path = os.path.join("store", "index.faiss")
labels_file_path = os.path.join("store", "labels.txt")
# if os.path.exists(index_file_path):
#     index = faiss.read_index(index_file_path)
# else:
#     # Assuming vectors have 128 dimensions as an example
#     index = faiss.IndexFlatL2(128)


def read_labels():
    with open(labels_file_path, 'r') as file:
        return [line.strip() for line in file]


def normalize_vector(vector):
    return vector / np.linalg.norm(vector)


# Load the pre-trained ResNet model
resnet_model_url = "https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/5"
resnet_model = hub.load(resnet_model_url)


def vectorize_image(image_path):
    # Load and preprocess the image
    image = tf.keras.preprocessing.image.load_img(
        image_path, target_size=(224, 224))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.keras.applications.resnet.preprocess_input(image)
    # Expand dimensions for batch processing
    image = tf.expand_dims(image, axis=0)

    # Perform the forward pass to obtain the feature vector
    feature_vector = resnet_model(image)

    D = feature_vector.shape[1]
    return feature_vector.numpy(), D


def load_index():
    if os.path.exists(index_file_path):
        return faiss.read_index(index_file_path)
    else:
        raise FileNotFoundError(f"Index file not found: {index_file_path}")


@app.route('/vec-finder/search', methods=['POST'])
def search():
    try:
        if not os.path.exists(index_file_path):
            return jsonify({
                'code': 200,
                'message': 'Successful.',
                'data': []
            })

        image_path = request.json['image_path']

        if not os.path.exists(image_path):
            return jsonify({
                'code': 404,
                'message': 'Image file not found!',
            })

        query_vector, _ = vectorize_image(image_path)

        query_vector = np.array(query_vector, dtype='float32').reshape(
            1, -1)  # reshape if necessary

        query_vector = normalize_vector(
            query_vector)  # Normalize the query vector

        index = load_index()
        D, I = index.search(query_vector, k=10)

        labels = read_labels()
        response_data = []
        for i, idx in enumerate(I[0]):
            label = labels[idx] if idx < len(labels) else "Unknown"
            distance = float(D[0][i])
            if int(idx) >= 0:
                response_data.append({
                    'vector_id': int(idx),
                    'label': label,
                    'distance': distance
                })

        return jsonify({
            'code': 200,
            'message': 'Successful.',
            'data': response_data
        })
    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({
            'code': 500,
            'message': 'Something went wrong!'
        })


@app.route('/vec-finder/add-vector', methods=['POST'])
def add_vector():
    try:
        data = request.json
        image_path = data['image_path']

        if not os.path.exists(image_path):
            return jsonify({
                'code': 404,
                'message': 'Image file not found!',
            })
        # vector = data['vector']
        vector, D = vectorize_image(image_path)

        # You can store labels in a separate data structure or database
        label = data['label']

        vector = vector.squeeze()
        # Convert vector to a NumPy array and ensure it's a 2D array with a single row
        vector = np.array(vector, dtype='float32').reshape(1, -1)

        if not os.path.exists(index_file_path):
            index = faiss.IndexFlatIP(D)
        else:
            index = load_index()

        vector = normalize_vector(vector)  # Normalize the vector

        # Add vector to the index
        index.add(vector)

        # Save the updated index to disk
        faiss.write_index(index, index_file_path)

        # Store label in a file
        with open(labels_file_path, 'a') as file:
            file.write(f"{label}\n")

        return jsonify({
            'code': 200,
            'message': 'Vector added successfully'
        })
    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({
            'code': 500,
            'message': 'Something went wrong!'
        })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
