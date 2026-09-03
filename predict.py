
import argparse
import json
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image


def process_image(image):
    """Takes a NumPy array and returns a processed NumPy array (224, 224, 3)."""
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (224, 224))
    image /= 255.0
    return image.numpy()


def predict(image_path, model, top_k=1):
    """
    Takes an image path, a model, and top_k.
    Returns top_k probabilities and corresponding class labels.
    """
    im = Image.open(image_path)
    test_image = np.asarray(im)

    processed_image = process_image(test_image)
    expanded_image = np.expand_dims(processed_image, axis=0)

    predictions = model.predict(expanded_image)

    top_k_probs, top_k_indices = tf.math.top_k(predictions[0], k=top_k)

    top_k_probs = top_k_probs.numpy().tolist()
    # 0-indexed, no + 1 offset
    top_k_classes = [str(idx) for idx in top_k_indices.numpy()]

    return top_k_probs, top_k_classes


def main():
    # ── Argument Parser ──────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description='Predict flower class from an image using a trained Keras model.'
    )

    parser.add_argument(
        'image_path',
        type=str,
        help='Path to the input image file'
    )

    parser.add_argument(
        'model_path',
        type=str,
        help='Path to the saved Keras model (.h5 file)'
    )

    parser.add_argument(
        '--top_k',
        type=int,
        default=1,
        help='Return the top K most likely classes (default: 1)'
    )

    parser.add_argument(
        '--category_names',
        type=str,
        default=None,
        help='Path to a JSON file mapping class indices to flower names'
    )

    args = parser.parse_args()

    # ── Load Model ───────────────────────────────────────────────
    print(f'\nLoading model from: {args.model_path}')
    model = tf.keras.models.load_model(
        args.model_path,
        custom_objects={'KerasLayer': hub.KerasLayer}
    )
    print('Model loaded successfully.\n')

    # ── Load Category Names (optional) ───────────────────────────
    class_names = None
    if args.category_names:
        with open(args.category_names, 'r') as f:
            class_names = json.load(f)
        print(f'Category names loaded from: {args.category_names}\n')

    # ── Run Prediction ───────────────────────────────────────────
    probs, classes = predict(args.image_path, model, args.top_k)

    # ── Print Results ────────────────────────────────────────────
    print(f'Top {args.top_k} Predictions for: {args.image_path}')
    print('-' * 45)

    for i, (prob, cls) in enumerate(zip(probs, classes), start=1):
        if class_names:
            flower_name = class_names.get(cls, f'Class {cls}')
            print(f'{i}. {flower_name:<30} Probability: {prob:.4f} ({prob*100:.2f}%)')
        else:
            print(f'{i}. Class {cls:<25} Probability: {prob:.4f} ({prob*100:.2f}%)')


if __name__ == '__main__':
    main()