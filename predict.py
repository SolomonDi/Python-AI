from PIL import Image

from inference import Predictor


def predict(image_path, predictor=None):
    predictor = predictor or Predictor()
    prob = predictor.predict(Image.open(image_path))

    diagnosis = "PNEUMONIA" if prob > predictor.threshold else "NORMAL"
    print(f"File: {image_path}")
    print(f"Pneumonia probability: {prob:.4f}")
    print(f"Diagnosis (threshold {predictor.threshold:.4f}): {diagnosis}")


if __name__ == "__main__":
    import sys
    predictor = Predictor()
    for path in sys.argv[1:]:
        predict(path, predictor)
