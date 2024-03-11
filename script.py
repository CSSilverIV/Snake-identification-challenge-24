import pandas as pd
import numpy as np
import onnxruntime as ort
import os
from tqdm import tqdm
import timm
import torchvision.transforms as T
from PIL import Image
import torch

def is_gpu_available():
    """Check if the python package `onnxruntime-gpu` is installed."""
    return torch.cuda.is_available()


class PytorchWorker:
    """Run inference using ONNX runtime."""

    def __init__(self, onnx_path: str):
        print("Setting up Pytorch Model")

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        print(f"Using devide: {self.device}")
        self.model = timm.create_model("hf-hub:BVRA/tf_efficientnet_b3.in1k_ft_df20_224", pretrained=True)
        self.model = self.model.eval()
        self.model.to(self.device)

        self.transforms = T.Compose([T.Resize((224, 224)),
                                     T.ToTensor(),
                                     T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    def predict_image(self, image: np.ndarray) -> list():
        """Run inference using ONNX runtime.

        :param image: Input image as numpy array.
        :return: A list with logits and confidences.
        """

        logits = self.model(self.transforms(image).unsqueeze(0))

        return logits.tolist()


def make_submission(test_metadata, model_path, output_csv_path="./submission.csv", images_root_path="/tmp/data/private_testset"):
    """Make submission with given """

    model = PytorchWorker(model_path)

    predictions = []

    for _, row in tqdm(test_metadata.iterrows(), total=len(test_metadata)):
        image_path = os.path.join(images_root_path, row.image_path)

        test_image = Image.open(image_path).convert("RGB")

        logits = model.predict_image(test_image)

        predictions.append(np.argmax(logits))

    test_metadata["class_id"] = predictions

    user_pred_df = test_metadata.drop_duplicates("observation_id", keep="first")
    user_pred_df[["observation_id", "class_id"]].to_csv(output_csv_path, index=None)


if __name__ == "__main__":

    import zipfile

    with zipfile.ZipFile("/tmp/data/private_testset.zip", 'r') as zip_ref:
        zip_ref.extractall("/tmp/data")

    HFHUB_MODEL_PATH = "hf-hub:BVRA/tf_efficientnet_b3.in1k_ft_df20_224"

    metadata_file_path = "./FungiCLEF2024_TestMetadata.csv"
    test_metadata = pd.read_csv(metadata_file_path)

    make_submission(
        test_metadata=test_metadata,
        model_path=HFHUB_MODEL_PATH,
    )
