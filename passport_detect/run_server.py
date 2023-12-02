import hydra
import numpy as np
import onnxruntime as rt
import torch
from dvc.fs import DVCFileSystem
from flask import Flask, request
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../configs", config_name="run")
def run_server(cfg: DictConfig):
    fs = DVCFileSystem()
    fs.get_file(cfg.model.path, cfg.model.path)

    app = Flask(__name__)
    session = rt.InferenceSession(
        cfg.model.path, providers=rt.get_available_providers()
    )
    inputs_name = session.get_inputs()[0].name

    @app.post("/predict")
    def predict():
        data = request.json["inputs"]
        answers = []
        for x in data:
            inputs = torch.from_numpy(np.array([x])).float()
            try:
                outputs = session.run(None, {inputs_name: inputs.numpy()})[0]
                answers.append(outputs[0].tolist())
            except Exception as e:
                return {"error": str(e)}, 400
        return {"outputs": answers}, 200

    app.run(host="0.0.0.0", port=5000)


if __name__ == "__main__":
    run_server()
