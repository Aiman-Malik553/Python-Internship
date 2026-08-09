import mlflow
import mlflow.pyfunc
import os
import json

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

mlflow.set_experiment("BLIP_Model_Registry")


class BLIPConfigModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        with open(context.artifacts["config"], "r") as f:
            self.config = json.load(f)

    def predict(self, context, model_input):
        return self.config


with mlflow.start_run(run_name="BLIP_Baseline_v1"):
    mlflow.log_param("model_name", "Salesforce/blip-image-captioning-base")
    mlflow.log_param("num_beams", 5)
    mlflow.log_param("max_new_tokens", 30)
    mlflow.log_metric("BLEU", 0.1919)
    mlflow.log_metric("ROUGE_1", 0.5556)
    mlflow.log_metric("ROUGE_2", 0.3198)
    mlflow.log_metric("ROUGE_L", 0.5337)

    mlflow.pyfunc.log_model(
        name="model",
        python_model=BLIPConfigModel(),
        artifacts={"config": "model_artifact.json"},
        registered_model_name="BLIP_Image_Captioning_Baseline"
    )

print("Model registered successfully as version in MLflow Registry!")