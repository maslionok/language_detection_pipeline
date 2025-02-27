from transformers import PreTrainedModel, AutoModel, AutoConfig, PretrainedConfig
import floret, torch
import os, shutil
from configuration_stacked import ImpressoConfig
from transformers.modeling_utils import (
    get_parameter_device as original_get_parameter_device,
)


import torch

# Import Hugging Face dependencies
import transformers.modeling_utils
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_utils import (
    get_parameter_device as original_get_parameter_device,
)


# Custom get_parameter_device
def custom_get_parameter_device(module):
    """
    Custom get_parameter_device() to handle floret models.
    Returns 'cpu' for FloretModelWrapper, otherwise uses the original implementation.
    """
    # Check if the model is an instance of your FloretModelWrapper
    if isinstance(module, FloretModelWrapper):
        print(
            "Custom get_parameter_device(): Detected FloretModelWrapper. Returning 'cpu'."
        )
        return torch.device("cpu")

    # Otherwise, fall back to Hugging Face's original implementation
    return original_get_parameter_device(module)


# Custom device property
@property
def custom_device(self) -> torch.device:
    """
    Custom device() method to handle floret models.
    Always returns torch.device('cpu') for FloretModelWrapper.
    """
    # Check if the model is an instance of your FloretModelWrapper
    if isinstance(self, FloretModelWrapper):
        print(
            "Custom device(): Detected FloretModelWrapper. Returning torch.device('cpu')."
        )
        return torch.device("cpu")

    # Otherwise, fall back to Hugging Face's original implementation
    return torch.device("cpu")  # original_device.__get__(self, type(self))


# Monkey-patch get_parameter_device and device property
transformers.modeling_utils.get_parameter_device = custom_get_parameter_device
PreTrainedModel.device = custom_device

print("Monkey-patch applied: get_parameter_device and device property")

# logger = logging.getLogger(__name__)

original_device = PreTrainedModel.device


def get_info(label_map):
    num_token_labels_dict = {task: len(labels) for task, labels in label_map.items()}
    return num_token_labels_dict


class FloretModelWrapper:
    """
    Wrapper for floret model to make it compatible with Hugging Face pipeline.
    Mocks the .device attribute and passes predict() unchanged.
    """

    def __init__(self, floret_model):
        self.floret_model = floret_model

        # Mocking the .device attribute to make Hugging Face happy
        self.device = torch.device("cpu")  # floret is always on CPU

    def predict(self, text, k=1):
        """
        Pass-through for floret's predict() method.
        """
        return self.floret_model.predict(text, k=k)


class ExtendedMultitaskModelForTokenClassification(PreTrainedModel):

    config_class = ImpressoConfig
    # Monkey-patch get_parameter_device

    def __init__(self, config: PretrainedConfig, *inputs, **kwargs):
        super().__init__(config)
        self.config = config
        print("Doest is it even pass through here?")
        print(
            f"The config in ExtendedMultitaskModelForTokenClassification is: {self.config}"
        )
        # self.model = floret.load_model(self.config.filename)

    def predict(self, text, k=1):
        predictions = self.model.predict(text, k)
        return predictions

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        print("Calling from_pretrained...")

        # Initialize model with config
        model = cls(ImpressoConfig())

        # Load model using floret
        print(f"---Loading model from: {model.config.filename}")
        floret_model = floret.load_model(model.config.filename)

        # Wrap the model to fake .device attribute
        model.model = FloretModelWrapper(floret_model)

        print(model.model, "device:", model.model.device)

        print(f"Model loaded and wrapped from: {model.config.filename}")

        return model

    def save_pretrained(self, save_directory, *args, **kwargs):
        # Ignore Hugging Face-specific arguments
        max_shard_size = kwargs.pop("max_shard_size", None)
        safe_serialization = kwargs.pop("safe_serialization", False)

        # Ensure directory exists
        os.makedirs(save_directory, exist_ok=True)

        # Save the model file
        model_file = os.path.join(save_directory, "LID-40-3-2000000-1-4.bin")
        shutil.copy(self.config.filename, model_file)

        # Save the config file
        config_file = os.path.join(save_directory, "config.json")
        self.config.save_pretrained(save_directory)

        print(f"Model saved to: {save_directory}")

    def get_parameter_device(module):
        """
        Custom get_parameter_device() to handle floret models.
        Returns 'cpu' for floret models, and falls back to the original method otherwise.
        """
        # Check if the model is an instance of your FloretModelWrapper
        if isinstance(module, FloretModelWrapper):
            print(
                "Custom get_parameter_device(): Detected FloretModelWrapper. Returning 'cpu'."
            )
            return "cpu"

        # Otherwise, fall back to Hugging Face's original implementation
        return original_get_parameter_device(module)
