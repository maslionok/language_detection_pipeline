import os
import shutil
import argparse
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
)
from huggingface_hub import HfApi, Repository
from transformers.pipelines import PIPELINE_REGISTRY

# import json
from configuration_stacked import ImpressoConfig
from modeling_stacked import ExtendedMultitaskModelForTokenClassification
import subprocess
from lang_ident import LangIdentPipeline


def get_latest_checkpoint(checkpoint_dir):
    checkpoints = [
        d
        for d in os.listdir(checkpoint_dir)
        if os.path.isdir(os.path.join(checkpoint_dir, d))
        and d.startswith("checkpoint-")
    ]
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]), reverse=True)
    return os.path.join(checkpoint_dir, checkpoints[0])


def get_info(label_map):
    num_token_labels_dict = {task: len(labels) for task, labels in label_map.items()}
    return num_token_labels_dict


def push_model_to_hub(checkpoint_dir, repo_name):
    # checkpoint_path = get_latest_checkpoint(checkpoint_dir)
    checkpoint_path = checkpoint_dir
    config = ImpressoConfig.from_pretrained(checkpoint_path)
    print(config)

    config.pretrained_config = ImpressoConfig.from_pretrained(config.filename)
    config.save_pretrained("floret")
    config = ImpressoConfig.from_pretrained("floret")
    PIPELINE_REGISTRY.register_pipeline(
        "lang-ident",
        pipeline_class=LangIdentPipeline,
        pt_model=ExtendedMultitaskModelForTokenClassification,
    )

    # PIPELINE_REGISTRY.register_pipeline(
    #     "pair-classification",
    #     pipeline_class=PairClassificationPipeline,
    #     pt_model=AutoModelForSequenceClassification,
    #     tf_model=TFAutoModelForSequenceClassification,
    # )

    config.custom_pipelines = {
        "lang-ident": {
            "impl": "lang_ident.LangIdentPipeline",
            "pt": ["AutoModelForSequenceClassification"],
            "tf": [],
        }
    }
    model = ExtendedMultitaskModelForTokenClassification.from_pretrained(
        checkpoint_path, config=config
    )

    local_repo_path = "lang-detect"
    repo_url = HfApi().create_repo(repo_id=repo_name, exist_ok=True)
    repo = Repository(local_dir=local_repo_path, clone_from=repo_url)

    try:
        # Try to pull the latest changes from the remote repository using subprocess
        subprocess.run(["git", "pull"], check=True, cwd=local_repo_path)
    except subprocess.CalledProcessError as e:
        # If fast-forward is not possible, reset the local branch to match the remote branch
        subprocess.run(
            ["git", "reset", "--hard", "origin/main"],
            check=True,
            cwd=local_repo_path,
        )

    # Copy all Python files to the local repository directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    for filename in os.listdir(current_dir):
        if filename.endswith(".py") or filename.endswith(".json"):
            shutil.copy(
                os.path.join(current_dir, filename),
                os.path.join(local_repo_path, filename),
            )

    ImpressoConfig.register_for_auto_class()

    AutoConfig.register("floret", ImpressoConfig)
    AutoModelForSequenceClassification.register(
        ImpressoConfig, ExtendedMultitaskModelForTokenClassification
    )
    ExtendedMultitaskModelForTokenClassification.register_for_auto_class(
        "AutoModelForSequenceClassification"
    )
    # model.save_pretrained(local_repo_path)

    from transformers import AutoModelForTokenClassification, AutoTokenizer
    from transformers import pipeline

    # Define the model name to be used for token classification, we use the Impresso NER
    # that can be found at "https://huggingface.co/impresso-project/ner-stacked-bert-multilingual"
    MODEL_NAME = "Maslionok/lang-detect"
    #

    # # Add, commit and push the changes to the repository
    subprocess.run(["git", "add", "."], check=True, cwd=local_repo_path)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit including model and configuration"],
        check=True,
        cwd=local_repo_path,
    )
    subprocess.run(["git", "push"], check=True, cwd=local_repo_path)
    #
    # Push the model to the hub (this includes the README template)
    model.push_to_hub(repo_name)

    lang_pipeline = pipeline(
        "lang-ident", model=MODEL_NAME, trust_remote_code=True, device="cpu"
    )
    lang_pipeline.push_to_hub(MODEL_NAME)
    sentence = "En l'an 1348, au plus fort des ravages de la peste noire à travers l'Europe, le Royaume de France se trouvait à la fois au bord du désespoir et face à une opportunité. À la cour du roi Philippe VI, les murs du Louvre étaient animés par les rapports sombres venus de Paris et des villes environnantes. La peste ne montrait aucun signe de répit, et le chancelier Guillaume de Nogaret, le conseiller le plus fidèle du roi, portait le lourd fardeau de gérer la survie du royaume."
    #
    print(lang_pipeline(sentence))
    # lang_pipeline.push_to_hub(MODEL_NAME)
    print(f"Model and repo pushed to: {repo_url}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Push NER model to Hugging Face Hub")
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        help="Type of the model (e.g., langident)",
    )
    parser.add_argument(
        "--language",
        type=str,
        required=True,
        help="Language of the model (e.g., multilingual)",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        default="LID-40-3-2000000-1-4.bin",
        help="Directory containing checkpoint folders",
    )
    args = parser.parse_args()
    repo_name = f"Maslionok/lang-detect"
    push_model_to_hub(args.checkpoint_dir, repo_name)

    # PIPELINE_REGISTRY.register_pipeline(
    #     "generic-ner",
    #     pipeline_class=MultitaskTokenClassificationPipeline,
    #     pt_model=ExtendedMultitaskModelForTokenClassification,
    # )
    # model.config.custom_pipelines = {
    #     "generic-ner": {
    #         "impl": "generic_ner.MultitaskTokenClassificationPipeline",
    #         "pt": ["ExtendedMultitaskModelForTokenClassification"],
    #         "tf": [],
    #     }
    # }
    # classifier = pipeline(
    #     "generic-ner", model=model, tokenizer=tokenizer, label_map=label_map
    # )
    # from pprint import pprint
    #
    # pprint(
    #     classifier(
    #         "1. Le public est averti que Charlotte née Bourgoin, femme-de Joseph Digiez, et Maurice Bourgoin, enfant mineur représenté par le sieur Jaques Charles Gicot son curateur, ont été admis par arrêt du Conseil d'Etat du 5 décembre 1797, à solliciter une renonciation générale et absolue aux biens et aux dettes présentes et futures de Jean-Baptiste Bourgoin leur père."
    #     )
    # )
    # repo.push_to_hub(commit_message="Initial commit of the trained NER model with code")
