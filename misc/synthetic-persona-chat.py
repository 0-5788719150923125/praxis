"""SPC: Synthetic-Persona-Chat."""

import datasets
import pandas as pd

logger = datasets.logging.get_logger(__name__)


_CITATION = """\
@misc{jandaghi2023faithful,
      title={Faithful Persona-based Conversational Dataset Generation with Large Language Models}, 
      author={Pegah Jandaghi and XiangHai Sheng and Xinyi Bai and Jay Pujara and Hakim Sidahmed},
      year={2023},
      eprint={2312.10007},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""

_LICENSE = "This dataset is released under the CC BY 4.0 license."

_DESCRIPTION = """\
The Synthetic-Persona-Chat dataset is a persona-based conversational dataset, consisting of two parts. \
The first part, consisting of 4,723 personas and 10,906 conversations, is an extension to Persona-Chat,\
 which has the same user profile pairs as Persona-Chat but new synthetic conversations, with the same train/validation/test split as Persona-Chat.\
The second part is new synthetic personas and synthetic conversations based on that,\
 consisting of 5,648 synthetic personas and 11,001 conversations.\ 
Synthetic-Persona-Chat is created using the Generator-Critic framework introduced in\ 
Faithful Persona-based Conversational Dataset Generation with Large Language Models.\
"""


dataset_name = "Synthetic-Persona-Chat"

_URL = "https://raw.githubusercontent.com/google-research-datasets/Synthetic-Persona-Chat/main/data/"
_URLS = {
    "train": _URL + dataset_name + "_train.csv",
    "dev": _URL + dataset_name + "_valid.csv",
    "test": _URL + dataset_name + "_test.csv",
    "synth": _URL + "New-Persona-New-Conversations.csv",
}


class SPC(datasets.GeneratorBasedBuilder):
    """SPC: The Synthetic-Persona-Chat Dataset. Version 1"""

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="full",
            version=datasets.Version("1.0.0"),
            description="Plain text",
        ),
        datasets.BuilderConfig(
            name="persona-chat-compatible",
            version=datasets.Version("1.0.0"),
            description="Plain text",
        ),
    ]

    DEFAULT_CONFIG_NAME = "full"

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "user_1_persona": datasets.features.Sequence(
                        datasets.Value("string")
                    ),
                    "user_2_persona": datasets.features.Sequence(
                        datasets.Value("string")
                    ),
                    "utterances": datasets.features.Sequence(datasets.Value("string")),
                }
            ),
            citation=_CITATION,
            license=_LICENSE,
        )

    def _split_generators(self, dl_manager):
        downloaded_files = dl_manager.download_and_extract(_URLS)
        data_splits = [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": downloaded_files["train"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": downloaded_files["dev"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": downloaded_files["test"]},
            ),
        ]
        if self.config.name == "full":
            data_splits.append(
                datasets.SplitGenerator(
                    name=datasets.NamedSplit("synthetic"),
                    gen_kwargs={"filepath": downloaded_files["synth"]},
                ),
            )
        return data_splits

    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""
        logger.info("generating examples from = %s", filepath)
        key = 0

        spc_df = pd.read_csv(filepath)
        for _, row in spc_df.iterrows():
            try:
                user_1_personas = row["user 1 personas"].split("\n")
                user_2_personas = row["user 2 personas"].split("\n")
                raw_utterances = row["Best Generated Conversation"].split("\n")
                utterances = [u[7:].strip() for u in raw_utterances]
            except:
                continue
            yield key, {
                "user_1_persona": user_1_personas,
                "user_2_persona": user_2_personas,
                "utterances": utterances,
            }
            key += 1


if __name__ == "__main__":
    # Import necessary modules
    from datasets import load_dataset_builder

    # Instantiate the dataset builder with desired configuration
    # You can choose "full" or "persona-chat-compatible"
    config_name = "full"  # Change to "persona-chat-compatible" if needed
    builder = SPC(config_name=config_name)

    # Download and prepare the dataset
    print("Downloading and preparing the dataset...")
    builder.download_and_prepare()

    # Load the dataset into memory
    print("Loading the dataset into memory...")
    dataset = builder.as_dataset()

    # Display information about the dataset
    print("\nDataset splits available:")
    for split in dataset.keys():
        print(f"- {split} (Number of examples: {len(dataset[split])})")

    # Generate and display some examples from each split
    for split in dataset.keys():
        print(f"\n--- Examples from the '{split}' split ---")
        for i, example in enumerate(dataset[split]):
            print(f"\nExample {i + 1}:")
            print("User 1 Persona:")
            for persona in example["user_1_persona"]:
                print(f"  - {persona}")
            print("User 2 Persona:")
            for persona in example["user_2_persona"]:
                print(f"  - {persona}")
            print("Conversation:")
            for utterance in example["utterances"]:
                print(f"  {utterance}")
            # Limit to first 2 examples per split for brevity
            if i >= 1:
                break
