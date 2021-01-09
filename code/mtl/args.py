from dataclasses import dataclass, field
from typing import Optional

from mtl.utils import task_id_to_name


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default='/root/paddlejob/workspace/train_data/datasets/data67127',
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    freeze: bool = field(
        default=False,
        metadata={"help": "Whether to bert model or not."},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """
    tokenizer_dir: Optional[str] = field(
        default='/root/paddlejob/workspace/train_data/datasets/data67127',
        metadata={"help": "raw data directory"},
    )
    data_dir: Optional[str] = field(
        default='/root/paddlejob/workspace/train_data/datasets/data67230',
        metadata={"help": "raw data directory"},
    )
    data_save_dir: Optional[str] = field(
        default='./data/processed',
        metadata={"help": "processed data save directory"},
    )
    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_id_to_name.values())},
    )
    max_seq_length: int = field(
        default=300,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=True, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    debug: bool = field(
        default=False, metadata={"help": "Whether to use debug mode."}
    )
    pseudo: bool = field(
        default=True, metadata={"help": "Whether to use debug mode."}
    )
    test_b: bool = field(
        default=False, metadata={"help": "Whether is test b stage."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    train_val_split_ratio: float = field(
        default=0.1,
        metadata={
            "help": "num of examples, val: train"
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
