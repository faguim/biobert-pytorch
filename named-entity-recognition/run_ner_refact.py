# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Fine-tuning the library models for named entity recognition on CoNLL-2003. """


import logging
import os
import sys
import pdb
import subprocess

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from seqeval.metrics import f1_score, precision_score, recall_score
from torch import nn

from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoModel,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from utils_ner_refact import NerDataset, Split, get_labels

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    use_fast: bool = field(default=False, metadata={"help": "Set this flag to use fast tokenization."})
    # If you want to tweak more attributes on your tokenizer, you should do it in a distinct script,
    # or just modify its tokenizer_config.json.
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    data_dir: str = field(
        metadata={"help": "The input data dir. Should contain the .txt files for a CoNLL-2003-formatted task."}
    )
    labels: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a file containing all labels. If not specified, CoNLL-2003 labels are used."},
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

    # per_device_eval_batch_size: int = field(
    #     default=4, metadata={"help": "Overwrite the cached training and evaluation sets"}
    # )
    block_size: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )


def check_overwrite_output_dir(training_args):
    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome.")


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if check_overwrite_output_dir(training_args):
        print()



    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
        # training_args.per_device_eval_batch_size
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    # Prepare CONLL-2003 task
    print(data_args.labels)
    labels = get_labels(data_args.labels)

    print(labels)

    label_map: Dict[int, str] = {i: label for i, label in enumerate(labels)}
    print(label_map)
    num_labels = len(labels)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        id2label=label_map,
        label2id={label: i for i, label in enumerate(labels)},
        cache_dir=model_args.cache_dir,
        # block_size=128
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast,
        do_lower_case=False
    )
    model = AutoModelForTokenClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )


    # Get datasets
    train_dataset = (
        NerDataset(
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            labels=labels,
            model_type=config.model_type,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.train,
        )
        if training_args.do_train
        else None
    )
    eval_dataset = (
        NerDataset(
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            labels=labels,
            model_type=config.model_type,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.dev,
        )
        if training_args.do_eval
        else None
    )

    def align_predictions(predictions: np.ndarray, label_ids: np.ndarray) -> Tuple[List[int], List[int]]:
        preds = np.argmax(predictions, axis=2)
        # print('segundo soft ', preds)
        batch_size, seq_len = preds.shape

        out_label_list = [[] for _ in range(batch_size)]
        preds_list = [[] for _ in range(batch_size)]

        for i in range(batch_size):
            for j in range(seq_len):
                # print(str('label_ids['+ str(i)+', '+ str(j) +'] '), label_ids[i, j])
                # print(nn.CrossEntropyLoss().ignore_index)
                if label_ids[i, j] != nn.CrossEntropyLoss().ignore_index:
                    out_label_list[i].append(label_map[label_ids[i][j]])
                    # print(label_map[label_ids[i][j]])
                    # print('--------------------------------------------------preds')

                    # print(preds[i][j])
                    preds_list[i].append(label_map[preds[i][j]])
                else: out_label_list[i].append('O')

        return preds_list, out_label_list

    def compute_metrics(p: EvalPrediction) -> Dict:
        preds_list, out_label_list = align_predictions(p.predictions, p.label_ids)

        return {
            "precision": precision_score(out_label_list, preds_list),
            "recall": recall_score(out_label_list, preds_list),
            "f1": f1_score(out_label_list, preds_list),
        }

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
# talvez comentar isto se der erro
        compute_metrics=compute_metrics,
    )

    # Training
    if training_args.do_train:
        # print(model_args)
        # print(training_args)
        # trainer.resize_token_embeddings(len(tokenizer))

        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()

        # json.dump(model_config,open(os.path.join("out_base","model_config.json"),"w"))
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        # if trainer.is_world_master():
        tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        result = trainer.evaluate()

        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        # if trainer.is_world_master():
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key, value in result.items():
                logger.info("  %s = %s", key, value)
                writer.write("%s = %s\n" % (key, value))

        results.update(result)


    # Predict
    if training_args.do_predict:
        test_dataset = NerDataset(
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            labels=labels,
            model_type=config.model_type,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.test,
        )

        predictions, label_ids, metrics = trainer.predict(test_dataset)
        # print('predictions ' + str(predictions))
        preds_soft = np.argmax(predictions, axis=2)
        # print('preds_soft ', preds_soft)
        preds_list, _ = align_predictions(predictions, label_ids)
        # print('preds_listttttttttttttttttttttttttttttttttt ', preds_list)
        # Save predictions
        # output_test_results_file = os.path.join(training_args.output_dir, "test_results.txt")
        # # if trainer.is_world_master():
        # with open(output_test_results_file, "w") as writer:
        #     logger.info("***** Test results *****")
        #     for key, value in metrics.items():
        #         logger.info("  %s = %s", key, value)
        #         writer.write("%s = %s\n" % (key, value))


        output_test_predictions_file = os.path.join(training_args.output_dir, "test_predictions.txt")
        # if trainer.is_world_master():
        with open(output_test_predictions_file, "w") as writer:
            with open(os.path.join(data_args.data_dir, "test.txt"), "r") as f:
                example_id = 0

                for line in f:
                    # print('line ', line)
                    if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                        writer.write(line)
                        if not preds_list[example_id]:
                            example_id += 1
                    elif preds_list[example_id]:
                        # print('preds_list ', preds_list)

                        entity_label = preds_list[example_id].pop(0)
                        # print('entity_label ',entity_label)
                        output_line = line.split()[0] + " " + entity_label + "\n"
                        # if entity_label == 'O':
                        #     output_line = line.split()[0] + " " + entity_label + "\n"
                        # else:
                        #     output_line = line.split()[0] + " " + entity_label[0] + "\n"
                        # output_line = line.split()[0] + " " + preds_list[example_id].pop(0) + "\n"
                        # print(output_line)
                        writer.write(output_line)
                    else:
                        logger.warning(
                            "Maximum sequence length exceeded: No prediction for '%s'.", line.split()[0]
                        )


    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
