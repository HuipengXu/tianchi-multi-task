import os
import json
import logging
from functools import partial

from mtl.model import BertMultiTaskModel
from mtl.args import ModelArguments, DataTrainingArguments
from mtl.utils import (
    task_data,
    task_num_classes,
    task_id_to_name,
    preprocess,
    SingleTaskDataset,
    MultiTaskDataset,
    MultiTaskBatchSampler,
    collate_fn,
    compute_metrics,
    load_json
)
from mtl.trainer_utils import PREFIX_CHECKPOINT_DIR
from mtl.transformer_trainer import Trainer

import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from transformers import (
    TrainingArguments,
    BertConfig,
    HfArgumentParser,
    PreTrainedModel
)

from typing import Any, Optional, Callable, Dict

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class MultiTaskTrainer(Trainer):

    def __init__(self, model: PreTrainedModel, args: Any, model_args: Any, data_args: Any,
                 train_dataset: MultiTaskDataset, eval_dataset: MultiTaskDataset, compute_metrics: Callable,
                 max_seq_len: int):
        super(MultiTaskTrainer, self).__init__(model=model, args=args, model_args=model_args, data_args=data_args,
                                               train_dataset=train_dataset, eval_dataset=eval_dataset,
                                               compute_metrics=compute_metrics)
        self.collate = partial(collate_fn, max_seq_len=max_seq_len)

    def get_train_dataloader(self) -> DataLoader:
        batch_sampler = MultiTaskBatchSampler(datasets=self.train_dataset,
                                              batch_size=self.args.train_batch_size)
        dataloader = DataLoader(dataset=self.train_dataset,
                                batch_sampler=batch_sampler,
                                collate_fn=self.collate)
        return dataloader

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        batch_sampler = MultiTaskBatchSampler(datasets=eval_dataset,
                                              batch_size=self.args.eval_batch_size,
                                              shuffle=False)
        dataloader = DataLoader(dataset=eval_dataset,
                                batch_sampler=batch_sampler,
                                collate_fn=self.collate)
        return dataloader

    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        batch_sampler = MultiTaskBatchSampler(datasets=test_dataset,
                                              batch_size=self.args.eval_batch_size,
                                              shuffle=False)
        dataloader = DataLoader(dataset=test_dataset,
                                batch_sampler=batch_sampler,
                                collate_fn=self.collate)
        return dataloader

    def _training_step(
            self, model: nn.Module, inputs: Dict[str, torch.Tensor], optimizer: torch.optim.Optimizer
    ) -> float:
        model.train()
        task_id = inputs.pop('task_id')
        for k, v in inputs.items():
            inputs[k] = v.to(self.args.device)

        inputs['task_id'] = task_id
        outputs = model(**inputs)
        loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        loss.backward()

        return loss.item()


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    cache_data_path = {}
    if not os.path.exists(data_args.data_save_dir) or data_args.overwrite_cache:
        for id_, task in task_id_to_name.items():
            task_cache_data_path = preprocess(data_args, task, data_args.test_b)
            cache_data_path[id_] = task_cache_data_path
    else:
        for id_, task_name in task_id_to_name.items():
            cache_data_path[id_] = {
                'train': os.path.join(data_args.data_save_dir, task_name, 'train.pt'),
                'val': os.path.join(data_args.data_save_dir, task_name, 'val.pt'),
                'predict': os.path.join(data_args.data_save_dir, task_name, 'predict.pt')
            }

    multi_task_train_dataset = {}
    multi_task_val_dataset = {}
    multi_task_predict_dataset = {}

    for id_, cache_path in cache_data_path.items():
        multi_task_train_dataset[id_] = SingleTaskDataset(cache_path['train'])
        multi_task_val_dataset[id_] = SingleTaskDataset(cache_path['val'])
        multi_task_predict_dataset[id_] = SingleTaskDataset(cache_path['predict'])

    train_dataset = MultiTaskDataset(multi_task_train_dataset)
    val_dataset = MultiTaskDataset(multi_task_val_dataset)
    predict_dataset = MultiTaskDataset(multi_task_predict_dataset)

    if not model_args.freeze:
        bert_config = BertConfig.from_pretrained(model_args.model_name_or_path,
                                                 output_hidden_states=True)
        model = BertMultiTaskModel(config=bert_config, task_num_classes=task_num_classes,
                                   model_path=model_args.model_name_or_path)
    else:
        model = BertMultiTaskModel.from_pretrained(model_args.model_name_or_path, task_num_classes=task_num_classes,
                                                   model_path=model_args.model_name_or_path)

    if model_args.freeze:
        for p in model.bert.parameters():
            p.requires_grad = False

    trainer = MultiTaskTrainer(model=model, args=training_args, model_args=model_args, data_args=data_args,
                               train_dataset=train_dataset, eval_dataset=val_dataset, compute_metrics=compute_metrics,
                               max_seq_len=data_args.max_seq_length)

    # Training
    if training_args.do_train:
        _, model, best_score = trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
    else:
        best_score = 1.0

        # Evaluation
    eval_results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        eval_result = trainer.evaluate()
        avg_score = sum(eval_result[metric] for metric in
                        ['eval_ocnli', 'eval_ocemotion', 'eval_tnews']) / 3
        if avg_score > best_score:
            # Save model checkpoint
            output_dir = os.path.join(training_args.output_dir,
                                      f"{PREFIX_CHECKPOINT_DIR}-{trainer.global_step}"
                                      f"-{avg_score:.4f}")
            trainer.best_model_path = output_dir

            trainer.save_model(output_dir)

            if trainer.is_world_master():
                trainer._rotate_checkpoints()

        logger.info(f"***** Eval results *****")
        for key, value in eval_result.items():
            logger.info(f"{key} = {value}")

        eval_results.update(eval_result)

    if training_args.do_predict:
        logger.info("*** Test ***")

        predictions_output = trainer.predict(test_dataset=predict_dataset)
        predictions = predictions_output.predictions
        for task_id, preds in predictions.items():
            preds = preds.cpu().tolist()
            task_name = task_id_to_name[task_id]
            submit_file_dir = os.path.join(os.path.dirname(os.path.dirname(training_args.output_dir)),
                                           'prediction_result')
            if not os.path.exists(submit_file_dir):
                os.makedirs(submit_file_dir)
            submit_file = os.path.join(submit_file_dir, f'{task_name}_predict.json')
            label2id = load_json(os.path.join(data_args.data_save_dir, task_name, 'label2id.json'))
            id2label = {id_: label for label, id_ in label2id.items()}
            with open(submit_file, 'w', encoding='utf8') as f:
                for id_, pred in enumerate(preds):
                    label = id2label[pred]
                    line = {'id': str(id_), 'label': label}
                    f.write(json.dumps(line))
                    if id_ < len(preds) - 1:
                        f.write('\n')

        trainer.evaluate()

        if data_args.pseudo:
            task_probs = predictions_output.task_probs
            predictions = predictions_output.predictions
            for task_id, probs in task_probs.items():
                task_name = task_id_to_name[task_id]
                label2id = load_json(os.path.join(data_args.data_save_dir, task_name, 'label2id.json'))
                id2label = {id_: label for label, id_ in label2id.items()}
                pseudo_dir = os.path.join(training_args.output_dir, 'pseudo')
                if not os.path.exists(pseudo_dir):
                    os.makedirs(pseudo_dir)
                cur_pseudo_train_file = os.path.join(pseudo_dir,
                                                     task_data[
                                                         task_id_to_name[task_id]
                                                     ]['train'].split('_')[0] + '_pseudo.csv')
                cur_task_test_file = os.path.join(data_args.data_dir,
                                                  task_data[task_id_to_name[task_id]]['predict'])
                count = 0
                test_df = pd.read_csv(cur_task_test_file, sep='\t', header=None)
                with open(cur_pseudo_train_file, 'w', encoding='utf8') as f:
                    for id_, prob in enumerate(probs):
                        if prob.max() >= 0.99:
                            count += 1
                            label = id2label[predictions[task_id].cpu().tolist()[id_]]
                            if task_id == '0':
                                text = '\t'.join(test_df.iloc[id_, 1:3].tolist())
                            else:
                                text = test_df.iloc[id_, 1]
                            row = '\t'.join(['test', text, label])
                            f.write(row)
                            f.write('\n')


if __name__ == "__main__":
    main()
