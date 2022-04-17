import csv, logging
import torch
import json
import math
from torch.utils.data import DataLoader
from sentence_transformers import InputExample, SentenceTransformer, models, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
print(str(n_gpu)+ ' '+ str(device) + ' Device available: '+ torch.cuda.get_device_name(0))

torch.cuda.empty_cache()


DATASET = 'mli/'
# DATASET = 'snli_1.0/'

DATASET_PATH = '../datasets/NLI/' + DATASET


# MODEL_PATH = 'fagner/envoy'
# MODEL_PATH = 'bert-base-cased'
MODEL_PATH = 'dmis-lab/biobert-base-cased-v1.1'
# DATASET = '../datasets/NLI/snli_1.0/'

OUTPUT_MODEL = 'output/' + MODEL_PATH + '/' + DATASET

word_embedding_model = models.Transformer(MODEL_PATH, max_seq_length=128)


pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
word_embedding_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
word_embedding_model.save(OUTPUT_MODEL)

num_epochs = 30
train_batch_size = 2

# Convert the dataset to a DataLoader ready for training
logging.info("Read MLI train dataset")

# sts_dataset_path = '../datasets/NLI/snli_1.0/'

train_samples = []
dev_samples = []
test_samples = []

with open(DATASET_PATH + 'mli_train_v1.jsonl', "r") as f:
    json_list = list(f)
    for j in json_list:
        row = json.loads(j)
        gold_label = row['gold_label']  # Normalize score to range 0 ... 1
        float_gold_label = float(0)

        if gold_label == 'contradiction':
            float_gold_label = 0 / 3.0
        if gold_label == 'entailment':
            float_gold_label = 1 / 3.0
        if gold_label == 'neutral':
            float_gold_label = 2 / 3.0

        inp_example = InputExample(texts=[row['sentence1'], row['sentence2']], label=float_gold_label)
        train_samples.append(inp_example)

logging.info("Read MLI benchmark dev dataset")
with open(DATASET_PATH + 'mli_dev_v1.jsonl', "r") as f:
# with open(DATASET_PATH + 'snli_1.0_test.txt', "r") as f:

    json_list = list(f)
    for j in json_list:
        row = json.loads(j)

        gold_label = row['gold_label']  # Normalize score to range 0 ... 1
        float_gold_label = float(0)

        if gold_label == 'contradiction':
            float_gold_label = 0 / 3.0
        if gold_label == 'entailment':
            float_gold_label = 1 / 3.0
        if gold_label == 'neutral':
            float_gold_label = 2 / 3.0

        inp_example = InputExample(texts=[row['sentence1'], row['sentence2']], label=float_gold_label)

        dev_samples.append(inp_example)



train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
# # train_dataloader = DataLoader(train_samples[:80000], shuffle=True, batch_size=train_batch_size)
train_loss = losses.CosineSimilarityLoss(model=word_embedding_model)

# Development set: Measure correlation between cosine score and gold labels
evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name='mli-dev')

# Configure the training. We skip evaluation in this example
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))
logging.info("checkpoint_save_steps: {}".format(10*len(train_dataloader)))


# Train the model
word_embedding_model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=10000,
          warmup_steps=warmup_steps,
          output_path=OUTPUT_MODEL,
          checkpoint_path=OUTPUT_MODEL,
          checkpoint_save_steps=10*len(train_dataloader),
          save_best_model = True,
          checkpoint_save_total_limit=3)


##############################################################################
#
# Load the stored model and evaluate its performance on MLI test dataset
#
##############################################################################

# with open(DATASET_PATH + 'snli_1.0_test.jsonl', "r") as f:
with open(DATASET_PATH + 'mli_test_v1.jsonl', "r") as f:
    json_list = list(f)
    for j in json_list:
        row = json.loads(j)

        gold_label = row['gold_label']  # Normalize score to range 0 ... 1
        float_gold_label = float(0)

        if gold_label == 'contradiction':
            float_gold_label = 0 / 3.0
        if gold_label == 'entailment':
            float_gold_label = 1 / 3.0
        if gold_label == 'neutral':
            float_gold_label = 2 / 3.0

        inp_example = InputExample(texts=[row['sentence1'], row['sentence2']], label=float_gold_label)

        test_samples.append(inp_example)

model = SentenceTransformer(OUTPUT_MODEL)
test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, batch_size=train_batch_size, name='test')
test_evaluator(model, output_path=OUTPUT_MODEL)
