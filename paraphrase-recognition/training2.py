import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, models, InputExample, LoggingHandler, losses
import csv, logging, math
# from sentence_transformers import , SentenceTransformer, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
print(str(n_gpu)+ ' '+ str(device) + ' Device available: '+ torch.cuda.get_device_name(0))
torch.cuda.empty_cache()

max_seq_length = 128
evaluation_steps = 512
num_epochs = 20
train_batch_size = 2

# MODEL_PATH = 'bert-base-cased'
# MODEL_PATH = 'bert-base-uncased'
MODEL_PATH = 'dmis-lab/biobert-base-cased-v1.1'

# MODEL_PATH = 'fagner/envoy'

OUTPUT_MODEL = 'output/sts/' + MODEL_PATH + '/' + num_epochs + 'epochs'

# DATASET = 'data-eval/stsbenchmark.tsv'
DATASET = '../datasets/STS/stsbenchmark.tsv'



word_embedding_model = models.Transformer(MODEL_PATH, max_seq_length=max_seq_length)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
model.save(OUTPUT_MODEL)

# Convert the dataset to a DataLoader ready for training
logging.info("Read STS benchmark train dataset")

train_samples = []
dev_samples = []
test_samples = []

with open(DATASET, "r") as f:
    reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        if row['split'] == 'train':
            score = float(row['score']) / 5.0 #Normalize score to range 0 ... 1
            train_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))

        if row['split'] == 'dev':
            score = float(row['score']) / 5.0 #Normalize score to range 0 ... 1
            dev_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))

dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name='sts-dev')

train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
train_loss = losses.MultipleNegativesRankingLoss(model)

warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))

model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=dev_evaluator,
          epochs=num_epochs,
          evaluation_steps=evaluation_steps,
          warmup_steps=warmup_steps,
          output_path=OUTPUT_MODEL,
          # use_amp=use_amp,
          checkpoint_path=OUTPUT_MODEL,
          checkpoint_save_steps=1025,
          # checkpoint_save_total_limit=3
          )
