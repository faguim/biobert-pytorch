import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, models, InputExample, LoggingHandler, losses
import csv, logging, math
# from sentence_transformers import , SentenceTransformer, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

logger = logging.getLogger(__name__)

learning_rate = 5e-05
# training_args.gradient_accumulation_steps = 16
# training_args.eval_steps = 32
# training_args.save_steps = 32

# Setup logging
# logging.basicConfig(
#     format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
#     datefmt="%m/%d/%Y %H:%M:%S",
#     level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
# )
# logger.warning(
#     "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
#     training_args.local_rank,
#     training_args.device,
#     training_args.n_gpu,
#     bool(training_args.local_rank != -1),
#     training_args.fp16,
#     # training_args.per_device_eval_batch_size
# )
# logger.info("Training/evaluation parameters %s", training_args)

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
num_epochs = 1
train_batch_size = 2


DATASET = '../datasets/STS/stsbenchmark.tsv'

# MODEL_PATH = 'bert-base-cased'
MODEL_PATH = 'dmis-lab/biobert-base-cased-v1.1'
# MODEL_PATH = 'fagner/envoy'
OUTPUT_MODEL = 'output/' + MODEL_PATH + '/' + str(num_epochs) + 'epochs'


word_embedding_model = models.Transformer(MODEL_PATH, max_seq_length=max_seq_length)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
model.save(OUTPUT_MODEL)


logging.info("Reading STS benchmark dataset")

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

evaluation_steps = len(train_samples)
logging.info("Train samples: {}".format(len(train_samples)))
logging.info("Evaluation samples: {}".format(len(dev_samples)))

dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name='sts-dev')

train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
train_loss = losses.MultipleNegativesRankingLoss(model)

warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.05) #10% of train data for warm-up
logging.info("Warmup steps: {}".format(warmup_steps))

print(10*len(train_dataloader))
# steps = len(train_dataloader)
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=dev_evaluator,
          epochs=num_epochs,
          # steps_per_epoch = None,
          evaluation_steps=evaluation_steps,
          warmup_steps=warmup_steps,
          output_path=OUTPUT_MODEL,
          save_best_model = True,
          # use_amp=use_amp,
          checkpoint_path=OUTPUT_MODEL,
          checkpoint_save_steps=10*len(train_dataloader),
          checkpoint_save_total_limit=3
          )
