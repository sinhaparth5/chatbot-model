import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from data_module import QADataModule
from transformer_model import ChatTransformer
from transformers import AutoTokenizer

with open('config/params.yaml') as f:
    config = yaml.safe_load(f)

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
dm = QADataModule(
    data_path='data/raw/train_data_chatbot_processed.json',
    tokenizer_name='bert-base-uncased',
    batch_size=config['training']['batch_size'],
    max_length=config['model']['max_seq_length']
)

model = ChatTransformer(config, tokenizer)

checkpoint_callback = ModelCheckpoint(
    dirpath='models/',
    filename='chatbot-{epoch:02d}-{val_loss:.2f}',
    save_top_k=3,
    monitor='val_loss'
)

trainer = pl.Trainer(
    max_epochs=config['training']['epochs'],
    accumulate_grad_batches=config['training']['accumulation_steps'],
    callbacks=[checkpoint_callback],
    accelerator='auto'
)

trainer.fit(model, dm)