import torch
import torch.nn as nn 
import torch.optim as optim

from utils import load_config, set_seed, save_outputs
from data_loader import TwitterDataset  
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader

import ffnn, cnn, lstm

if __name__ == "__main__": 
    cfg = load_config()
    set_seed(cfg.general.seed)        

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.llm.model)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    if cfg.neural_baselines.model == "ffnn": 
        num_epochs = cfg.ffnn.epochs
        batch_size = cfg.ffnn.batch_size
        resume_from_checkpoint = cfg.ffnn.resume_from_checkpoint
        model = ffnn.FFNN(cfg, len(tokenizer), device)
        criterion = nn.BCEWithLogitsLoss()
    elif cfg.neural_baselines.model == "cnn": 
        num_epochs = cfg.cnn.epochs
        batch_size = cfg.cnn.batch_size
        resume_from_checkpoint = cfg.cnn.resume_from_checkpoint
        model = cnn.TextCNN(cfg, len(tokenizer), device)
        criterion = nn.BCEWithLogitsLoss()
    elif cfg.neural_baselines.model == "lstm":
        num_epochs = cfg.lstm.epochs
        batch_size = cfg.lstm.batch_size
        resume_from_checkpoint = cfg.lstm.resume_from_checkpoint
        model = lstm.LSTM(cfg, len(tokenizer), device)
        criterion = nn.BCELoss()
    else: 
        raise NotImplementedError

    print(f"{cfg.neural_baselines.model} parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    model.to(device)
    optimizer = optim.Adam(model.parameters())

    twitter = TwitterDataset(cfg)
    tokenized_dataset = twitter.tokenize_to_hf(tokenizer)

    train_dl = DataLoader(tokenized_dataset['train'], shuffle=True, batch_size=batch_size, collate_fn=data_collator)
    eval_dl = DataLoader(tokenized_dataset['eval'], shuffle=True, batch_size=batch_size, collate_fn=data_collator)
    test_dl = DataLoader(tokenized_dataset['test'], shuffle=False, batch_size=batch_size, collate_fn=data_collator)

    epoch_init = -1
    if resume_from_checkpoint:
        checkpoint = torch.load(resume_from_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_init = checkpoint['epoch']

    max_acc = 0
    for epoch in range(epoch_init+1, num_epochs):
        # Train the model on the train set
        model.run_train(train_dl, optimizer, criterion, epoch, num_epochs)

        # Evaluate the model on the val set
        acc = model.run_eval(eval_dl, epoch, num_epochs)['accuracy']

        if acc >= max_acc:
            max_acc = acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f"{cfg.data.checkpoint_path}/best.ckpt")
            
            outputs = model.run_test(test_dl)
            save_outputs(outputs, cfg.general.run_id)
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, f"{cfg.data.checkpoint_path}/final.ckpt")

 