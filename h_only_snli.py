from transformers import AlbertTokenizer, AlbertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
# from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup

from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.dummy import DummyClassifier

import torch
import gc
import pandas as pd
import numpy as np

import time
import datetime
import random

device = torch.device("cuda:2")

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

def load_data(file_path, oversample=False):
    df = pd.read_csv(file_path)
    
    sentences = df.sentences.values
    labels = df.labels.values

    tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
    idxs = []
    for i in range(len(sentences)):
        if pd.isnull(sentences[i]):
            idxs.append(i)
    sentences = np.delete(sentences, idxs)
    labels = np.delete(labels, idxs)

    encoding = tokenizer(list(sentences), return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)

    input_ids = encoding['input_ids']
    attention_masks = encoding['attention_mask']
    labels = torch.tensor(labels).unsqueeze(1).to(device)

    return input_ids, attention_masks, labels



def run(seed=42):
    ## META VARIABLES
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    epochs = 10
    batch_size = 64
    
    # X_train, mask_train, y_train = load_data('data/snli_h_train.csv')
    # torch.save(X_train, 'snli_train_x.pt')
    # torch.save(mask_train, 'snli_train_mask.pt')
    # torch.save(y_train, 'snli_train_y.pt')
    X_train = torch.load('snli_train_x.pt')
    mask_train = torch.load('snli_train_mask.pt')
    y_train = torch.load('snli_train_y.pt')
    print(type(y_train) != type(np.array([])))
    print(type(y_train.detach().cpu().numpy()) != type(np.array([])))

    X_val, mask_val, y_val = load_data('data/snli_h_dev.csv')
    X_test, mask_test, y_test = load_data('data/snli_h_test.csv')
    print(type(y_train), type(y_val), type(y_test))


    train_dataset = TensorDataset(X_train, mask_train, y_train)
    val_dataset = TensorDataset(X_val, mask_val, y_val)
    test_dataset = TensorDataset(X_test, mask_test, y_test)
    
    # Dataloading
    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)
    validation_dataloader = DataLoader(val_dataset, sampler=RandomSampler(val_dataset), batch_size=batch_size)
    prediction_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=prediction_sampler, batch_size=batch_size)

    model = AlbertForSequenceClassification.from_pretrained('albert-base-v2', num_labels = 3).to(device)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader) * epochs)

    total_t0 = time.time()
    best_state_dict = None
    best_val = 0
    for epoch_i in range(epochs):
    
        # ========================================
        #               Training
        # ========================================
        
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        # Measure how long the training epoch takes.
        t0 = time.time()
        total_train_loss = 0

        model.train()

        predictions_train = np.array([])
        for step, batch in enumerate(train_dataloader):

            # Progress update every 50 batches.
            if step % 50 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            model.zero_grad()        
            loss, logits = model(b_input_ids, 
                                token_type_ids=None, 
                                attention_mask=b_input_mask, 
                                labels=b_labels)

            total_train_loss += loss.item() * b_labels.shape[0]
            loss.backward()
            optimizer.step()
            scheduler.step()

            gc.collect()

            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            predictions_train = np.append(predictions_train, np.argmax(logits, axis=1).flatten())
       
        # Calculate the average loss over all of the batches.
        if type(y_train) != type(np.array([])):
            y_train = y_train.detach().cpu().numpy()
        accuracy_train = np.sum(predictions_train == y_train) / y_train.shape[0]
        f1_macro_train = f1_score(y_train, predictions_train, average='macro')
        f1_micro_train = f1_score(y_train, predictions_train, average='micro')
        print("\n  Training Accuracy: {0:.2f}".format(accuracy_train))
        print("  Training F1-MACRO: {0:.2f}".format(f1_macro_train))
        print("  Training F1-MICRO: {0:.2f}".format(f1_micro_train))

        avg_train_loss = total_train_loss / y_train.shape[0]        
        training_time = format_time(time.time() - t0)
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(training_time))
            
        # ========================================
        #               Validation
        # ========================================

        print("\nRunning Validation...")

        t0 = time.time()
        model.eval()

        total_val_loss = 0
        predictions_val = np.array([])
        for batch in validation_dataloader:            
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            
            with torch.no_grad():        
                (loss, logits) = model(b_input_ids, 
                                    token_type_ids=None, 
                                    attention_mask=b_input_mask,
                                    labels=b_labels)
                
            total_val_loss += loss.item() * b_labels.shape[0]

            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            predictions_val = np.append(predictions_val, np.argmax(logits, axis=1).flatten())
        if type(y_val) != type(np.array([])):
            y_val = y_val.detach().cpu().numpy()
        accuracy_val = np.sum(predictions_val == y_val) / y_val.shape[0]
        f1_macro_val = f1_score(y_val, predictions_val, average='macro')
        f1_micro_val = f1_score(y_val, predictions_val, average='micro')
        print("  Accuracy: {0:.2f}".format(accuracy_val))
        print("  F1-MACRO: {0:.2f}".format(f1_macro_val))
        print("  F1-MICRO: {0:.2f}".format(f1_micro_val))

        performance_metric = f1_macro_val

        if performance_metric > best_val:
            print("Best Model Updated.")
            best_val = performance_metric
            best_state_dict = model.state_dict()

        avg_val_loss = total_val_loss / y_val.shape[0]
        validation_time = format_time(time.time() - t0)
        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))

    print("\nTraining complete!")
    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

    # ========================================
    #               Test
    # ========================================

    model.load_state_dict(best_state_dict)
    model.eval()

    predictions_test = np.array([])
    for batch in test_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        
        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None, 
                            attention_mask=b_input_mask)

        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        predictions_test = np.append(predictions_test, np.argmax(logits, axis=1).flatten())

    y_test = y_test.detach().cpu().numpy()
    best_accr = np.sum(predictions_test == y_test) / y_test.shape[0]
    best_macro_f1 = f1_score(y_test, predictions_test, average='macro')
    best_micro_f1 = f1_score(y_test, predictions_test, average='micro')
    best_confusion_matrix = confusion_matrix(y_test, predictions_test)
    print("  Test Accuracy: {0:.2f}".format(best_accr))
    print("  Test F1-MACRO: {0:.2f}".format(best_macro_f1))
    print("  Test F1-MICRO: {0:.2f}".format(best_micro_f1))



    # ========================================
    #              Dummy Test
    # ========================================

    X_train = X_train.detach().cpu().numpy()
    X_test = X_test.detach().cpu().numpy()
    dummy_clf = DummyClassifier(strategy="uniform")
    dummy_clf.fit(X_train, y_train)
    predictions_dummy = dummy_clf.predict(X_test)

    dummy_accr = np.sum(predictions_dummy == y_test) / y_test.shape[0]
    dummy_macro_f1 = f1_score(y_test, predictions_dummy, average='macro')
    dummy_micro_f1 = f1_score(y_test, predictions_dummy, average='micro')
    dummy_confusion_matrix = confusion_matrix(y_test, predictions_dummy)
    print("  Dummy Accuracy: {0:.2f}".format(dummy_accr))
    print("  Dummy F1-MACRO: {0:.2f}".format(dummy_macro_f1))
    print("  Dummy F1-MICRO: {0:.2f}".format(dummy_micro_f1))
    
    torch.save(best_state_dict, './output/best_model_snli.pt')

    return {
        'seed': seed,
        'best_accr': best_accr,
        'best_macro_f1': best_macro_f1,
        'best_micro_f1': best_micro_f1,
        'dummy_accr': dummy_accr,
        'dummy_macro_f1': dummy_macro_f1,
        'dummy_micro_f1': dummy_micro_f1
    }


def run_many_times():
    run_time = 1
    result_record = pd.DataFrame([], columns=['seed', 'best_accr', 'best_macro_f1', 'best_micro_f1', 'dummy_accr', 'dummy_macro_f1', 'dummy_micro_f1'])
    for i in range(run_time):
        result_df = run(seed=42+i)
        result_record = result_record.append(result_df, ignore_index=True)
        
        result_record.to_csv('./output/result_snli.csv')
        result_record.to_pickle('./output/result_snli.pkl')

def main():
    run_many_times()

if __name__ == "__main__":
    main()