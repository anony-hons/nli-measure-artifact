from transformers import AlbertTokenizer, AlbertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
# from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup

from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.dummy import DummyClassifier

from collections import defaultdict

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

def load_data(dataset, num_sample):
    sampled_sentences = []
    sampled_labels = []
    sample_per_class = int(num_sample/12)
    for label, sentences in dataset.items():
        s_pairs = random.choices(sentences, k=sample_per_class)
        for pair in s_pairs:
            sampled_sentences += [pair[0], pair[1]]
        sampled_labels += list(label) * sample_per_class
    
    assert(len(sampled_sentences) == sample_per_class * 12)
    assert(len(sampled_labels) == sample_per_class * 12)

    tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
    encoding = tokenizer(sampled_sentences, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)

    input_ids = encoding['input_ids']
    attention_masks = encoding['attention_mask']
    labels = torch.tensor(sampled_labels).unsqueeze(1).to(device)

    return input_ids, attention_masks, labels


def load_sentences(file_path):
    label_map ={
        'entailment': 0,
        'neutral': 1,
        'contradiction': 2
    }
    df = pd.read_csv(file_path, sep='\t')
    
    sentences = df.sentences.values
    labels = df.labels.values
    l = list(range(len(sentences)))

    sentence_per_class = defaultdict(list)
    for i in range(0, len(sentences), 2):
        i1 = i
        i2 = i + 1
        sentence_per_class[(label_map[labels[i1]], label_map[labels[i2]])]\
            .append((sentences[i1], sentences[i2]))
        
    return sentence_per_class



def run(datasets, seed=42):
    ## META VARIABLES
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    epochs = 20
    batch_size = 32

    X_train, mask_train, y_train = load_data(datasets['train'], num_sample=int(625*0.7))
    X_val, mask_val, y_val = load_data(datasets['val'], num_sample=int(625*0.1))
    X_test, mask_test, y_test = load_data(datasets['test'], num_sample=int(625*0.2))


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
        true_label_train = np.array([])
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
            true_label_train = np.append(true_label_train, label_ids)
            
       
        # Calculate the average loss over all of the batches.
        accuracy_train = np.sum(predictions_train == true_label_train) / true_label_train.shape[0]
        f1_macro_train = f1_score(true_label_train, predictions_train, average='macro')
        f1_micro_train = f1_score(true_label_train, predictions_train, average='micro')
        print("\n  Training Accuracy: {0:.2f}".format(accuracy_train))
        print("  Training F1-MACRO: {0:.2f}".format(f1_macro_train))
        print("  Training F1-MICRO: {0:.2f}".format(f1_micro_train))

        avg_train_loss = total_train_loss / true_label_train.shape[0]        
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
        true_label_val = np.array([])
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
            true_label_val = np.append(true_label_val, label_ids)

        accuracy_val = np.sum(predictions_val == true_label_val) / true_label_val.shape[0]
        f1_macro_val = f1_score(true_label_val, predictions_val, average='macro')
        f1_micro_val = f1_score(true_label_val, predictions_val, average='micro')
        print("  Accuracy: {0:.2f}".format(accuracy_val))
        print("  F1-MACRO: {0:.2f}".format(f1_macro_val))
        print("  F1-MICRO: {0:.2f}".format(f1_micro_val))

        performance_metric = f1_macro_val

        if performance_metric > best_val:
            print("Best Model Updated.")
            best_val = performance_metric
            best_state_dict = model.state_dict()

        avg_val_loss = total_val_loss / true_label_val.shape[0]
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
    true_label_test = np.array([])
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
        true_label_test = np.append(true_label_test, label_ids)
    
    best_accr = np.sum(predictions_test == true_label_test) / true_label_test.shape[0]
    best_macro_f1 = f1_score(true_label_test, predictions_test, average='macro')
    best_micro_f1 = f1_score(true_label_test, predictions_test, average='micro')
    best_confusion_matrix = confusion_matrix(true_label_test, predictions_test)
    print("  Test Accuracy: {0:.2f}".format(best_accr))
    print("  Test F1-MACRO: {0:.2f}".format(best_macro_f1))
    print("  Test F1-MICRO: {0:.2f}".format(best_micro_f1))



    # ========================================
    #              Dummy Test
    # ========================================

    X_train = X_train.detach().cpu().numpy()
    X_test = X_test.detach().cpu().numpy()
    y_train = y_train.detach().cpu().numpy().squeeze(1)
    y_test = y_test.detach().cpu().numpy().squeeze(1)
    dummy_clf = DummyClassifier(strategy="uniform")
    dummy_clf.fit(X_train, y_train)
    predictions_dummy = dummy_clf.predict(X_test)

    dummy_accr = np.sum(predictions_dummy == y_test) / y_test.shape[0]
    dummy_macro_f1 = f1_score(y_test, predictions_dummy, average='macro')
    dummy_micro_f1 = f1_score(y_test, predictions_dummy, average='micro')
    print("  Dummy Accuracy: {0:.2f}".format(dummy_accr))
    print("  Dummy F1-MACRO: {0:.2f}".format(dummy_macro_f1))
    print("  Dummy F1-MICRO: {0:.2f}".format(dummy_micro_f1))
    
    # torch.save(best_state_dict, './output/best_model_kaushik_sample.pt')

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
    run_time = 34
    seed = 108
    result_record = pd.DataFrame([], columns=['seed', 'best_accr', 'best_macro_f1', 'best_micro_f1', 'dummy_accr', 'dummy_macro_f1', 'dummy_micro_f1'])
    
    # Data PREP
    train_dict = load_sentences('data/kaushik_train.tsv')
    val_dict = load_sentences('data/kaushik_dev.tsv')
    test_dict = load_sentences('data/kaushik_test.tsv')
    datasets = {
        'train': train_dict,
        'val': val_dict,
        'test': test_dict
    }
    
    for i in range(run_time):
        result_df = run(datasets, seed=seed+i)
        result_record = result_record.append(result_df, ignore_index=True)
        
        result_record.to_csv('./output/result_kaushik_sample_{0}.csv'.format(seed))
        # result_record.to_pickle('./output/result_kaushik_sample.pkl')
        print("RUNTIME is", i)

def main():
    run_many_times()

if __name__ == "__main__":
    main()