from transformers import AlbertTokenizer, AlbertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
# from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup

from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.dummy import DummyClassifier

from collections import defaultdict, Counter

import torch
import gc
import pandas as pd
import numpy as np

import time
import datetime
import random

device = torch.device("cuda:6")

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def run(input_sen, output, testloaders, logit_df, seed=42):
    ## META VARIABLES
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    epochs = 20
    batch_size = 32
    
    # Train-Val-Test Split
    X_train, X_test, y_train, y_test, _, test_idx = train_test_split(input_sen, output, np.arange(len(input_sen)), test_size=0.2)
    y_majority = max(Counter(y_test).values()) / len(y_test)
    assert((input_sen[test_idx] == X_test).all())
    assert((output[test_idx] == y_test).all())
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.125)

    ros = RandomOverSampler()
    X_train, y_train = ros.fit_resample(X_train.reshape(-1, 1), y_train)

    # Tokenization & Dataloading
    tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')

    #Train
    encoding = tokenizer(list(map(lambda x: x[0], X_train)), return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
    input_ids = encoding['input_ids']
    attention_masks = encoding['attention_mask']
    labels = torch.tensor(y_train).unsqueeze(1).to(device)
    train_dataset = TensorDataset(input_ids, attention_masks, labels)
    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)


    #Val
    encoding = tokenizer(list(X_val), return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
    input_ids = encoding['input_ids']
    attention_masks = encoding['attention_mask']
    labels = torch.tensor(y_val).unsqueeze(1).to(device)
    val_dataset = TensorDataset(input_ids, attention_masks, labels)
    validation_dataloader = DataLoader(val_dataset, sampler=RandomSampler(val_dataset), batch_size=batch_size)


    #Test
    encoding = tokenizer(list(X_test), return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
    input_ids = encoding['input_ids']
    attention_masks = encoding['attention_mask']
    labels = torch.tensor(y_test).unsqueeze(1).to(device)
    test_dataset = TensorDataset(input_ids, attention_masks, labels)
    prediction_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=prediction_sampler, batch_size=batch_size)


    model = AlbertForSequenceClassification.from_pretrained('albert-base-v2', num_labels = 3).to(device)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5)
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

    # testloaders['indomain'] = (test_dataloader, None, None)
    test_measures = defaultdict(dict)

    for name, content in testloaders.items():
        testloader, sentences, labels = content
        predictions_test = np.array([])
        true_label_test = np.array([])
        logit_stack = np.empty((0,3))

        for batch in testloader:
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

            if name == 'indomain':
                logit_stack = np.vstack((logit_stack, logits))
        

        if name == 'indomain':
            for cnt, idx in enumerate(test_idx):
                logit_df.iloc[idx,:] += logit_stack[cnt,:]
            easy_examples = test_idx[predictions_test == true_label_test]
            hard_examples = test_idx[predictions_test != true_label_test]


        best_accr = np.sum(predictions_test == true_label_test) / true_label_test.shape[0]
        best_macro_f1 = f1_score(true_label_test, predictions_test, average='macro')
        print(name + ":  Test Accuracy: {0:.2f}".format(best_accr))
        print(name + ":  Test F1-MACRO: {0:.2f}".format(best_macro_f1))

        test_measures[name]['accr'] = best_accr
        test_measures[name]['f1'] = best_macro_f1



    # ========================================
    #              Dummy Test
    # ========================================
        if name == 'indomain':
            dummy_clf = DummyClassifier(strategy="uniform")
            dummy_clf.fit(X_train, y_train)
            predictions_dummy = dummy_clf.predict(X_test)

            dummy_accr = np.sum(predictions_dummy == y_test) / y_test.shape[0]
            dummy_macro_f1 = f1_score(y_test, predictions_dummy, average='macro')
            print("  Dummy Accuracy: {0:.2f}".format(dummy_accr))
            print("  Dummy F1-MACRO: {0:.2f}".format(dummy_macro_f1))
        else:
            dummy_clf = DummyClassifier(strategy="uniform")
            dummy_clf.fit(sentences, labels)
            predictions_dummy = dummy_clf.predict(sentences)

            dummy_accr = np.sum(predictions_dummy == labels) / labels.shape[0]
            dummy_macro_f1 = f1_score(labels, predictions_dummy, average='macro')
            print("  Dummy Accuracy: {0:.2f}".format(dummy_accr))
            print("  Dummy F1-MACRO: {0:.2f}".format(dummy_macro_f1))            
        test_measures[name]['dummy_accr'] = dummy_accr
        test_measures[name]['dummy_f1'] = dummy_macro_f1


    return {
        'seed': seed,
        # 'best_accr': test_measures['indomain']['accr'],
        # 'best_f1': test_measures['indomain']['f1'],
        # 'dummy_accr': test_measures['indomain']['dummy_accr'],
        # 'dummy_f1': test_measures['indomain']['dummy_f1'],
        'best_accr_sw': test_measures['ood_sw']['accr'],
        'best_f1_sw': test_measures['ood_sw']['f1'],
        'dummy_accr_sw': test_measures['ood_sw']['dummy_accr'],
        'dummy_f1_sw': test_measures['ood_sw']['dummy_f1'],
        'best_accr_bs': test_measures['ood_bs']['accr'],
        'best_f1_bs': test_measures['ood_bs']['f1'],
        'dummy_accr_bs': test_measures['ood_bs']['dummy_accr'],
        'dummy_f1_bs': test_measures['ood_bs']['dummy_f1'],
        # 'majority': y_majority
    }
    

def run_many_times(file_path):
    run_time = 50
    start_seed = 150
    result_record = pd.DataFrame([], columns=['seed', 'best_accr', 'best_f1', \
        'dummy_accr', 'dummy_f1', 'best_accr_sw', 'best_f1_sw',\
        'dummy_accr_sw', 'dummy_f1_sw', 'best_accr_bs', 'best_f1_bs',\
        # 'best_accr_mmatch', 'best_f1_mmatch', 'best_accr_mismatch', 'best_f1_mismatch',\
        'dummy_accr_bs', 'dummy_f1_bs'])
    testloaders = {
        'ood_sw': load_test_data('./data/valid_data_cond1.csv'),
        'ood_bs': load_test_data('./data/valid_data_baseline.csv')
    }
    

    sentences, labels = load_data(file_path)
    logit_df = pd.DataFrame(0, index =np.arange(len(sentences)), columns=[0,1,2])
    easy_and_hard_df = pd.DataFrame(0, index =np.arange(len(sentences)), columns=[0,1])
    for i in range(run_time):
        result_df = run(sentences, labels, testloaders, logit_df, seed=start_seed+i)
        # for idx in easy_examples:
        #     easy_and_hard_df.iloc[idx,0] += 1
        # for idx in hard_examples:
        #     easy_and_hard_df.iloc[idx,1] += 1

        result_record = result_record.append(result_df, ignore_index=True)
        result_record.to_csv('./output/result_multiart_ood_{0}.csv'.format(start_seed))
        # result_record.to_pickle('./output/result_multiart.pkl')
        # logit_df.to_csv('./output/result_multiart_logit_ood_{0}.csv'.format(start_seed))
        # easy_and_hard_df.to_csv('./output/result_multiart_eandh_ood_{0}.csv'.format(start_seed))
        print("RUNTIME is", i)



def load_test_data(file_path, batch_size=32, tsv=False):


    df = pd.read_csv(file_path)
    sentences = df.h.values
    raw_labels = df.label.values


    tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
    encoding = tokenizer(list(sentences), return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)

    input_ids = encoding['input_ids']
    attention_masks = encoding['attention_mask']
    labels = torch.tensor(raw_labels).unsqueeze(1).to(device)
    test_dataset = TensorDataset(input_ids, attention_masks, labels)
    
    # Dataloading
    prediction_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=prediction_sampler, batch_size=batch_size)

    return (test_dataloader, sentences, raw_labels)



def load_data(file_path):
    # base_df = pd.read_csv('data/valid_data_multiart.csv')
    df = pd.read_csv(file_path)
    
    sentences = df.h.values
    labels = df.label.values
    return sentences, labels



def main():
    file_path = 'data/valid_data_multiart.csv'
    run_many_times(file_path)
    

if __name__ == "__main__":
    main()