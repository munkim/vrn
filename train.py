# From the python package
from collections import OrderedDict
import argparse
import os
import numpy as np
import time

# From pytorch package
import torch
import torch.nn as nn
from torch.autograd import Variable

# From the directory
from model import VariousRelationsNet
from data_loader import tensor_data_loader


# Define sub-functions (helpers)
def zero_pad_batch(batch_list, max_length, batch_type=None):  # List of sentences that are in varying size
    zero_padded_tensors = []
    if batch_type == 'story':
        for batch in batch_list:
            padded = []
            for sentence in batch:
                tensor = torch.LongTensor(sentence)
                padded += [
                    torch.cat([tensor, tensor.new(max_length - tensor.size(0), *tensor.size()[1:]).zero_()]).view(1,
                                                                                                                  max_length)]
            zero_padded_tensors += [torch.cat(padded, dim=0)]  # [LongTensor Batch x Max_Sent_Len of the Batch]
        zero_padded_batch = torch.cat(zero_padded_tensors, 0)
    elif batch_type == 'question':
        padded = []
        for question in batch_list:
            tensor = torch.LongTensor(question)
            padded += [torch.cat([tensor, tensor.new(max_length - tensor.size(0), *tensor.size()[1:]).zero_()]).view(1,
                                                                                                                     max_length)]
        zero_padded_batch = torch.cat(padded, dim=0)  # [LongTensor Batch x Max_Sent_Len of the Batch]

    return zero_padded_batch

def validate(vrn, validation_batches, log_file):
    print ("\n\n(PAUSE) Calculating Validation Accuracy")
    num_validation_batches = len(validation_batches)
    total = []
    corrects = []
    accu = []
    for i, batch in enumerate(validation_batches):
        story_batch = Variable(batch[0][0].long())  # [LongTensor (Num_Sent_Combined, Max_Sent_Len)]
        question_batch = Variable(batch[1][0].long())  # [LongTensor (Batch, Max_Question_Len)]
        answer_batch = Variable(batch[2].long())  # [LongTensor (N)]
        story_batch_lengths = batch[3][0].long()

        # Get sentence lengths in the batch.
        story_sent_batch_lengths = [list(map(int, story_length.tolist()[0])) for story_length in batch[4]]  # List of sentences lengths of each batch
        question_batch_lengths = list(map(int, batch[5].tolist()[0]))  # List of question lengths

        if torch.cuda.is_available():
            story_batch = story_batch.cuda()
            question_batch = question_batch.cuda()
            answer_batch = answer_batch.cuda()

        # Get the prediction (output) from the model
        predicted_batch = vrn(story_batch, story_batch_lengths, story_sent_batch_lengths, question_batch,
                              question_batch_lengths)

        # Calculate accuracy
        predicted_batch_np = torch.max(predicted_batch, 1)[1].cpu().data.numpy().squeeze()
        scores = np.array(predicted_batch_np == answer_batch.data.tolist()[0], int)
        total += [len(scores)]
        corrects += [sum(scores)]
        accu += [corrects[i] * 1.0 / total[i]]
        print ('\nVALIDATION Step [%d/%d]: ACCU %.4f' % (i, num_validation_batches, accu[i]))

    validation_avg_accu = sum(accu) / float(len(accu))
    return validation_avg_accu

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-train_data_path', default='../datasets/train_qa1.txt')
    parser.add_argument('-test_data_path', default='../datasets/test_qa1.txt')
    parser.add_argument('-save_path', default='./save')
    parser.add_argument('-save_every_epoch', default=True, action='store_false')
    parser.add_argument('-save_uniquely', default=False, action='store_true')
    parser.add_argument('-save_start_epoch', default=20)
    parser.add_argument('-batch_size', default=32)
    parser.add_argument('-lr_rate', default=0.0001)
    parser.add_argument('-num_epochs', default=100)
    parser.add_argument('-comment', default='None')
    args = parser.parse_args()


    # Load training data and answer vocab
    training_batches, train_answer_vocab = tensor_data_loader(args.train_data_path, int(args.batch_size), shuffle=True)
    validation_batches, validation_answer_vocab = tensor_data_loader(args.test_data_path, int(args.batch_size))
    num_training_batches = len(training_batches)

    # Load vocabs
    w2i = np.load('../datasets/word2idx.npy')
    i2w = np.load('../datasets/idx2word.npy')
    vocab_size = len(w2i.tolist())

    # Hyperparameters
    hidden_size = 1024
    max_num_hiddens = 25
    lstm_n_layers = 2
    lstm_directions = 2

    assert train_answer_vocab == validation_answer_vocab, \
        "Train and validation answer vocab size (%d and %d) do not match." % (len(train_answer_vocab), len(validation_answer_vocab))

    # Define model configurations (i.e. hyperparameters)
    vrn_config = OrderedDict([('input_vocab_size', vocab_size),
                              ('question_vocab_size', vocab_size),
                              ('output_vocab_size', len(validation_answer_vocab)),
                              ('hidden_size', hidden_size),
                              ('max_num_hiddens', max_num_hiddens),
                              ('lstm_n_layers', lstm_n_layers),
                              ('lstm_directions', lstm_directions),
                              ('batch_size', int(args.batch_size))
                              ])

    config_log = ("Train Data Path: %s\n" % args.train_data_path
                  + "Test Data Path: %s\n" % args.test_data_path
                  + "Save Every Epoch: %s\n" % args.save_every_epoch
                  + "Save Unique: %s\n" % args.save_every_epoch
                  + "Save Start Epoch: %d\n" % int(args.save_start_epoch)
                  + "Learning Rate: %f\n" % float(args.lr_rate)
                  + "Num_Epoch: %s\n" % int(args.num_epochs)
                  + "Batch_Size: %d \n" % int(args.batch_size)

                  + "\nInput Vocab Size: %d \n" % vocab_size
                  + "Question Vocab SIze: %d \n" % vocab_size
                  + "Output Vocab Size: %d \n" % len(validation_answer_vocab)
                  + "Hidden_Size: %d \n" % hidden_size
                  + "Max_Num_Hiddens: %d \n" % max_num_hiddens
                  + "LSTM_N_Layers: %d \n" % lstm_n_layers
                  + "LSTM_Directions: %d \n" % lstm_directions

                  + "\nComment: %s" % args.comment
                  )

    # Load model
    vrn = VariousRelationsNet(vrn_config=vrn_config)
    if torch.cuda.is_available():
        print("CUDA enabled.")
        vrn.cuda()

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Define optimizer
    optimizer = torch.optim.SGD(
        vrn.parameters(),
        lr=float(args.lr_rate)
    )

    QA_task_number = args.train_data_path.split('/')[-1].split('.')[0].split('_')[1]

    # Prepare a txt file to print training log
    if args.save_uniquely is True:
        args.save_path = '%s/%02d%02d%02d%02d' % (args.save_path, time.localtime()[1], time.localtime()[2],
                                                  time.localtime()[3], time.localtime()[4])

    if not os.path.exists(args.save_path):
        print ("\nCreated a directory (%s) for saving logs and trained models." % args.save_path)
        os.makedirs(args.save_path)

    save_model_path = (args.save_path + '/' + QA_task_number)

    if not os.path.exists(save_model_path):
        print ("\nSaving model in %s for QA Task %s." % (save_model_path, QA_task_number))
        os.makedirs(save_model_path)

    log_file = open('%s/%s_train_log.txt' % (args.save_path, QA_task_number), 'w')

    # Print Log
    print (config_log)
    log_file.write(config_log)
    log_file.flush()

    for epoch in range(int(args.num_epochs)):
        total = []
        corrects = []
        accu = []
        avg_accu = []
        for i, batch in enumerate(training_batches):
            story_batch = Variable(batch[0][0].long())  # [LongTensor (Num_Sent_Combined, Max_Sent_Len)]
            question_batch = Variable(batch[1][0].long())  # [LongTensor (Batch, Max_Question_Len)]
            answer_batch = Variable(batch[2].long())  # [LongTensor (N)]
            story_batch_lengths = batch[3][0].long()

            # Get sentence lengths in the batch.
            story_sent_batch_lengths = [list(map(int, story_length.tolist()[0])) for story_length in batch[4]]  # List of sentences lengths of each batch
            question_batch_lengths = list(map(int, batch[5].tolist()[0]))  # List of question lengths

            contexts = Variable(batch[0][0].long())  # [LongTensor (Batch * Num Sentences for Each Context, Max_Sent_Len)]
            questions = Variable(batch[3][0].long())  # [LongTensor (Batch)]
            answers = Variable(batch[4][0].long())  # [LongTensor (Batch, Max_Question_len)]
            context_num_sent = batch[1][0]  # [FloatTensors (Batch)]
            context_sent_lengths = batch[2]  # List (Batch Size) of [FloatTensors (1, Num_Sent)]



            if torch.cuda.is_available():
                story_batch = story_batch.cuda()
                question_batch = question_batch.cuda()
                answer_batch = answer_batch.cuda()

            optimizer.zero_grad()

            # Get the prediction (output) from the model
            predicted_batch = vrn(story_batch, story_batch_lengths, story_sent_batch_lengths, question_batch, question_batch_lengths)

            # Calculate Loss
            answer_batch = answer_batch.contiguous().view(-1).long()
            loss = criterion(predicted_batch, answer_batch)

            # Perform backprop
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            predicted_batch_np = torch.max(predicted_batch, 1)[1].cpu().data.numpy().squeeze()
            scores = np.array(predicted_batch_np == answer_batch.data.tolist(), int)
            total += [len(scores)]
            corrects += [sum(scores)]
            accu += [corrects[i]*1.0/total[i]]

            # Print log info to both console and file
            if i % int(args.batch_size) == 0:
                print("\n\n#####################################################")
                log = ('\nEpoch [%d/%d], Step [%d/%d], Loss: %.4f, Accu: %.4f'
                       % (epoch, int(args.num_epochs), i, num_training_batches, loss.data[0], accu[i]))
                print(log)
                log_file.write("{}".format(log))

            print ('\nTraining Step %d: LOSS %.4f \t ACCU %.4f' % (i, loss.data.tolist()[0], accu[i]))

        if (epoch >= int(args.save_start_epoch)) or (args.save_every_epoch is True):
            # For each epoch calculate train avg accu
            train_avg_accu = sum(accu) / float(len(accu))
            train_avg_log = ('\nTrain Average Accuracy: %.4f' % train_avg_accu)
            log_file.write(train_avg_log)

            # For each epoch calculate validation avg accu
            validation_avg_accu = validate(vrn, validation_batches, log_file)
            validation_log = ('\n---- Validation Average Accuracy: %.4f\n' % validation_avg_accu)
            print (validation_log)
            log_file.write(validation_log)
            log_file.flush()

            save_model = '%s/%s_epoch_%d_acc_%1.3f.pth' % (save_model_path,
                                                           QA_task_number,
                                                           epoch,
                                                           validation_avg_accu)
            print ("\n\nSaving the model: %s" % save_model)
            torch.save(obj=vrn, f=save_model)


if __name__ == '__main__':
    main()