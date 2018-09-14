import torch
import os
import random
import itertools
import torch.utils.data as data

class babi_Dataset(data.Dataset):
    def __init__(self, data):
        assert data
        # assert len(data[0]) >= batch_size, "Training Data Size (%d) < Batch Size (%d)" % (len(data[0]), batch_size)

        self.story = data['story_batches']
        self.question = data['question_batches']
        self.answer = data['answer_batches']
        self.story_lengths = data['story_batches_lengths']
        self.story_sent_lengths = data['story_sent_batches_lengths']
        self.question_lengths = data['question_batches_lengths']

    def __getitem__(self, index):
        return self.story[index], self.question[index], self.answer[index], \
               self.story_lengths[index], self.story_sent_lengths[index], self.question_lengths[index]

    def __len__(self):
        return len(self.story)

def make_batch(l, n, shuffle=False):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

# Define sub-functions (helpers)
def zero_pad_batch(batch_list, max_length):  # List of sentences that are in varying size
    zero_padded_tensors = []
    padded = []
    for story in batch_list:
        tensor = torch.Tensor(story)
        padded += [torch.cat([tensor, tensor.new(max_length - tensor.size(0), *tensor.size()[1:]).zero_()]).view(1, max_length)]
        # zero_padded_tensors += [torch.cat(padded, dim=0)]  # [LongTensor Batch x Max_Seq_Len of the Batch]
    zero_padded_batch = torch.cat(padded, dim=0)

    return zero_padded_batch

def clean(batches):
    story_batches = []
    question_batches = []
    answer_batches = []
    story_sent_batches_lengths = []
    story_batches_lengths = []
    question_batches_lengths = []
    story_sent_batches_lengths_tensors = []
    story_batches_lengths_tensors = []
    question_batches_lengths_tensors = []
    answer_vocab = []
    for i, batch in enumerate(batches):
        answer_batch = []
        for line in batch:
            answer_batch += [int(line.rstrip('\n').split("::")[2])]
        for answer in answer_batch:
            if answer not in answer_vocab:
                answer_vocab.append(answer)

    answer_vocab = sorted(answer_vocab)

    for i, batch in enumerate(batches):
        # print (batch)
        story_batch = []
        question_batch = []
        answer_batch = []
        for line in batch:
            story_ = line.rstrip('\n').split("::")[0].split("\t")
            story_batch += [[list(map(int, sent.split(' '))) for sent in story_]]
            question_batch += [list(map(int, line.rstrip('\n').split("::")[1].split(' ')))]
            answer_batch += [int(line.rstrip('\n').split("::")[2])]

        # Convert the # Change the answer index to be within number of output vocab size
        answer_batch = [answer_vocab.index(answer) for answer in answer_batch]

        story_batch_ = [list(itertools.chain(*story)) for story in story_batch]  # Concat the story into one long sentence.

        # Get sentence lengths in the batch.
        story_sent_batches_lengths += [[[len(sent) for sent in story] for story in story_batch]]
        story_batches_lengths += [[sum(story) for story in story_sent_batches_lengths[i]]]
        question_batches_lengths += [[len(question) for question in question_batch]]

        # Get maximum sent (or sequence) length of the batch.
        max_batch_story_len = max(story_batches_lengths[i])  # Find the maximum length of all sentences combined in the batch.
        max_batch_question_len = max(question_batches_lengths[i])  # Find the maximum sequence length of the batch.

        story_batches_lengths_tensors += [torch.Tensor([len(story) for story in story_batch_])]
        story_sent_batches_lengths_tensors += [[torch.Tensor([len(sent) for sent in story]) for story in story_batch]]
        question_batches_lengths_tensors += [torch.Tensor([len(question) for question in question_batch])]
        # Zero pad, convert to tensor, and convert to tensor variable
        story_batch = zero_pad_batch(story_batch_, max_batch_story_len)  # [Variable LongTensor (Num_Sent_Combined, Max_Sent_Len)]
        question_batch = zero_pad_batch(question_batch, max_batch_question_len)  # [Variable LongTensor (Batch, Max_Question_Len)]
        answer_batch = torch.LongTensor(answer_batch)  # [Variable LongTensor (N)]

        story_batches.append(story_batch)
        question_batches.append(question_batch)
        answer_batches.append(answer_batch)

    return story_batches, question_batches, answer_batches, \
           story_batches_lengths_tensors, story_sent_batches_lengths_tensors, question_batches_lengths_tensors, \
           sorted(answer_vocab)


def get_data(filepath, batch_size, shuffle=True):
    new_filename = '%s-batchsize_%s.pt' % (filepath.split('/')[-1].split('.')[0], batch_size)

    if os.path.exists('../datasets/%s' % new_filename) and shuffle is False:
        print ("\n%s Exists. Loading from it..." % new_filename)
        data = torch.load('../datasets/%s' % new_filename)
        # print ("\nLOADED DATA (Number of data = %d)" % len(lines))
        print ("Train Batch Size = %d" % len(data['story_batches'][0]))
        print ("Number of Train Batches = %d " % len(data['story_batches']))
        print ("\nData Loading Done!")
    else:
        # If loading the new data, apply some cleaning to acquire data that are appropriate for the task.
        print ("\nThe loaded data will be save into a new file, %s" % new_filename)

        with open(filepath, 'r') as f:
            lines = f.readlines()  # Read the whole corpus
        if shuffle is True:
            print ("\nShuffling data...")
            random.shuffle(lines)

        batches = list(make_batch(lines, batch_size))
        story_batches, question_batches, answer_batches, \
        story_batches_lengths, story_sent_batches_lengths, question_batches_lengths, \
        answer_vocab = clean(batches)

        data = {}
        data['story_batches'] = story_batches
        data['question_batches'] = question_batches
        data['answer_batches'] = answer_batches
        data['story_batches_lengths'] = story_batches_lengths
        data['story_sent_batches_lengths'] = story_sent_batches_lengths
        data['question_batches_lengths'] = question_batches_lengths

        print ("\nLOADED DATA (Number of data = %d)" % len(lines))
        print ("Batch Size = %d" % batch_size)
        print ("Number of Train Batches = %d " % len(data['story_batches']))

        torch.save(data, '../datasets/%s' % new_filename)
        print ("Also, data is saved in %s" % new_filename)

    return data, answer_vocab

def tensor_data_loader(filepath, batch_size, shuffle=True, num_workers=2):
    babi_data, answer_vocab = get_data(filepath, batch_size, shuffle)

    # Load wmt_dataset
    babi = babi_Dataset(
        data=babi_data
    )

    data_loader = torch.utils.data.DataLoader(
        dataset=babi,
        batch_size=1,
        shuffle=shuffle,
        num_workers=num_workers
    )

    return data_loader, answer_vocab

def main():
    # For Unit Testing
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-train_data_path', default='../datasets/train_qa3.txt')
    parser.add_argument('-test_data_path', default='../datasets/test_qa3.txt')
    parser.add_argument('-batch_size', default=4)
    args = parser.parse_args()

    tensor_data_train, train_answer_vocab = tensor_data_loader(args.train_data_path, args.batch_size)
    tensor_data_test, validation_answer_vocab = tensor_data_loader(args.test_data_path, args.batch_size)
    print (train_answer_vocab)
    print (validation_answer_vocab)
    assert train_answer_vocab == validation_answer_vocab, \
        "Train and validation answer vocab size (%d and %d) do not match." % (
        len(train_answer_vocab), len(validation_answer_vocab))

    print (tensor_data_train)
    exit()
    for i, batch in enumerate(tensor_data_train):
        print (len(batch[0]))
        exit()
    #     story, question, answer = batch[0], batch[1], batch[2]
    #     story_lengths, story_sent_lengths, question_lengths = batch[3].long(), batch[4], batch[5]
    #     story_sent_lengths = [list(map(int, sent_length.tolist()[0])) for sent_length in story_sent_lengths]
    #     question_lengths = question_lengths.tolist()[0]
        # print (story_lengths)
        # print (story_sent_lengths)
        # exit()
        # print (story[0].size())
        # print (question[0].size())
        # print (answer[0])
        # print (story_lengths)
        # print (question_lengths)
        if i > 2: exit()

if __name__=='__main__':
    main()