import numpy as np
import random

import torch
import torch.nn as nn
from torch.autograd import Variable
import itertools
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
from pprint import pprint

""" 
#### MODULES ####
"""


class InputModule(nn.Module):
    def __init__(self, input_vocab_size, hidden_size, n_layers=2, lstm_direction=2):
        super(InputModule, self).__init__()
        self.n_layers = n_layers
        self.lstm_direction = lstm_direction
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers, batch_first=False, bidirectional=True)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, story_batch, story_batch_lengths, story_sent_batch_lengths):
        """
        :param
            Story: [Variable LongTensor (Num_Sent_Combined, Max_Sent_Len)] (i.e. Num_Sent_Combined = Num_Sent (Batch 1) + Num_Sent (Batch 2) ...)
            story_batch_lengths: List of story sentences lengths (i.e. [ [10, 10, 10], [9, 9, 9] ], Batch size of 2)
        :return:
            outputs: [Variable FloatTensor (Num_Hiddens, Num_Layer * Num_Directions, Hidden_Size)]
        """

        # Define sub-function (helper)
        def get_position_encoding(batch_size, sequence_length, embedding_size):
            position_encoding = torch.zeros(sequence_length, embedding_size)
            for pos in range(sequence_length):
                radians = torch.FloatTensor([pos / 1000 ** (2 * i / embedding_size) if pos != 0 else 0 for i in range(embedding_size)])
                position_encoding[pos] = torch.sin(radians) if pos % 2 == 0 else torch.cos(radians)
                # position_encoding[pos] = torch.add(radians, pos)
            if torch.cuda.is_available():
                position_encoding = position_encoding.cuda()
            position_encoding = Variable(position_encoding.unsqueeze(0).expand(batch_size, sequence_length, embedding_size))
            # print (position_encoding.transpose(0, 1))
            # exit()
            return position_encoding.transpose(0, 1)

        sorted_seq_lengths, perm_idx = story_batch_lengths.sort(0, descending=True)
        reverse_perm_idx = np.argsort(perm_idx.numpy())  # Get inverse permutation indexes. (To sort to the original order). Reference: https://stackoverflow.com/questions/11649577/how-to-invert-a-permutation-array-in-numpy
        reverse_perm_idx = torch.LongTensor(reverse_perm_idx)  # Convert to tensor since the tensor.sort likes idx that are in tensor
        if torch.cuda.is_available():
            sorted_seq_lengths = sorted_seq_lengths.cuda()
            perm_idx = perm_idx.cuda()
            reverse_perm_idx = reverse_perm_idx.cuda()

        # Sort the batch
        sorted_story_batch = story_batch[perm_idx]
        # Get embedded input
        embedded = self.embedding(sorted_story_batch).transpose(0, 1)  # [FloatTensor (Max_Sent_Len, Batch_Size, Hidden_Size)]

        # Get Position Encoding
        pos_enc = get_position_encoding(embedded.size(1), embedded.size(0), embedded.size(2))  # Exponential and Sinusoidal PE
        lstm_input = torch.add(embedded, pos_enc) # [FloatTensor (Max_Sent_Len, Num_Sent, Hidden_Size)]
        # Pack embeddings according to the original encoding
        packed_lstm_input = rnn_utils.pack_padded_sequence(lstm_input, sorted_seq_lengths.tolist())

        # Initial States
        hidden = Variable(torch.zeros(self.n_layers * self.lstm_direction, lstm_input.size(1), self.hidden_size))  # [Variable FloatTensor (Num_Layers * Num_Directions, Batch_Size, Hidden_Size)]
        cell = Variable(torch.zeros(self.n_layers * self.lstm_direction, lstm_input.size(1), self.hidden_size))
        if torch.cuda.is_available():
            hidden, cell = hidden.cuda(), cell.cuda()

        # # Get hidden states
        outputs, (_, _) = self.lstm(packed_lstm_input, (hidden, cell))  # PackedSequence Object
        unpacked_outputs, _ = rnn_utils.pad_packed_sequence(outputs)  # [Variable FloatTensor (Max_Sent_Len, Batch_Size, Hidden_Size * Num_Directions)]
        unpacked_outputs = self.dropout(unpacked_outputs)
        reverse_permuted_outputs = unpacked_outputs.transpose(0,1)[reverse_perm_idx]  # [Variable FloatTensor (Batch_Size, Max_Sent_Len, Hidden_Size * Num_Directions])

        batch_hidden_states = []
        for i, batch in enumerate(reverse_permuted_outputs):
            # Each batch has (Max_Sent_Len, Hidden_Size * Num_Directions)
            current_hidden_idx = 0
            hidden_states = []
            for sent_len in story_sent_batch_lengths[i]:
                current_hidden_idx += sent_len
                hidden_states += [batch[current_hidden_idx - 1].unsqueeze(0)]  # List of hidden_states for each story [Variable FloatTensor (1, Hidden_Size * Num_Directions)]
            batch_hidden_states += [torch.cat(hidden_states, dim=0)]  # List (:batch size) of [Variable FloatTensor (Num_Sent, Hidden_Size * Num_Directions)]

        # To alleviate the memory burdening
        del hidden_states, sorted_seq_lengths, perm_idx

        return batch_hidden_states  # List (:batch size) of [Variable FloatTensor (Num_Sent, Hidden_Size * Num_Directions)]



class QuestionModule(nn.Module):
    def __init__(self, question_vocab_size, hidden_size, n_layers=2, rnn_direction=2):
        super(QuestionModule, self).__init__()
        self.hidden_size = hidden_size
        self.rnn_direction = rnn_direction
        self.n_layers = n_layers
        self.embedding = nn.Embedding(question_vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers, batch_first=False, bidirectional=True)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, question_batch, question_batch_lengths):
        '''
        :param
            question: [Variable LongTensor (Batch Size, Max_Seq_Len)]
            question_batch_lengths: List of question lengths (i.e. [4, 4, 4], Batch size=3)
        :return:
            outputs: [Variable FloatTensor (1, Num_Layers * Num_Directions, Hidden_Size)]
        '''
        question_batch_lengths = torch.LongTensor(question_batch_lengths)
        sorted_seq_lengths, perm_idx = question_batch_lengths.sort(0, descending=True)
        reverse_perm_idx = torch.LongTensor(np.argsort(perm_idx.numpy()))
        if torch.cuda.is_available():
            sorted_seq_lengths = sorted_seq_lengths.cuda()
            perm_idx = perm_idx.cuda()
            reverse_perm_idx = reverse_perm_idx.cuda()

        permuted_question_batch = question_batch[perm_idx]
        embedded = self.embedding(permuted_question_batch).transpose(0, 1)  # [Variable FloatTensor (Max_Seq_Len, Num_Sent, Hidden_Size)]
        packed_lstm_input = rnn_utils.pack_padded_sequence(embedded, sorted_seq_lengths.tolist())  # PackedSequence Object

        hidden = Variable(torch.zeros(self.n_layers * self.rnn_direction, embedded.size(1), self.hidden_size))
        cell = Variable(torch.zeros(self.n_layers * self.rnn_direction, embedded.size(1), self.hidden_size))
        if torch.cuda.is_available():
            hidden, cell = hidden.cuda(), cell.cuda()

        outputs, (_, _) = self.lstm(packed_lstm_input, (hidden, cell))  # PackedSequence Object
        unpacked_outputs, _ = rnn_utils.pad_packed_sequence(outputs)  # [Variable FloatTensor (Max Seq_Len, Batch, Hidden_Size * Num_Directions)]
        unpacked_outputs = self.dropout(unpacked_outputs)
        reverse_permuted_outputs = unpacked_outputs.transpose(0, 1)[reverse_perm_idx]  # [Variable FloatTensor (Batch, Max_Seq_Len, Hidden_Size * Num_Directions])

        batch_hidden_states = [reverse_permuted_outputs.unsqueeze(0)[:, sent, length - 1, :]
                         for sent, length in zip(range(reverse_permuted_outputs.size(0)), question_batch_lengths)]

        return batch_hidden_states  # List (:batch size) of [Variable FloatTensor (1, Sent_Len, Hidden_Size * Num_Directions)]


""" 
#### MODEL #### 
"""

class VariousRelationsNet(nn.Module):
    def __init__(self, vrn_config):
        super(VariousRelationsNet, self).__init__()
        self.input_vocab_size, self.question_vocab_size, \
        self.output_vocab_size, self.hidden_size, \
        self.max_n_hiddens, self.n_layers, self.directions, self.batch_size \
            = vrn_config.values()

        self.input_module = InputModule(self.input_vocab_size, self.hidden_size, self.n_layers, self.directions)
        self.question_module = QuestionModule(self.question_vocab_size, self.hidden_size, self.n_layers, self.directions)

        input_channel = 2 * self.hidden_size * self.directions  # For Story_Hiddens, Reversed_Story_Hiddens, Question_Hidden_Duplicated
        kernel_size_s = (2, 1)  # (Hidden_Size, Window_Size)
        kernel_size_m = (3, 1)
        kernel_size_l = (2, 2)

        self.conv_s1 = nn.Sequential(nn.Conv2d(input_channel, 256, kernel_size_s, stride=1),
                                     nn.ReLU(),
                                     nn.BatchNorm2d(256),
                                     nn.Dropout2d(p=0.5)  # Num_Filter = Window_Size + 1
                                     )
        self.conv_m1 = nn.Sequential(nn.Conv2d(input_channel, 256, kernel_size_m, stride=1),
                                     nn.ReLU(),
                                     nn.BatchNorm2d(256),
                                     nn.Dropout2d(p=0.5)  # Num_Filter = Window_Size + 1
                                     )
        self.conv_l1 = nn.Sequential(nn.Conv2d(input_channel, 256, kernel_size_l, stride=1),
                                     nn.ReLU(),
                                     nn.BatchNorm2d(256),
                                     nn.Dropout2d(p=0.5)   # Num_Filter = Window_Size + 1
                                     )

        self.conv_s2 = nn.Sequential(nn.Conv2d(256, 256, 1, stride=1),
                                     nn.ReLU(),
                                     nn.BatchNorm2d(256),
                                     nn.Dropout2d(p=0.5)
                                     )
        self.conv_m2 = nn.Sequential(nn.Conv2d(256, 256, 1, stride=1),
                                     nn.ReLU(),
                                     nn.BatchNorm2d(256),
                                     nn.Dropout2d(p=0.5)
                                     )
        self.conv_l2 = nn.Sequential(nn.Conv2d(256, 256, 1, stride=1),
                                     nn.ReLU(),
                                     nn.BatchNorm2d(256),
                                     nn.Dropout2d(p=0.5)
                                     )

        self.conv_s3 = nn.Sequential(nn.Conv2d(256, 256, 1, stride=1),
                                     nn.ReLU(),
                                     nn.BatchNorm2d(256),
                                     nn.Dropout2d(p=0.5)
                                     )
        self.conv_m3 = nn.Sequential(nn.Conv2d(256, 256, 1, stride=1),
                                     nn.ReLU(),
                                     nn.BatchNorm2d(256),
                                     nn.Dropout2d(p=0.5)
                                     )
        self.conv_l3 = nn.Sequential(nn.Conv2d(256, 256, 1, stride=1),
                                     nn.ReLU(),
                                     nn.BatchNorm2d(256),
                                     nn.Dropout2d(p=0.5)
                                     )

        # The size of conv_s output transposed is (C=25, H=3, W=256)
        self.conv_s4 = nn.Sequential(nn.Conv2d(256, 256, 1, stride=1),
                                     nn.ReLU(),
                                     nn.BatchNorm2d(256),
                                     nn.Dropout2d(p=0.5)
                                     )
        # The size of conv_m output transposed is (C=25, H=2, W=256)
        self.conv_m4 = nn.Sequential(nn.Conv2d(256, 256, 1, stride=1),
                                     nn.ReLU(),
                                     nn.BatchNorm2d(256),
                                     nn.Dropout2d(p=0.5)
                                     )
        # The size of conv_l output transposed  is (C=25, H=3, W=256)
        self.conv_l4 = nn.Sequential(nn.Conv2d(256, 256, 1, stride=1),
                                     nn.ReLU(),
                                     nn.BatchNorm2d(256),
                                     nn.Dropout2d(p=0.5)
                                     )

        ''' Inverted Position-wise Convolution (Doesn't make sense, but works well in some tasks) '''
        self.position_conv_s = nn.Sequential(nn.Conv2d(25, 1, 1, stride=1),
                                             nn.BatchNorm2d(1),
                                             nn.Dropout2d(p=0.5))

        self.position_conv_m = nn.Sequential(nn.Conv2d(25, 1, 1, stride=1),
                                             nn.BatchNorm2d(1),
                                             nn.Dropout2d(p=0.5))

        self.position_conv_l = nn.Sequential(nn.Conv2d(24, 1, 1, stride=1),
                                             nn.Dropout2d(p=0.5),
                                             nn.BatchNorm2d(1))

        self.linear = nn.Sequential(
            nn.Linear((3+2+3)*256, (3+2+3)*256),  # The height of all position wise conv are concatenated together (3*25 + 2*25 + 3*24)
            nn.Linear((3+2+3)*256, 256),
            nn.Linear(256, self.output_vocab_size)
        )

        ''' Depth-wise Convolution '''
        self.conv_s = nn.Sequential(nn.Conv2d(256, 256, (3, 25), stride=1, groups=256),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(256),
                                    nn.Dropout2d(p=0.5)
                                    )
        self.conv_m = nn.Sequential(nn.Conv2d(256, 256, (2, 25), stride=1, groups=256),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(256),
                                    nn.Dropout2d(p=0.5)
                                    )
        self.conv_l = nn.Sequential(nn.Conv2d(256, 256, (3, 24), stride=1, groups=256),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(256),
                                    nn.Dropout2d(p=0.5)
                                    )

        ''' Depth-wise Summation '''
        # Not Necessary

        ''' Depth-wise MaxPooling '''
        # self.maxpool_s = nn.MaxPool2d((3, 25))
        # self.maxpool_m = nn.MaxPool2d((2, 25))
        # self.maxpool_l = nn.MaxPool2d((3, 24))

        ''' For Any Depth-wise Operations '''
        # # Either Depth-wise Summation, Depth-wise Convolution, Depth-wise Max Pooling
        # self.linear_s = nn.Sequential(
        #     nn.Linear(256, 256),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(256, 256),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(256, 256),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.5)
        # )
        #
        # self.linear_m = nn.Sequential(
        #     nn.Linear(256, 256),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(256, 256),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(256, 256),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.5)
        # )
        #
        # self.linear_l = nn.Sequential(
        #     nn.Linear(256, 256),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(256, 256),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(256, 256),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.5)
        # )
        #
        # self.linear = nn.Sequential(
        #     nn.Linear(3*256, 3*256),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(3*256, 3*256),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(3*256, self.output_vocab_size)
        # )



    def forward(self, story_batch, story_batch_lengths, story_sent_batch_lengths, question_batch, question_batch_lengths):
        """
        :param
            story: [LongTensor (Combined_Sent_Length, Max_Sent_Len)
            question: [LongTensor (Batch, Sent_Len)]
        :return:
            output: [Variable FloatTensor (Output_Vocab_Size)]
        """

        # Define sub-functions (helpers)
        def pad_to_max(tensor, max_n_hiddens, pad_tensor=None):
            """Tensor: [Variable FloatTensor (1 or Sent_Len, Hidden_Size * Num_Directions)]"""
            if pad_tensor is None:
                pad_tensor = Variable(torch.zeros((1, tensor.size(1))))  # Zero Padding

            if torch.cuda.is_available():
                tensor = tensor.cuda()
                pad_tensor = pad_tensor.cuda()

            padded_tensor = tensor

            for i in range(max_n_hiddens - tensor.size(0)):
                padded_tensor = torch.cat((padded_tensor, pad_tensor), dim=0)
            if torch.cuda.is_available():
                padded_tensor = padded_tensor.cuda()

            return padded_tensor

        def create_relation_map(batch_story_states, batch_question_states, max_n_hiddens):
            """
            :param
                batch_story_states: List of (:Batch_Size) [Variable FloatTensor (Sent_Len, Story_Hidden_Size)]
                batch_question_states: List of (:Batch_Size) [Variable FloatTensor (1, Story_Hidden_Size)]
                max_n_hiddens: Scalar
            :return:
                batch_relation_map: [Variable FloatTensor (Batch_Size, Channel=4, Sent_Len, Story_Hidden_Size x Question_Hidden_Size)]
            """
            # print (batch_question_states)
            batch_relation_map = []
            for story_states, question_state in zip(batch_story_states, batch_question_states):
                states = [torch.cat((state, question_state.squeeze(0)), dim=0).unsqueeze(0) for state in story_states]  # Unsqueeze to create (1, Story_Hidden_Size x Question_Hidden_Size)
                first_layer = pad_to_max(torch.cat(states[::-1], dim=0), self.max_n_hiddens).unsqueeze(0)  # Unsqueeze to create (1, Sent_Len, Story_Hidden_Size x Question_Hidden_Size)
                second_layer = pad_to_max(torch.cat(states[1::2] + states[0::2], dim=0), self.max_n_hiddens).unsqueeze(0)
                third_layer = pad_to_max(torch.cat(states[0::2] + states[1::2], dim=0), self.max_n_hiddens).unsqueeze(0)
                fourth_layer = pad_to_max(torch.cat(states, dim=0), self.max_n_hiddens).unsqueeze(0)
                batch_relation_map += [torch.cat((first_layer, second_layer, third_layer, fourth_layer), dim=0).unsqueeze(0)]  # List of [Variable FloatTensor (Channel=4, Sent_Len, Story_Hidden_Size x Question_Hidden_Size)]

            batch_relation_map = torch.cat(batch_relation_map, dim=0)  # [Variable FloatTensor (Batch_Size, Channel=4, Max_Sent_Len, Story_Hidden_Size x Question_Hidden_Size)]
            return batch_relation_map

        # Get Hidden States
        batch_story_states = self.input_module(story_batch, story_batch_lengths, story_sent_batch_lengths)  # List of [Variable FloatTensor (Sent_Len, Hidden_Size * Num_Directions)]
        batch_question_state = self.question_module(question_batch, question_batch_lengths)  # List (:Batchsize) of [Variable FloatTensor (1, Hidden_Size * Num_Directions)]
        relation_map = create_relation_map(batch_story_states, batch_question_state, self.max_n_hiddens).transpose(1, 3).transpose(2, 3)  # [Variable FloatTensor (Batch_Size, Channel=4, Max_Sent_Len, Story_Hidden_Size x Question_Hidden_Size)]

        # Zero Padding
        # conv_s_out = F.relu(self.batchnorm_s(self.dropout_s(self.conv_s1(F.pad(relation_map, (0, 0, 0, 1))))))  # [Variable FloatTensor (Batch_Size, Num_Small_Filters,  Height, Max_Num_Hiddens)]
        # conv_s_out = F.relu(self.batchnorm_s(self.dropout_s(self.conv_s2(F.pad(conv_s_out, (0, 0, 0, 1))))))  # Padding will maintain the dimension.
        # conv_s_out = F.relu(self.batchnorm_s(self.dropout_s(self.conv_s3(F.pad(conv_s_out, (0, 0, 0, 1))))))
        # conv_s_out = F.relu(self.batchnorm_s(self.dropout_s(self.conv_s4(F.pad(conv_s_out, (0, 0, 0, 1))))))  # [Variable FloatTensor (Batch_Size, Num_Small_Filters,  Height, Max_Num_Hiddens)]
        # conv_m_out = F.relu(self.batchnorm_m(self.dropout_m(self.conv_m1(F.pad(relation_map, (0, 0, 0, 2))))))  # [Variable FloatTensor (Batch_Size, Num_Medium_Filters,  Height, Max_Num_Hiddens)]
        # conv_m_out = F.relu(self.batchnorm_m(self.dropout_m(self.conv_m2(F.pad(conv_m_out, (0, 0, 0, 2))))))  # Padding will maintain the dimension
        # conv_m_out = F.relu(self.batchnorm_s(self.dropout_m(self.conv_m3(F.pad(conv_m_out, (0, 0, 0, 2))))))
        # conv_m_out = F.relu(self.batchnorm_s(self.dropout_m(self.conv_m4(F.pad(conv_m_out, (0, 0, 0, 2))))))  # [Variable FloatTensor (Batch_Size, Num_Medium_Filters, Height, Max_Num_Hiddens)]
        # conv_l_out = F.relu(self.batchnorm_l(self.dropout_l(self.conv_l1(F.pad(relation_map, (0, 1, 0, 1))))))
        # conv_l_out = F.relu(self.batchnorm_l(self.dropout_l(self.conv_l2(F.pad(conv_l_out, (0, 1, 0, 1))))))
        # conv_l_out = F.relu(self.batchnorm_s(self.dropout_l(self.conv_l3(F.pad(conv_l_out, (0, 1, 0, 1))))))
        # conv_l_out = F.relu(self.batchnorm_s(self.dropout_l(self.conv_l4(F.pad(conv_l_out, (0, 1, 0, 1))))))  # [Variable FloatTensor (Batch_Size, Num_Large_Filters,  Height, Max_Num_Hiddens)]

        conv_s_out = self.conv_s1(relation_map)  # [Variable FloatTensor (Batch_Size, Num_Small_Filters, Height, Max_Num_Hiddens)]
        conv_s_out = self.conv_s2(conv_s_out)  # Padding will maintain the dimension.
        conv_s_out = self.conv_s3(conv_s_out)
        conv_s_out = self.conv_s4(conv_s_out)  # [Variable FloatTensor (Batch_Size, Num_Small_Filters, Height, Max_Num_Hiddens)]

        conv_m_out = self.conv_m1(relation_map)  # [Variable FloatTensor (Batch_Size, Num_Medium_Filters, Height, Max_Num_Hiddens)]
        conv_m_out = self.conv_m2(conv_m_out)  # Padding will maintain the dimension
        conv_m_out = self.conv_m3(conv_m_out)
        conv_m_out = self.conv_m4(conv_m_out)  # [Variable FloatTensor (Batch_Size, Num_Medium_Filters, Height, Max_Num_Hiddens)]

        conv_l_out = self.conv_l1(relation_map)
        conv_l_out = self.conv_l2(conv_l_out)
        conv_l_out = self.conv_l3(conv_l_out)
        conv_l_out = self.conv_l4(conv_l_out)  # [Variable FloatTensor (Batch_Size, Num_Large_Filters, Height, Max_Num_Hiddens)]


        ''' Inverted Position-wise Convolution (Doesn't make sense, but works well in some tasks) '''
        # Transpose the position of Channel and Width. (i.e. (Batch, 256, 3, 25) --> (Batch, 25, 3, 256)).
        # This is to perform the weighted sum (position_wise conv) on filters.
        # The output of the weighted sum should be, for example, (Batch, 1, 3, 256).
        position_conv_s_out = F.relu(self.position_conv_s(conv_s_out.transpose(1, 3)).view(conv_s_out.size(0), -1))  # [Variable FloatTensor (Batch_Size, Height_s x Max_Num_Hiddens)]
        position_conv_m_out = F.relu(self.position_conv_m(conv_m_out.transpose(1, 3)).view(conv_m_out.size(0), -1))  # [Variable FloatTensor (Batch_Size, Height_m x Max_Num_Hiddens)]
        position_conv_l_out = F.relu(self.position_conv_l(conv_l_out.transpose(1, 3)).view(conv_l_out.size(0), -1))  # [Variable FloatTensor (Batch_Size, Height_l x Max_Num_Hiddens)]

        # Concatenate the maxpool out
        various_relation_vector = torch.cat((position_conv_s_out, position_conv_m_out, position_conv_l_out), dim=1)  # [Variable FloatTensor (Batch_Size, (H_s + H_m + H_l) x Max_Num_Hiddens = 2048)]

        # Get various_relation_vector
        output = self.linear(various_relation_vector.view(various_relation_vector.size(0), -1))  # [Variable FloatTensor (Batch_Size, Output_Vocab_Size)]

        ''' Depth-wise Convolution '''
        # conv_s_out = self.conv_s(conv_s_out).squeeze(3).squeeze(2)
        # conv_m_out = self.conv_m(conv_m_out).squeeze(3).squeeze(2)
        # conv_l_out = self.conv_l(conv_l_out).squeeze(3).squeeze(2)
        #
        # # Few fc layer for each
        # linear_s_out = self.linear_s(conv_s_out)
        # linear_m_out = self.linear_m(conv_m_out)
        # linear_l_out = self.linear_l(conv_l_out)
        #
        # various_relation_vector = torch.cat((linear_s_out, linear_m_out, linear_l_out), dim=1)
        #
        # # Get various_relation_vector
        # output = self.linear(various_relation_vector.view(various_relation_vector.size(0), -1))  # [Variable FloatTensor (Batch_Size, Output_Vocab_Size)]


        ''' Depth-wise Summation '''
        # # Perform depth-wise summation
        # s_channel_sum = torch.sum(torch.sum(conv_s_out, dim=2), dim=3).squeeze(3).squeeze(2)
        # m_channel_sum = torch.sum(torch.sum(conv_m_out, dim=2), dim=3).squeeze(3).squeeze(2)
        # l_channel_sum = torch.sum(torch.sum(conv_l_out, dim=2), dim=3).squeeze(3).squeeze(2)
        #
        # # Few fc layer for each
        # linear_s_out = self.linear_s(s_channel_sum)
        # linear_m_out = self.linear_m(m_channel_sum)
        # linear_l_out = self.linear_l(l_channel_sum)
        #
        # # Concatenate the maxpool out
        # various_relation_vector = torch.cat((linear_s_out, linear_m_out, linear_l_out), dim=1)  # [Variable FloatTensor (Batch_Size, 3 * 256 = 768)]
        #
        # # Get various_relation_vector
        # output = self.linear(various_relation_vector.view(various_relation_vector.size(0), -1))  # [Variable FloatTensor (Batch_Size, Output_Vocab_Size)]


        return output