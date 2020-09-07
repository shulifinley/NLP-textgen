import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
from Data_loader import Data_for_train
from Data_loader import freinds_parsing
import random
import sklearn
from losses import KLD , similarity_loss
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.tensorboard import SummaryWriter

class AttnDecoderGen_lstm(nn.Module):
    def __init__(self, hidden_size, embedding_size, input_size,output_size, n_layers=4,embbed_pretrained_weight=None,seq_len=19,bidirectional=False,encoder_bidirictional=True):
        super(AttnDecoderGen_lstm, self).__init__()

        # Keep parameters for reference
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.input_size = input_size
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(input_size, embedding_size)
        # using matched pretrained embeddings:
        if not  embbed_pretrained_weight is None: # Using zero initialization for nonvocabulary words:
            self.embedding.weight.data.copy_(torch.from_numpy(embbed_pretrained_weight))
        self.embedding.weight.requires_grad = True
        self.drop_out_embedding = nn.Dropout(0.3)

        self.drop_out = nn.Dropout(0.4)

        # Define layers
        self.encoder_bidirictional = encoder_bidirictional
        self.lstm2 = nn.LSTM(self.embedding_size + self.hidden_size, hidden_size, n_layers,batch_first=True,dropout = 0.5,bidirectional=self.bidirectional)
        self.out = nn.Linear(hidden_size, output_size)

#
        self.attn = Attn(hidden_size)

    def forward(self, word_input, last_hidden, encoder_outputs):
        # Note: we run this one step at a time

        # Get the embedding of the current input word (last output word)

        # Calculate attention from current RNN state and all encoder outputs; apply to encoder outputs
        attn_weights = self.attn(last_hidden[0][0], encoder_outputs)
        context = attn_weights.unsqueeze(1).bmm(encoder_outputs)  # B x 1 x N

        word_embedded = self.embedding(word_input.squeeze()) # S=1 x B x N

        word_embedded= self.drop_out_embedding(word_embedded)
        # Combine embedded input word and last context, run through RNN
        context= context.squeeze()
        word_embedded=word_embedded.squeeze()
        if len(np.shape(word_embedded))<2 :
            word_embedded = word_embedded.unsqueeze(0)
            context=context.unsqueeze(0)
        rnn_input = torch.cat((word_embedded, context), 1)
        rnn_output, hidden = self.lstm2(rnn_input.unsqueeze(1), last_hidden)


        # Final output layer (next word prediction) using the RNN hidden state and context vector

        rnn_output = rnn_output.squeeze(0)  # S=1 x B x N -> B x N
        context = context.squeeze(1)  # B x S=1 x N -> B x N
        output = self.drop_out(self.out(rnn_output))

        # Return final output, hidden state
        return output, hidden
    def init_hidden(self,batch_size):
        hidden = (torch.zeros(self.n_layers, batch_size, self.hidden_size).cuda(),
                  torch.zeros(self.n_layers, batch_size, self.hidden_size).cuda())
        return hidden

class Eecoder_Gen_lstm(nn.Module):
    def __init__(self, hidden_size, embedding_size, input_size,output_size, n_layers=4,embbed_pretrained_weight=None,seq_len=19,bidirectional=True,embedding=None,sampling= False):
        super(Eecoder_Gen_lstm, self).__init__()

        # Keep parameters for reference
        self.sampling =sampling
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.const_embed = []
        self.bidirectional = bidirectional

        #self.batch_size= batch_size
        # Define layers
        #self.embedding = nn.Embedding(input_size, embedding_size)
        self.embedding = embedding
        # using matched pretrained embeddings:
        #if not  embbed_pretrained_weight is None: # Using zero initialization for nonvocabulary words:
        #    ind_2_train = np.where(np.sum(embbed_pretrained_weight,axis=1) >0)[0]
#
        #    self.embedding.weight.data.copy_(torch.from_numpy(embbed_pretrained_weight))
        #self.embedding.weight.requires_grad = True

        self.lstm = nn.LSTM(embedding_size, hidden_size, n_layers,dropout = 0.3,batch_first=True,bidirectional=self.bidirectional)
        self.mu_est = nn.Parameter(torch.FloatTensor(self.hidden_size*2,self.hidden_size*2).cuda())
        self.sigma_est = nn.Parameter(torch.FloatTensor(self.hidden_size*2,self.hidden_size*2).cuda())

        #self.sigma_est = nn.Linear(hidden_size, hidden_size)


    def forward(self, word_input, last_hidden):
        # Note: we run this one step at a time (word by word...)

        # Get the embedding of the current input word (last output word)
        word_embedded = self.embedding(word_input)# S=1 x B x N


        # run through LSTM

        rnn_output, hidden = self.lstm(word_embedded, last_hidden)


        if self.sampling:

            mu=torch.matmul(rnn_output,self.mu_est)
            logvar=torch.matmul(rnn_output,self.sigma_est)

            logvar= torch.clamp(logvar,-10,10)
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            mu= torch.clamp(mu,-100,100)

            if len(np.shape(mu.tolist())) <3:
                rand_tensor = torch.FloatTensor(
                    np.random.rand(np.shape(mu.tolist())[0], np.shape(mu.tolist())[1])).cuda()
            else:
                rand_tensor = torch.FloatTensor(np.random.rand(np.shape(mu.tolist())[0],np.shape(mu.tolist())[1] ,np.shape(mu.tolist())[2]  )).cuda()

            sampled_tensor =  mu + eps*std
            rnn_output = sampled_tensor


        #output = F.softmax(self.out(rnn_output_last), 1)
        # Return final output and hidden state
        return rnn_output, hidden ,mu,logvar


    def init_hidden(self,batch_size):
        hidden = (torch.zeros((1+self.bidirectional)*self.n_layers, batch_size, self.hidden_size).cuda(),
                  torch.zeros( (1+self.bidirectional)*self.n_layers, batch_size,self.hidden_size).cuda())
        return hidden

class Attn(nn.Module):
    def __init__(self, hidden_size):
        super(Attn, self).__init__()

        self.hidden_size = hidden_size
        self.v = nn.Parameter(torch.FloatTensor(self.hidden_size).cuda())
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)

    def forward(self, hidden, encoder_outputs):
        seq_len = np.shape(encoder_outputs)[1]

        attn_energies = torch.zeros(seq_len).cuda()
        attn_energies = []
        for i in range(seq_len):
            #attn_energies[i] = self.score(hidden.squeeze(), encoder_outputs[:,i,:])
            attn_energies.append(self.score(hidden.squeeze(), encoder_outputs[:,i,:]))
        concat_atten =[]
        for i in range(len(attn_energies)):
            if len(concat_atten) ==0 :
                concat_atten = attn_energies[i]
            else:
                concat_atten=torch.cat(  (concat_atten,attn_energies[i]),0  )
        # Normalize energies to weights in range 0 to 1, resize to 1 x 1 x seq_len
        return torch.transpose( F.softmax(concat_atten, 0),0,1)

    def score(self, hidden, encoder_output):
        '''Aditive Attention'''
        if len(np.shape(hidden)) < 2 :
            hidden = hidden.unsqueeze(0)
        attn_input = torch.cat((hidden, encoder_output), 1)
        energy = self.attn(attn_input)
        energy = torch.mm(self.v.unsqueeze(0),(torch.transpose(energy,0,1)))
        return energy

def prepare_sequence_data(charecters=None,sequence_length=15,dict_data=None):
        _train = {}
        i = 0
        trainning = []
        label_data = []
        for ind, name in enumerate(charecters):
            dict = dict_data[name]
            # (dict)
            for vals in dict.values():
                val_temp = np.zeros(sequence_length+1, dtype=np.int)
                val_temp[0] =label_charecter[ind]
                if len(vals) >= sequence_length:
                    for ii in range(1,np.min([20,len(vals) - sequence_length]),5):
                        label_zero = np.zeros((len(charecters)), dtype=np.int)
                        _train[i] = {}
                        label_zero[ind] = 1
                        _train[i]['label'] = label_zero.copy()
                        _train[i]['values'] = vals[ii:sequence_length+ii]
                        val_temp[1:sequence_length+1] = vals[ii:sequence_length+ii]
                        if len(trainning) == 0:
                            trainning = np.expand_dims(val_temp.copy(), axis=0)
                            label_data = np.expand_dims(label_zero, axis=0)
                        else:
                            trainning = np.concatenate([trainning,np.expand_dims(val_temp.copy(),axis=0)],axis=0)
                            label_data = np.concatenate([label_data, np.expand_dims(label_zero, axis=0)], axis=0)
                        i = i + 1
        return trainning , label_data

def url_chooser(name='freinds',debug_mode=False):
    if name =='freinds':
        url_list = freinds_parsing()
    if name == 'lion_king':
        url_list = [r'https://transcripts.fandom.com/wiki/The_Lion_King',
                              r'https://transcripts.fandom.com/wiki/The_Lion_King_II:_Simba%27s_Pride',
                              r'https://transcripts.fandom.com/wiki/Never_Everglades',
                              r'http://www.lionking.org/scripts/TLK1.5-Script.html#specialpower']
    if name == 'games_of_thorne':
        url_list = [r'https://genius.com/Game-of-thrones-the-pointy-end-annotated',
                                   r'https://genius.com/Game-of-thrones-the-pointy-end-annotated',
                                   r'https://genius.com/Game-of-thrones-the-kingsroad-annotated', \
                                   r'https://genius.com/Game-of-thrones-cripples-bastards-and-broken-things-annotated',
                                   r'https://genius.com/Game-of-thrones-the-wolf-and-the-lion-annotated', \
                                   r'https://genius.com/Game-of-thrones-a-golden-crown-annotated',
                                   r'https://genius.com/Game-of-thrones-you-win-or-you-die-annotated',
                                   r'https://genius.com/Game-of-thrones-fire-and-blood-annotated',
                                   r'https://genius.com/Game-of-thrones-the-north-remembers-annotated', \
                                   r'https://genius.com/Game-of-thrones-the-night-lands-annotated',
                                   r'https://genius.com/Game-of-thrones-what-is-dead-may-never-die-annotated',
                                   r'https://genius.com/Game-of-thrones-garden-of-bones-annotated', \
                                   r'https://genius.com/Game-of-thrones-the-ghost-of-harrenhal-annotated',
                                   r'https://genius.com/Game-of-thrones-the-old-gods-and-the-new-annotated',
                                   r'https://genius.com/Game-of-thrones-a-man-without-honor-annotated', \
                                   r'https://genius.com/Game-of-thrones-the-prince-of-winterfell-annotated',
                                   r'https://genius.com/Game-of-thrones-blackwater-annotated',
                                   r'https://genius.com/Game-of-thrones-valar-morghulis-annotated', \
                                   r'https://genius.com/Game-of-thrones-valar-dohaeris-annotated']
    if name =='Harry_potter':
        url_list =[r'https://www.hogwartsishere.com/library/book/12647/chapter/1/',r'https://www.hogwartsishere.com/library/book/12647/chapter/2/'\
                   r'https://www.hogwartsishere.com/library/book/12647/chapter/3/']
    if debug_mode:  # subset of data just to debug code
        url_list = url_list[0:2 ]
    return  url_list

def accuracy_test(test,test_label,length_input=None):
    # Only for
    out_vector =[]
    target_vec = []
    for i in range(int( len(test_label)/batch_size)):
        sequence_tensor = torch.LongTensor(test[i*(batch_size):i*(batch_size)+batch_size, : ]  ).cuda()
        hidden= encoder.init_hidden(batch_size=batch_size)

        encoder_output, hidden_encoder = encoder.forward(sequence_tensor, last_hidden=hidden)
        decoder_hidden =  decoder_test.init_hidden(batch_size=batch_size)
        decoder_context = torch.zeros(batch_size, decoder_test.hidden_size).cuda()

        for ii in range(sequence_L):
            sequence_tensor = torch.LongTensor(
                test[i*(batch_size):i*(batch_size)+batch_size , ii] ).cuda()
            output, decoder_hidden = decoder_test(sequence_tensor.view(batch_size, 1),
                                                                   decoder_hidden, encoder_output)
        #hidden = decoder_test.init_hidden(batch_size=1)

        outputing =(output.view(-1,output_size).cpu().detach().numpy())
        out_vector.append(np.argmax(outputing,axis=1))
        target_vec.append(np.argmax(test_label[i*batch_size:i*batch_size+batch_size, :],axis=1))

    [target_vec[i].tolist() for i in range(len(target_vec))]
    target_tot =[]
    pred_tot = []
    for i in range(len(out_vector)):
        if len(target_tot) == 0:
            target_tot = target_vec[i]
            pred_tot=out_vector[i]
        else:
            target_tot = np.concatenate([target_tot,target_vec[i]])
            pred_tot = np.concatenate([pred_tot,out_vector[i]])
    val= 0
    for i in range(len(pred_tot)):
        val+= np.sum(pred_tot[i] == target_tot[i])
    #print(val/len(pred_tot))
    Cm = sklearn.metrics.confusion_matrix(pred_tot, target_tot)
    sum = np.sum(Cm, axis=1)
    print(Cm / np.repeat(np.reshape(sum, (output_size, 1)), output_size, 1))
    return val/len(pred_tot)

def generation_func(data_train,decoder,encoder, charecter = 'joey',gen_length = 11,sentence=None):
    decoder_context = torch.zeros(1, encoder.hidden_size * (1 + encoder.bidirectional)).cuda()

    encoded_list = []
    output_text= []
    charecter_ind = data_train.word2index[charecter + '1']
    batch_size=1
    hidden = encoder.init_hidden(batch_size=batch_size)

    decoder_hidden = decoder.init_hidden(batch_size=batch_size)


    encoded_list.append(charecter_ind)
    for i in range(len(sentence)):
        encoded_list.append(data_train.word2index[sentence[i]])
    encoder_input = torch.LongTensor(encoded_list).cuda()

    for k in range(len(sentence)+1 ):
        decoder_input = encoder_input[k]
        encoder_output, hidden_encoder = encoder.forward(encoder_input.view(batch_size, len(encoded_list)),
                                                         last_hidden=hidden)

        output, decoder_hidden = decoder(decoder_input.view(batch_size, 1)
                                                          , decoder_hidden, encoder_output)
    topv, topi = output.topk(k=2)
    # probs = torch.softmax(topv,2)


    choices = topi.tolist()
    ni = np.random.choice(choices[0])
    decoder_input = torch.LongTensor([ni]).cuda()
    encoded_list.append(ni)
    encoder_input = torch.LongTensor(encoded_list).cuda()
    output_text.append(data_train.index2word[ni])

    for i in range(gen_length):
        encoder_output, hidden_encoder = encoder.forward(encoder_input.view(batch_size, len(encoded_list)), last_hidden=hidden)

        output, decoder_hidden = decoder(decoder_input.view(batch_size, 1),
                                                                decoder_hidden, encoder_output)


        topv, topi = output.topk(k=2)
        #probs = torch.softmax(topv,2)

        ni = topi[0][0].item()
        choices = topi.tolist()
        ni = np.random.choice(choices[0])
        decoder_input = torch.LongTensor([ni]).cuda()
        encoded_list.append(ni)
        encoder_input = torch.LongTensor(encoded_list).cuda()
        output_text.append(data_train.index2word[ni] )

    return output_text

    # target_tensor2 = torch.zeros(batch_size,data_train.n_words).cuda()
    # for lk in range(batch_size):
    #    target_tensor2[i,target_tensor[lk,ii]] =1
    # sentence_loss+= torch.sum(torch.abs(output.squeeze()-target_tensor2))
    output_softmax = torch.log_softmax((output.view(-1, batch_size, data_train.n_words)), 2).squeeze()

def generation_func_combination(data_train,decoder,encoder, charecters = ['joey','rachel'],gen_length = 11,sentence=None):
    encoded_list1 = []
    encoded_list2 = []

    output_text= []
    charecter_ind1 = data_train.word2index[charecters[0] + '1']
    charecter_ind2 = data_train.word2index[charecters[1] + '1']

    batch_size=1
    hidden = encoder.init_hidden(batch_size=batch_size)

    decoder_hidden = decoder.init_hidden(batch_size=batch_size)


    encoded_list1.append(charecter_ind1)
    encoded_list2.append(charecter_ind2)

    for i in range(len(sentence)):
        encoded_list1.append(data_train.word2index[sentence[i]])
        encoded_list2.append(data_train.word2index[sentence[i]])

    encoder_input1 = torch.LongTensor(encoded_list1).cuda()
    encoder_input2 = torch.LongTensor(encoded_list2).cuda()

    for k in range(len(sentence)):
        encoder_output1, hidden_encoder1 ,mu,logvar= encoder.forward(encoder_input1.view(batch_size, len(encoded_list1)),
                                                         last_hidden=hidden)
        encoder_output2, hidden_encoder2,mu,logvar = encoder.forward(encoder_input2.view(batch_size, len(encoded_list2)),
                                                         last_hidden=hidden)
        decoder_input = encoder_input1[k+1] # its the same start word

        encoder_output=(encoder_output1[0:] +encoder_output2[0:])/2   # in order to skip the identity of  a speaker in the decoder.

        output, decoder_hidden = decoder(decoder_input.view(batch_size, 1),
                                                          decoder_hidden, encoder_output)
    topv, topi = output.topk(k=4)
    # probs = torch.softmax(topv,2)


    choices = topi.tolist()
    ni = np.random.choice(choices[0])
    decoder_input = torch.LongTensor([ni]).cuda()
    encoded_list1.append(ni)
    encoded_list2.append(ni)

    encoder_input1 = torch.LongTensor(encoded_list1).cuda()
    encoder_input2 = torch.LongTensor(encoded_list2).cuda()

    output_text.append(data_train.index2word[ni])

    for i in range(gen_length):
        encoder_output1, hidden_encoder1,mu,logvar = encoder.forward(encoder_input1.view(batch_size, len(encoded_list1)), last_hidden=hidden)
        encoder_output2, hidden_encoder2,mu,logvar = encoder.forward(encoder_input2.view(batch_size, len(encoded_list2)), last_hidden=hidden)

        encoder_output=(encoder_output1[0:] +encoder_output2[0:])/2   # in order to skip the identity of  a speaker in the decoder.

        output, decoder_hidden = decoder(decoder_input.view(batch_size, 1)
                                                               , decoder_hidden, encoder_output)


        topv, topi = output.topk(k=4)
        #probs = torch.softmax(topv,2)

        ni = topi[0][0].item()
        choices = topi.tolist()
        flage=1
        while (flage):
            ni = np.random.choice(choices[0])
            decoder_input = torch.LongTensor([ni]).cuda()
            encoded_list1.append(ni)
            encoded_list2.append(ni)

            encoder_input1 = torch.LongTensor(encoded_list1).cuda()
            encoder_input2 = torch.LongTensor(encoded_list2).cuda()
            word_add = data_train.index2word[ni]
            if word_add !=output_text[-1] :
                flage= 0
                output_text.append(data_train.index2word[ni] )

    return output_text

    # target_tensor2 = torch.zeros(batch_size,data_train.n_words).cuda()
    # for lk in range(batch_size):
    #    target_tensor2[i,target_tensor[lk,ii]] =1
    # sentence_loss+= torch.sum(torch.abs(output.squeeze()-target_tensor2))
    output_softmax = torch.log_softmax((output.view(-1, batch_size, data_train.n_words)), 2).squeeze()


if __name__ == '__main__':
    ### Data creation:
    url_list = url_chooser('freinds',debug_mode=False)
    charecters = ['rachel','monica','joey','chandler','pheuby','ross']

    ## preparation of the data might take time. so it saves it. ( if it the first time running so load_data_set = False)
    load_data_set = True
    if load_data_set == False:
        data_train= Data_for_train(url_list=url_list,charecters =charecters,contain_non_embbedding = True,embedding_size=50,embedding_path='./EMbeddings/glove_vectors_50d.npy',word2idx_path='/home/yuval/PycharmProjects/NLP_FINAL_PROJ/EMbeddings/wordsidx.txt' )
        data_train.create_trainning_Data()
        # adding an embedding space which specify which charecter is speaking:
        np.save('trainning_data.npy', data_train)
    else:
        data_train = np.load('trainning_data.npy', allow_pickle='TRUE').item()

    dict_data = data_train.dict

    label_charecter = []
    # Adding to the vocabulary starting sentence charecter specific token:
    for char in charecters:
        data_train.index_word(char+'1')
        label_charecter.append( data_train.word2index[char+'1'] )
    dict_data = data_train.dict

    sequence_L =15
    trainning , label_data= prepare_sequence_data(charecters=charecters,sequence_length=sequence_L,dict_data=dict_data)

    # Model parameters:
    embedding_size = 50
    output_size=len(charecters)
    hidden_size = 10
    n_layers_encoder = 2
    n_layers_decoder = 2

    decoder_test = AttnDecoderGen_lstm(hidden_size=hidden_size*(1+1), embedding_size=embedding_size,\
    output_size=data_train.n_words, input_size=data_train.n_words,embbed_pretrained_weight=data_train.embedding,\
    n_layers=n_layers_decoder,bidirectional=False,encoder_bidirictional=True  ).cuda()

    encoder  =  Eecoder_Gen_lstm(hidden_size=hidden_size, embedding_size=embedding_size, \
    output_size=data_train.n_words, input_size=data_train.n_words,embbed_pretrained_weight=data_train.embedding,\
                                 n_layers=n_layers_encoder,bidirectional=True,embedding=decoder_test.embedding,sampling=True).cuda()


    optimizer_decoder = optim.Adam(decoder_test.parameters(), lr=0.001)
    optimizer_encoder =  optim.Adam(encoder.parameters(), lr=0.001)
    load = True
    sample_model=True
    if load == True:
        if sample_model: # 2 modes of models (with sampling process or not )

            decoder = torch.load(r'/home/yuval/PycharmProjects/NLP_FINAL_PROJ/model_decoder_trained_newst200')
            encoder_load = torch.load(r'/home/yuval/PycharmProjects/NLP_FINAL_PROJ/model_encoder_trained_newst200')
            decoder_test.load_state_dict(decoder['model_state_dict'])
            encoder.load_state_dict(encoder_load['model_state_dict'])
            optimizer_decoder.load_state_dict(decoder['optimizer_state_dict'])
            optimizer_encoder.load_state_dict(encoder_load['optimizer_state_dict'])
        else:
            decoder = torch.load(r'/home/yuval/PycharmProjects/NLP_FINAL_PROJ/model_decoder_trained_newst')
            encoder_load = torch.load(r'/home/yuval/PycharmProjects/NLP_FINAL_PROJ/model_eecoder_trained_newst')
            decoder_test.load_state_dict(decoder['model_state_dict'])
            encoder.load_state_dict(encoder_load['model_state_dict'])
            optimizer_decoder.load_state_dict(decoder['optimizer_state_dict'])
            optimizer_encoder.load_state_dict(encoder_load['optimizer_state_dict'])


    ##  splliting data for validation and trainning:
    rand_keys = np.random.permutation(len(trainning)-1)
    train_num = int( 0.95*len(trainning) )

    trainning_train = ( trainning[rand_keys[0:train_num],:])
    label_train = label_data[rand_keys[0:train_num], :]

    validation = ( trainning[rand_keys[train_num:],:])
    label_validation = label_data[rand_keys[train_num:], :]

    batch_size=1000
    batch_size_val = 50
    n_epochs = 400
    factor_num=2

    hidden= encoder.init_hidden(batch_size=batch_size)
    # Loss definning:
    loss = nn.NLLLoss()

    writer = SummaryWriter('/home/yuval/PycharmProjects/NLP_FINAL_PROJ/tf1')

    scheduler_decoder  = CyclicLR(optimizer_decoder, base_lr=5e-4, max_lr=5e-2, step_size_up=1000,base_momentum=0.99,cycle_momentum=False)
    scheduler_encoder  = CyclicLR(optimizer_encoder, base_lr=5e-4, max_lr=5e-2, step_size_up=1000,base_momentum=0.99,cycle_momentum=False)

    torch.cuda.current_device()
    torch.cuda.device_count()
    no_use_teacher_forcing_ratio=0.25

    factor_cross = 1
    weight = torch.FloatTensor([1,0.1,0.1e-6]).cuda()

    # Trainning loop :
    for e in range(1, n_epochs + 1):
        tot_loss = 0
        count = 0
        for i in range(np.int(train_num/batch_size-1)):
            count= count+1
            decoder_hidden = decoder_test.init_hidden(batch_size=batch_size)

            no_use_teacher_forcing = random.random() < no_use_teacher_forcing_ratio
            sentence_loss = 0

            if no_use_teacher_forcing:
                encoded_list=torch.LongTensor(np.expand_dims ( trainning_train[i * batch_size: i * batch_size + batch_size ,  0] ,axis=1)).cuda()
                target_tensor = torch.LongTensor(
                    trainning_train[i * batch_size: i * batch_size + batch_size, :]).cuda()

                for ii in range(sequence_L):
                    encoder_input = encoded_list
                    decoder_input = torch.LongTensor(
                        trainning_train[i * batch_size: i * batch_size + batch_size, ii]).cuda()

                    # Due for generation purposes and not seq2seq -> run the encoder again .  every time with the new sequence :
                    encoder_output, hidden_encoder,mu,logvar = encoder.forward(encoder_input.view(batch_size, len(encoded_list[0])),
                                                                     last_hidden=hidden)
                    output, decoder_hidden = decoder_test(decoder_input.view(batch_size, 1), decoder_hidden, encoder_output)
                    output_softmax = torch.log_softmax((output.view(-1, batch_size, data_train.n_words)), 2).squeeze()
                    targets = target_tensor[:,ii]

                    argmax   = torch.argmax(output_softmax,1)
                    decoder_input = argmax
                    encoded_list= torch.cat([encoded_list,argmax.unsqueeze(1)],1)
                    sentence_loss +=  loss(output_softmax,targets)*weight[0]  + similarity_loss(decoder_input, targets,decoder_test)*weight[1] + KLD(mu,logvar)*weight[2]

            else:

                for ii in range(sequence_L):
                    sequence_tensor = torch.LongTensor(
                        trainning_train[i * batch_size: i * batch_size + batch_size, 0:ii+1]).cuda()
                    target_tensor = torch.LongTensor(
                        trainning_train[i * batch_size: i * batch_size + batch_size, :]).cuda()

                    encoder_output, hidden_encoder, mu, logvar = encoder.forward(sequence_tensor, last_hidden=hidden)
                    decoder_hidden = decoder_test.init_hidden(batch_size=batch_size)
                    decoder_context = torch.zeros(batch_size, encoder.hidden_size * (1 + encoder.bidirectional)).cuda()

                    decoder_input2 = torch.LongTensor(
                        trainning_train[i * batch_size: i * batch_size + batch_size, ii]).cuda()
                    output,decoder_hidden = decoder_test(decoder_input2.view(batch_size, 1) , decoder_hidden, encoder_output)
                    output_softmax = torch.log_softmax((output.view(-1, batch_size, data_train.n_words)), 2).squeeze()
                    argmax   = torch.argmax(output_softmax,1)

                    targets = target_tensor[:,ii+1]

                    sentence_loss +=  loss(output_softmax,targets)*weight[0] + similarity_loss(argmax, targets,decoder_test)*weight[1] + KLD(mu,logvar)*weight[2]


            # updating weights
            # averaging total loss
            sentence_loss.backward()

            _ = torch.nn.utils.clip_grad_norm_(
                decoder_test.parameters(), 5)
            _ = torch.nn.utils.clip_grad_norm_(
                encoder.parameters(), 5)
            optimizer_decoder.step()
            optimizer_encoder.step()
            scheduler_decoder.step()
            scheduler_encoder.step()
            hidden[0].detach()
            hidden[1].detach()
            #sentence_loss.detach()
            optimizer_decoder.zero_grad()
            optimizer_encoder.zero_grad()
            tot_loss += sentence_loss.cpu().detach().numpy()/np.int(train_num/batch_size-1)

            hidden = encoder.init_hidden(batch_size=batch_size)
            #hidden_encoder  = encoder.init_hidden(batch_size=batch_size)
        tot_loss_val = 0
        loss_val=0
        if False:
            for ll in range(np.int(len(validation)/batch_size_val-1)):
                hidden2 = encoder.init_hidden(batch_size=batch_size_val)
    #
                target_tensor2 = torch.LongTensor(trainning_train[i*batch_size_val: i*batch_size_val+ batch_size_val,1:sequence_L+1]).cuda()
                sequence_tensor2 = torch.LongTensor(validation[ll * batch_size_val: ll * batch_size_val + batch_size_val, 0:sequence_L]).cuda()
    ##
                # output, hidden = decoder_test(word_input=sequence_tensor, last_context=hidden, last_hidden=hidden)
    ##
                encoder_output2, hidden_encoder2 = encoder.forward(sequence_tensor2, last_hidden=hidden2)
                decoder_hidden2 = decoder_test.init_hidden(batch_size=batch_size_val)
    #
                decoder_context2 = torch.zeros(batch_size_val, decoder_test.hidden_size).cuda()
                for ii in range(0):
                    sequence_tensor2 = torch.LongTensor(
                        validation[ll * batch_size_val: ll * batch_size_val + batch_size_val, ii]).cuda()
                    output2, decoder_context2, decoder_hidden2 = decoder_test(sequence_tensor2.view(batch_size_val, 1),
                                                                           decoder_context2, decoder_hidden2, encoder_output2)
                    #sentence_loss+= torch.sum(torch.abs(output.squeeze()-target_tensor2))
                    output_softmax2 = torch.softmax((output2.view(-1, batch_size_val, data_train.n_words)), 2).squeeze()

                    output_softmax2 = torch.log_softmax((output2.view(-1, batch_size_val, data_train.n_words)), 2).squeeze()
                    loss_val += loss(output_softmax2,target_tensor2[:,ii])

                tot_loss_val += loss_val
            print('mean loss val')
        #print(tot_loss_val.cpu().detach().numpy())
        #writer.add_scalar('Validation',global_step=  e,scalar_value=tot_loss_val.cpu().detach().numpy())

        print('epoch:  ' + str(e))
        print('loss train')
        print(tot_loss)
        writer.add_scalar('trainning',global_step=  e,scalar_value= tot_loss)

        writer.add_scalar('lr',global_step=  e,scalar_value= scheduler_decoder.get_last_lr()[0])
        if np.mod(e,100)==0:
            torch.save({
                'epoch': e,
                'model_state_dict': decoder_test.state_dict(),
                'optimizer_state_dict': optimizer_decoder.state_dict(),
            }, r'/home/yuval/PycharmProjects/NLP_FINAL_PROJ/model_decoder_trained_newst'+str(e))
            # encoder model
            torch.save({
                'epoch': e,
                'model_state_dict': encoder.state_dict(),
                'optimizer_state_dict': optimizer_encoder.state_dict(),
            }, r'/home/yuval/PycharmProjects/NLP_FINAL_PROJ/model_encoder_trained_newst'+str(e))

        #writer.add_scalar('accuracy validation',global_step=  e,scalar_value= acuracy_val)

        #plt.figure(1)
        #plt.scatter(e,tot_loss_val.cpu().detach().numpy()/np.int(len(validation)/batch_size-1),color='r')
        #plt.scatter(e,tot_loss,color='b')
        #plt.pause(0.2)

# decoder model
#writer.add_graph(model=decoder_test)


torch.save({
            'epoch': e,
            'model_state_dict': decoder_test.state_dict(),
            'optimizer_state_dict': optimizer_decoder.state_dict(),
            }, r'/home/yuval/PycharmProjects/NLP_FINAL_PROJ/model_decoder_trained_newst')
# encoder model
torch.save({
            'epoch': e,
            'model_state_dict': encoder.state_dict(),
            'optimizer_state_dict': optimizer_encoder.state_dict(),
            }, r'/home/yuval/PycharmProjects/NLP_FINAL_PROJ/model_encoder_trained_newst')

# Write graph for the decoder:
writer.add_graph(decoder_test,input_to_model=(decoder_input.view(batch_size, 1),
                                                       decoder_context, decoder_hidden, encoder_output))
# Write graph for the encoder:
writer.add_graph(encoder,input_to_model=(sequence_tensor, hidden))

start_gen = True
encoded_list = []
for ii in range(sequence_L):
    sequence_tensor = torch.LongTensor(
        trainning_train[i * batch_size: i * batch_size + batch_size, ii]).cuda()
    output, decoder_context, decoder_hidden = decoder_test(sequence_tensor.view(batch_size, 1),
                                                           decoder_context, decoder_hidden, encoder_output)

# Generating
for char in charecters:
    start_string=['mister','bing']
    output_text = generation_func(data_train,decoder=decoder_test,encoder=encoder,charecter= char ,gen_length=10,sentence= start_string)
    print(char+': ')
    print( start_string + output_text)

decoder = decoder_test
## Generation process example::
start_string=['armadillo']
out=''
out_Str = generation_func_combination(data_train,decoder=decoder_test,encoder=encoder, charecters = ['ross','rachel']\
                                      ,gen_length=20,sentence= start_string)

for i in range(len(start_string)):
    out+=' '+start_string[i]
for i in range(len(out_Str)):
    out+=' '+out_Str[i]
print(out)



output, decoder_context, decoder_hidden = decoder_test(decoder_input.view(batch_size, 1),
                                                       decoder_context, decoder_hidden, encoder_output)

