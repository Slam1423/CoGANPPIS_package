import os
import time
import pickle
import numpy as np
import torch
from torch.optim import lr_scheduler
from torch.nn.init import xavier_normal,xavier_normal_
from torch import nn
import torch.utils.data.sampler as sampler
from config import DefaultConfig
from ppi_model import PPIModel
import data_generator


configs = DefaultConfig()


def test(model, loader,path_dir,threshold,test_name):

    # Model on eval mode
    model.eval()
    length = len(loader)
    result = []
    all_trues = []


    for batch_idx, (seq_data, pssm_data, dssp_data, local_data, label, msa_file, middle_fea) in enumerate(loader):
    
        # Create vaiables
        with torch.no_grad():
            if torch.cuda.is_available():
                seq_var = torch.autograd.Variable(seq_data.cuda().float())
                pssm_var = torch.autograd.Variable(pssm_data.cuda().float())
                dssp_var = torch.autograd.Variable(dssp_data.cuda().float())
                local_var = torch.autograd.Variable(local_data.cuda().float())
                target_var = torch.autograd.Variable(label.cuda().float())
                msa_var = torch.autograd.Variable(msa_file.cuda().float())
                middle_var = torch.autograd.Variable(middle_fea.cuda().float())
            else:
                seq_var = torch.autograd.Variable(seq_data.float())
                pssm_var = torch.autograd.Variable(pssm_data.float())
                dssp_var = torch.autograd.Variable(dssp_data.float())
                local_var = torch.autograd.Variable(local_data.float())
                target_var = torch.autograd.Variable(label.float())
                msa_var = torch.autograd.Variable(msa_file.float())
                middle_var = torch.autograd.Variable(middle_fea.float())

        output = model(seq_var, dssp_var, pssm_var, local_var, msa_var, middle_var)
        shapes = output.data.shape
        output = output.view(shapes[0]*shapes[1])
        result.append(output.data.cpu().numpy())

    #caculate
    all_preds = np.concatenate(result, axis=0)
#    print(all_preds)
    all_preds2 = all_preds >= threshold
    all_preds2 = all_preds2.astype(int)
#    print(all_preds)
#    print(threshold)
    result_file = "predict_result_dir/" + test_name + "_predict_result.pkl"

    with open(result_file,"wb") as fp:
        pickle.dump(all_preds,fp)

    print('prediction done.')


def predict(model_file,test_data,window_size,path_dir,threshold=0.51,test_name=''):
    test_sequences_file = ['features/{0}_sequence_data.pkl'.format(key) for key in test_data]
    test_dssp_file = ['features/{0}_netsurf_ss_14_standard.pkl'.format(key) for key in test_data]
    test_pssm_file = ['features/{0}_pssm_data.pkl'.format(key) for key in test_data]
    test_label_file = ['features/{0}_label.pkl'.format(key) for key in test_data]
    test_MSA_file = ['features/{0}_MSA_features_1.pkl'.format(key) for key in test_data]
    test_list_file = ['features/{0}_predict_list.pkl'.format(key) for key in test_data]
    all_list_file = ['features/{0}_list.pkl'.format(key) for key in test_data]
    
    # parameters
    batch_size = configs.batch_size

    # Datasets
    test_dataSet = data_generator.dataSet(window_size, test_sequences_file, test_pssm_file, test_dssp_file, test_label_file,
                                             all_list_file, test_MSA_file)
    # Models

    test_list = []
    for test_file in test_list_file:
        with open(test_file, "rb") as test_label:
            print(test_file)
            temp_list = pickle.load(test_label)
        test_list.extend(temp_list)

    test_samples = sampler.SequentialSampler(test_list)
    test_loader = torch.utils.data.DataLoader(test_dataSet, batch_size=batch_size,
                                              sampler=test_samples, pin_memory=True,
                                               num_workers=4,drop_last=False)

    # Models
    class_nums = 1
#    model = PPIModel(class_nums,window_size)
    model = torch.load(model_file)
    if torch.torch.cuda.is_available():
#        pretrained_model = torch.load(model_file)
#        model.load_state_dict(pretrained_model, strict=False)

#        model = torch.load('model_0826_test.pth')

#        torch.save(model, 'model_0826_test.pth')
        
        model = model.cuda()
        model.eval()
    else:
#        pretrained_model = torch.load(model_file, map_location='cpu')
#        model.load_state_dict(pretrained_model, strict=False)
        model.eval()

    test(model, test_loader,path_dir,threshold, test_name)


def make_prediction(model_file_name, dataset, threshold=0.50, test_name=''):

    window_size = 3
    path_dir = "./"

    if not os.path.exists(path_dir):
        os.makedirs(path_dir)

    print('Start predicting...')

    predict(model_file_name,dataset,window_size,path_dir,threshold, test_name)
