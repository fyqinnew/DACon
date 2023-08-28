import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn.init as init
import numpy as np
import torch.optim as optim
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
import cmath
import optuna
from sklearn.neighbors import NearestNeighbors
import os



class GradReverse(torch.autograd.Function):
    '''
    Extension of grad reverse layer
    '''
    @staticmethod
    def forward(ctx, x, constant):
        ctx.constant = constant
        return x.view_as(x)

    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * ctx.constant
        return grad_output, None

    def grad_reverse(x, constant):
        return GradReverse.apply(x, constant)

class FeatureExtractor(nn.Module):
    def __init__(self, input_dim):
        super(FeatureExtractor, self).__init__()
        self.ftrLayer1 = nn.Linear(input_dim, 64)
        self.ftrLayer2 = nn.Linear(64, 32)
        self.ftrlayer3 = nn.Linear(32, 16)

    def forward(self, x):
        x = F.relu(self.ftrLayer1(x))
        x = F.relu(self.ftrLayer2(x))
        x = F.relu(self.ftrlayer3(x))
        return x

class Class_Classifier(nn.Module):
    def __init__(self):
        super(Class_Classifier, self).__init__()
        self.ClassLayer1 = nn.Linear(16, 8)
        self.ClassLayer2 = nn.Linear(8, 2)

    def forward(self, x):
        x = F.relu(self.ClassLayer1(x))
        x = self.ClassLayer2(x)
        x = F.log_softmax(x, 1)
        return x

class Domain_Classifier(nn.Module):
    def __init__(self):
        super(Domain_Classifier, self).__init__()
        self.DomainLayer1 = nn.Linear(16, 8)
        self.DomainLayre2 = nn.Linear(8, 2)

    def forward(self, x, constant):
        x = GradReverse.grad_reverse(x, constant)
        x = F.relu(self.DomainLayer1(x))
        x = self.DomainLayre2(x)
        x = F.log_softmax(x, 1)
        return x

def evaluation_metrics(ori_label, pred_label):
    '''
    get the pd, pf, bal, precision, recall, f1, auc evaluation measures values
    :param ori_label: the original label of the sample
    :param pred_label: the predicted label of the sample
    :return: pd, pf, bal, precision, recall, f1, auc
    '''
    tn, fp, fn, tp = confusion_matrix(ori_label, pred_label).ravel()

    pd = tp /(tp + fn)
    pf = fp /(fp + tn)
    temp = pf**2 + (1-pd)**2
    bal = (1 - cmath.sqrt(temp/2)).real
    precision = precision_score(ori_label, pred_label, zero_division=0.0)
    recall = recall_score(ori_label, pred_label)
    f1 = f1_score(ori_label, pred_label, zero_division=0.0)
    auc = roc_auc_score(ori_label, pred_label)

    return pd, pf, bal, precision, recall, f1, auc


def train(training_mode, feature_extractor, class_classifier, domain_classifier, class_criterion, domain_criterion,
          source_metric, source_label, target_metric, target_label, optimizer, theta, epoch):
    '''
    Execute target domain adaptation
    :param training_mode: the training mode of the model
    :param feature_extractor: network used to extract feature from source and target samples
    :param class_classifier: network used to predict labels
    :param domain_classifier: network used to predict domain
    :param class_criterion:
    :param domain_criterion:
    :param source_metric: metric of source domain
    :param source_label: label of source domain
    :param target_metric: metric of target domain
    :param target_label: label of target domain
    :param optimizer:
    :param epoch:
    :return:
    '''

    # setup models
    feature_extractor.train()
    class_classifier.train()
    domain_classifier.train()

    if training_mode == 'dann':

        constant = theta

        # prepare the data
        input1 = source_metric
        label1 = source_label
        input2 = target_metric
        label2 = target_label

        # setup optimizer
        optimizer.zero_grad()

        # prepare domain labels
        source_labels = torch.zeros((input1.size()[0])).type(torch.LongTensor)
        target_labels = torch.ones((input2.size()[0])).type(torch.LongTensor)

        source_labels = source_labels.to(my_device)
        target_labels = target_labels.to(my_device)

        # compute the output of source domain and target domain
        src_feature = feature_extractor(input1)
        tgt_feature = feature_extractor(input2)

        # compute the class loss of src_feature
        class_preds = class_classifier(src_feature)
        class_loss = class_criterion(class_preds, label1)

        # compute the domain loss of src_feature and target_feature
        tgt_preds = domain_classifier(tgt_feature, constant)
        src_preds = domain_classifier(src_feature, constant)
        tgt_loss = domain_criterion(tgt_preds, target_labels)
        src_loss = domain_criterion(src_preds, source_labels)
        domain_loss = tgt_loss + src_loss

        loss = class_loss + theta * domain_loss
        loss.backward()
        optimizer.step()

 

def test(feature_extractor, class_classifier, domain_classifier, eval_metric, eval_label1):
    '''
    Test the performance of the model
    :param feature_extractor: network used to extract feature from target samples
    :param class_classifier: network used to predict labels
    :param domain_classifier: network used to predict domain
    :param eval_dataloader: dataloader of evaluate data
    :return: the prediction performance, pd, pf, bal, precision, recall, f1, auc
    '''

    # setup the network
    feature_extractor.eval()
    class_classifier.eval()
    domain_classifier.eval()

    eval_input = eval_metric
    eval_label = eval_label1

    eval_output = class_classifier(feature_extractor(eval_input))
    eval_pred = eval_output.data.max(1)[1]
    eval_ori_label = eval_label.tolist()
    eval_pred_label = eval_pred.tolist()

    eval_pd, eval_pf, eval_bal, eval_precision, eval_recall, eval_f1, eval_auc = evaluation_metrics(eval_ori_label, eval_pred_label)

    return eval_pd, eval_pf, eval_bal, eval_precision, eval_recall, eval_f1, eval_auc


def z_score(data_to_normalize):
    label = data_to_normalize[:,-1] 
    data = data_to_normalize[:,:-1]
    meanvalue, stdvalue = data.mean(axis=0), data.std(axis=0) 
    index = (stdvalue == 0)
    stdvalue[index] = 1.0 
    data -= meanvalue
    data /= stdvalue
    result = np.column_stack((data, label))
    return result

#SMOTE
def SMOTE(x_feature, y_label):
    index_min = (y_label==1)
    x_min = x_feature[index_min,:]
    diff = int(len(x_feature) - 2 * len(x_min))
    label_min = 1.
    if diff<=0:
        index_min = (y_label==0)
        x_min = x_feature[index_min,:]
        label_min = 0.
        diff = abs(diff)
    neighbors = NearestNeighbors(n_neighbors=5+1).fit(x_min)
    distances, indices0 = neighbors.kneighbors(x_min)
    indices = indices0[:,1:]

    sample_indices = np.random.randint(low=0, high=len(x_min), size=int(diff/2))
    point_set, count = [], 0
    for i in range(len(sample_indices)):
        
        sample = sample_indices[i]
        sample_index = np.random.randint(low=0, high=5, size=2)
       
        origin = x_min[sample]
        for j in range(0,2):
            
            target = x_min[indices[sample, sample_index[j]]]
            step = np.random.uniform(0,1)
            new_point = origin + step*(target - origin)
            point_set.append(list(new_point))
            count += 1
    x_new = np.vstack((x_feature, np.array(point_set)))
    y_new = np.hstack((y_label, np.array([label_min]*count)))
    return x_new, y_new

def objective(trial, mode, input_dim, src_train_metric, src_train_label, src_test_metric, src_test_label, tgt_metric,
              tgt_label, class_weight, theta, epochs, my_evaluation_metric, src_name, tgt_name):
    # init models
    feature_extractor = FeatureExtractor(input_dim)
    class_classifier = Class_Classifier()
    domain_classifier = Domain_Classifier()

   
    if my_evaluation_metric == 'balance':
        best_val_bal = 0.2929
    elif my_evaluation_metric == 'precision':
        best_val_precision = 0
    elif my_evaluation_metric == 'recall':
        best_val_recall = 0
    elif my_evaluation_metric == 'f1_measure':
        best_val_f1 = 0
    elif my_evaluation_metric == 'auc':
        best_val_auc = 0

    trigger_times = 0
    patience = 5

    if use_gpu:
        feature_extractor.to(my_device)
        class_classifier.to(my_device)
        domain_classifier.to(my_device)

    # init criterions
    class_criterion = nn.NLLLoss(weight=class_weight)
    domain_criterion = nn.NLLLoss()
    lr = trial.suggest_float("lr", 1e-5, 10, log=True)
    # init optimizer
    optimizer = optim.SGD([{'params': feature_extractor.parameters()},
                           {'params': class_classifier.parameters()},
                           {'params': domain_classifier.parameters()}], lr=lr, momentum=0.9)


    for epoch in range(epochs):
        #print('Epoch: {}'.format(epoch))
        train(mode, feature_extractor, class_classifier, domain_classifier, class_criterion, domain_criterion,
              src_train_metric, src_train_label, tgt_metric, tgt_label, optimizer, theta, epoch)

        test_pd, test_pf, test_bal, test_precision, test_recall, test_f1, test_auc = test(feature_extractor, class_classifier, domain_classifier, src_test_metric, src_test_label)
		
        # early stopping
        if my_evaluation_metric == 'balance':
            if test_bal <= best_val_bal:
                if test_bal > 0.5:
                    trigger_times += 1
                else:
                    trigger_times = 0
            else:
                trigger_times = 0
                best_val_bal = test_bal
            if trigger_times > patience:
                # print('early stopping!\n')
                break
        elif my_evaluation_metric == 'precision':
            if test_precision <= best_val_precision:
                if test_precision > 0.5:
                    trigger_times += 1
                else:
                    trigger_times = 0
            else:
                trigger_times = 0
                best_val_precision = test_precision
            if trigger_times > patience:
                # print('early stopping!\n')
                break
        elif my_evaluation_metric == 'recall':
            if test_recall <= best_val_recall:
                if test_recall > 0.5:
                    trigger_times += 1
                else:
                    trigger_times = 0
            else:
                trigger_times = 0
                best_val_recall = test_recall
            if trigger_times > patience:
                # print('early stopping!\n')
                break
        elif my_evaluation_metric == 'f1_measure':
            if test_f1 <= best_val_f1:
                if test_f1 > 0.5:
                    trigger_times += 1
                else:
                    trigger_times = 0
            else:
                trigger_times = 0
                best_val_f1 = test_f1
            if trigger_times > patience:
                # print('early stopping!\n')
                break
        elif my_evaluation_metric == 'auc':
            if test_auc <= best_val_auc:
                if test_auc > 0.5:
                    trigger_times += 1
                else:
                    trigger_times = 0
            else:
                trigger_times = 0
                best_val_auc = test_auc
            if trigger_times > patience:
                # print('early stopping!\n')
                break


    # save the model to a file
    checkpoint_path = "checkpoint_path/" + "checkpoint_" + src_name + "_to_"+ tgt_name + "_" +str(trial.number) + ".pt"
    torch.save({
        'trial_value': trial.number,
        'feature_extractor_state_dict': feature_extractor.state_dict(),
        'class_classifier_state_dict': class_classifier.state_dict(),
        'domain_classifier_state_dict': domain_classifier.state_dict(),
    }, checkpoint_path)

    if my_evaluation_metric == 'balance':
        return test_bal
    elif my_evaluation_metric == 'precision':
        return test_precision
    elif my_evaluation_metric == 'recall':
        return test_recall
    elif my_evaluation_metric == 'f1_measure':
        return test_f1
    elif my_evaluation_metric == 'auc':
        return test_auc



# The place where program begins
# These are model parameters
epochs = 500
training_mode = 'dann'

# model params
use_gpu = True

if use_gpu:
	my_device = torch.device('cuda')
else:
	my_device = torch.device('cpu')

theta = 10
repeat_times = 100

balance, precision, recall, f1_measure, auc
chosen_evaluation_metric = 'recall'
class_weight = torch.tensor([1, 1], dtype=torch.float, device=my_device)

#data
source_name = 'Accumulo'
target_name = 'Hadoop1'
source_path = 'ConBug/' + source_name + '.csv'
target_path = 'ConBug/' + target_name + '.csv'
result_path = 'Result/' + source_name + '_to_' + target_name + '_' + chosen_evaluation_metric + '.csv'
source_file = np.loadtxt(source_path, dtype='float', delimiter=',', skiprows=1, encoding='utf-8')
target_file = np.loadtxt(target_path, dtype='float', delimiter=',', skiprows=1, encoding='utf-8')
# use all the metrics in DACon
source_data, target_data = z_score(source_file.copy()), z_score(target_file.copy())

# only concurrency metric
# source_file_slice = source_file[:,49:]
# target_file_slice = target_file[:,49:]
# source_data, target_data = z_score(source_file_slice.copy()), z_score(target_file_slice.copy())

# only sequential metric
# source_file_slice = np.column_stack((source_file[:,:49], source_file[:,-1]))
# target_file_slice = np.column_stack((target_file[:,:49], target_file[:,-1]))
# source_data, target_data = z_score(source_file_slice.copy()), z_score(target_file_slice.copy())

source_metric_ori = source_data[:, :-1]
target_metric_ori = target_data[:, :-1]
source_label_ori = source_data[:, -1]
target_label_ori = target_data[:, -1]

if chosen_evaluation_metric == 'balance':
    prediction_result = np.zeros((repeat_times+3, 3))
elif chosen_evaluation_metric == 'precision':
    prediction_result = np.zeros((repeat_times+3, 1))
elif chosen_evaluation_metric == 'recall':
    prediction_result = np.zeros((repeat_times+3, 1))
elif chosen_evaluation_metric == 'f1_measure':
    prediction_result = np.zeros((repeat_times+3, 3))
elif chosen_evaluation_metric == 'auc':
    prediction_result = np.zeros((repeat_times+3, 1))

for repeat_time in range(repeat_times):
    # stratified sampling
    print("the current repetition is: ")
    print(repeat_time)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=1 / 3)
    for train_idx, test_idx in sss.split(source_metric_ori, source_label_ori):
        source_train_metric_ori, source_test_metric_ori = source_metric_ori[train_idx], source_metric_ori[test_idx]
        source_train_label_ori, source_test_label_ori = source_label_ori[train_idx], source_label_ori[test_idx]

    # smote
    source_train_metric_smote, source_train_label_smote = SMOTE(source_train_metric_ori, source_train_label_ori)

    # no smote
    # source_train_metric_smote = source_train_metric_ori
    # source_train_label_smote = source_train_label_ori

    src_train_metric_tensor = torch.from_numpy(source_train_metric_smote).float().to(my_device)
    src_train_label_tensor = torch.from_numpy(source_train_label_smote).long().to(my_device)
    src_test_metric_tensor = torch.from_numpy(source_test_metric_ori).float().to(my_device)
    src_test_label_tensor = torch.from_numpy(source_test_label_ori).long().to(my_device)
    tgt_metric_tensor = torch.from_numpy(target_metric_ori).float().to(my_device)
    tgt_label_tensor = torch.from_numpy(target_label_ori).long().to(my_device)

    input_dim = np.size(source_train_metric_ori, axis=1)


    study = optuna.create_study(sampler=optuna.samplers.TPESampler(), direction="maximize")
    study.optimize(lambda trial: objective(trial, training_mode, input_dim, src_train_metric_tensor, src_train_label_tensor,
                                           src_test_metric_tensor, src_test_label_tensor, tgt_metric_tensor, tgt_label_tensor,
                                           class_weight, theta, epochs, chosen_evaluation_metric, source_name, target_name), n_trials=100)


    diandian2 = study.best_params
    print("the best parameter is:\n")
    print(diandian2)

    # this is used to get the best model based on the max of evaluation measure
    best_trial_num = study.best_trial.number

    print("Best trial config： {}".format(study.best_trial.value))

    best_feature_extractor = FeatureExtractor(input_dim)
    best_class_classifier = Class_Classifier()
    best_domain_classifier = Domain_Classifier()

    if use_gpu:
        best_feature_extractor.to(my_device)
        best_class_classifier.to(my_device)
        best_domain_classifier.to(my_device)


    best_checkpoint_path = "checkpoint_path/" + "checkpoint_" + source_name + "_to_" + target_name + "_" + str(best_trial_num) + ".pt"
    best_checkpoint = torch.load(best_checkpoint_path)
    best_feature_extractor.load_state_dict(best_checkpoint['feature_extractor_state_dict'])
    best_class_classifier.load_state_dict(best_checkpoint['class_classifier_state_dict'])
    best_domain_classifier.load_state_dict(best_checkpoint['domain_classifier_state_dict'])

    # setup the network
    best_feature_extractor.eval()
    best_class_classifier.eval()
    best_domain_classifier.eval()


    tgt_pd, tgt_pf, tgt_bal, tgt_precision, tgt_recall, tgt_f1, tgt_auc = test(best_feature_extractor, best_class_classifier, best_domain_classifier, tgt_metric_tensor, tgt_label_tensor)

    if chosen_evaluation_metric == 'balance':
        print('target pd: ({:.4f}), pf: ({:.4f}), bal: ({:.4f})\n'.format(tgt_pd, tgt_pf, tgt_bal))
        prediction_result[repeat_time, :] = np.array([tgt_pd, tgt_pf, tgt_bal])
    elif chosen_evaluation_metric == 'precision':
        print('target precision: ({:.4f})\n'.format(tgt_precision))
        prediction_result[repeat_time, :] = np.array([tgt_precision])
    elif chosen_evaluation_metric == 'recall':
        print('target recall: ({:.4f})\n'.format(tgt_recall))
        prediction_result[repeat_time, :] = np.array([tgt_recall])
    elif chosen_evaluation_metric == 'f1_measure':
        print('target precision: ({:.4f}), recall: ({:.4f}), f1_measure: ({:.4f})\n'.format(tgt_precision, tgt_recall, tgt_f1))
        prediction_result[repeat_time, :] = np.array([tgt_precision, tgt_recall, tgt_f1])
    elif chosen_evaluation_metric == 'auc':
        print('target auc: ({:.4f})\n'.format(tgt_auc))
        prediction_result[repeat_time, :] = np.array([tgt_auc])


prediction_result_average = np.mean(prediction_result[:repeat_times, :], axis=0)
prediction_result_std = np.std(prediction_result[:repeat_times, :], axis=0, ddof=1)

prediction_result[repeat_times + 1,:] = prediction_result_average
prediction_result[repeat_times + 2,:] = prediction_result_std

np.savetxt(result_path, prediction_result, delimiter=',', fmt='%0.6f')


