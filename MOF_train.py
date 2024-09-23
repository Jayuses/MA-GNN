import os
import math
import pickle
import shutil
import yaml
import time
import torch
import pickle
import torch.nn as nn
import numpy as np
import pandas as pd
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Subset
from sklearn.metrics import mean_absolute_error,r2_score

from models.sc_net import TMCnet

from Dataset.MOF_data import MOF_dataset,MOF_wrapper

from torch_geometric.utils import degree


def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('./config.yaml', os.path.join(model_checkpoints_folder, 'config.yaml'))

class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor,device):
        """tensor is taken as a sample to calculate the mean and std"""
        if len(tensor.size()) > 1:
            self.mean = torch.mean(tensor,dim=0).to(device)
            self.std = torch.std(tensor,dim=0).to(device)
        else:
            self.mean = torch.mean(tensor).to(device)
            self.std = torch.std(tensor).to(device)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']

class tmQM_train(object):
    def __init__(self,dataset,config) -> None:
        self.config = config
        self.device = self._get_device()

        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        dir_name = current_time + '_' + config['experiment']['name']
        log_dir = os.path.join(config['experiment']['path'], dir_name)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.dataset = dataset

        self.criterion = nn.MSELoss()

    def _get_device(self):
        if torch.cuda.is_available() and self.config['gpu'] != 'cpu':
            device = self.config['gpu']
            torch.cuda.set_device(device)
        else:
            device = 'cpu'
        print("Running on:", device)

        return device
    
    def _step(self,model,data):
        pred = model(data)
        if self.config['label_normalize']:
            loss = self.criterion(pred, self.normalizer.norm(data.y.reshape(-1,self.config['out_dimention'])))
        else:
            loss = self.criterion(pred, data.y.reshape(-1,self.config['out_dimention']))

        return loss
        
    
    def train(self):
        print(time.ctime())
        train_loader, valid_loader, test_loader = self.dataset.get_data_loaders()
        with open(self.config['experiment']['path']+'/testset.pkl','wb') as f:
            pickle.dump(test_loader.sampler.indices,f)

        if self.config['model'] == 'PNA':
            train_dataset = Subset(train_loader.dataset,train_loader.sampler.indices)
            # Compute the maximum in-degree in the training data.
            max_degree = -1
            for data in train_dataset:
                d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
                max_degree = max(max_degree, int(d.max()))

            # Compute the in-degree histogram tensor
            deg = torch.zeros(max_degree + 1, dtype=torch.long)
            for data in train_dataset:
                d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
                deg += torch.bincount(d, minlength=deg.numel())
        else:
            deg = None

        model = TMCnet(
            GNN_config=self.config['GNN'],
            Heter_config=self.config['Heter'],
            out_dimention=self.config['out_dimention'],
            gnn=self.config['model'],
            deg=deg
        )

        model.to(self.device)

        if self.config['label_normalize']:
            labels = []
            for d in train_loader:
                labels.append(d.y.reshape(-1,self.config['out_dimention']))
            labels = torch.cat(labels)
            self.normalizer = Normalizer(labels,self.device)
            print('Average:',self.normalizer.mean)
            print('std:',self.normalizer.std)
            print('size:',labels.shape)

        optimizer = torch.optim.Adam(
            model.parameters(),
            weight_decay=self.config['weight_decay'],
            lr=self.config['init_lr']
        )

        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')
        _save_config_file(model_checkpoints_folder)

        best_valid_loss = np.inf
        start = time.time()

        for epoch in range(self.config['epochs']):
            num_iter = 0
            for bn,data in enumerate(train_loader):
                optimizer.zero_grad()
                
                data = data.to(self.device)
                loss = self._step(model,data)
                if num_iter % self.config['log_every_n_steps'] == 0:     
                    print(epoch, bn, loss.item())              
                loss.backward()
                optimizer.step()
                num_iter += 1
                
            self.writer.add_scalar('train_loss', loss, global_step=epoch)
            
            scheduler.step()

            # validate the model if requested
            if epoch % self.config['eval_every_n_epochs'] == 0:
                valid_loss = self._validate(model, valid_loader)
                if valid_loss < best_valid_loss:
                    # save the model weights
                    best_valid_loss = valid_loss
                    torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))

                self.writer.add_scalar('validation_loss', valid_loss, global_step=epoch)
            end = time.time()
            print(f'Running Time: {end-start} seconds')
            start = end

        self._test(model, test_loader)


        print(time.ctime())

    def _validate(self,model,valid_loader):
        predictions = []
        labels = []
        with torch.no_grad():
            model.eval()

            valid_loss = 0.0
            num_data = 0
            for bn, data in enumerate(valid_loader):           
                data = data.to(self.device)
                pred = model(data)
                loss = self._step(model,data)

                valid_loss += loss.item() * data.y.size(0)
                num_data += data.y.size(0)

                if self.config['label_normalize']:
                    pred = self.normalizer.denorm(pred)

                if self.device == 'cpu':
                    predictions.extend(pred.detach().numpy())
                    labels.extend(data.y.numpy())
                else:
                    predictions.extend(pred.cpu().detach().numpy())
                    labels.extend(data.y.cpu().numpy())

            valid_loss /= num_data
        
        model.train()

        predictions = np.array(predictions)
        labels = np.array(labels).reshape(-1,predictions.shape[1])
        mae = mean_absolute_error(labels, predictions,multioutput='raw_values')
        print('Validation loss:', valid_loss)
        print('MAE:',mae)
        return valid_loss
    
    def _test(self,model,test_loader):
        model_path = os.path.join(self.writer.log_dir, 'checkpoints', 'model.pth')
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        print("Loaded trained model with success.")

        # test steps
        predictions = []
        labels = []
        with torch.no_grad():
            model.eval()

            test_loss = 0.0
            num_data = 0
            for bn, data in enumerate(test_loader):
                data = data.to(self.device)

                pred = model(data)
                
                loss = self._step(model,data)

                test_loss += loss.item() * data.y.size(0)
                num_data += data.y.size(0)

                if self.config['label_normalize']:
                    pred = self.normalizer.denorm(pred)

                if self.device == 'cpu':
                    predictions.extend(pred.detach().numpy())
                    labels.extend(data.y.flatten().numpy())
                else:
                    predictions.extend(pred.cpu().detach().numpy())
                    labels.extend(data.y.cpu().flatten().numpy())

            test_loss /= num_data
        
        model.train()

        predictions = np.array(predictions)
        labels = np.array(labels).reshape(-1,predictions.shape[1])
        self.mae = mean_absolute_error(labels, predictions,multioutput='raw_values')
        self.R2 = r2_score(labels,predictions,multioutput='raw_values')
        print('Test loss:', test_loss)
        print(f'MAE:{self.mae}\n')

def main(config,predata=None):
    dataset = MOF_wrapper(config['path'],config['batch_size'],predata=predata,**config['data'])

    tmc = tmQM_train(dataset,config)
    tmc.train()

    return tmc.mae,tmc.R2

if __name__ == "__main__":
    config = yaml.load(open("config_mof.yaml", "r"), Loader=yaml.FullLoader)
    task_list1 = ['SchNet_hMOF_1_4_xyz_2_attention16']
    task_list2 = ['MPNN_QMOF_1_4_DG_1_attention16']
    task_list3 = ['AttentiveFP_CoREMOF_1_4_DG_0_attention16','AttentiveFP_CoREMOF_1_4_DG_1_attention16']
    task_list = task_list1
    # print('Generate datasets......')
    # predata = tmQM_dataset(config['path'],config['separated'])
    
    if config['experiment']['name'] == 'test':
            config['experiment']['path'] = 'experiment/test'
            config['experiment']['suffix'] = 'test'
            if not os.path.exists(config['experiment']['path']):
                os.mkdir(config['experiment']['path'])

            print(f"Experiment:{config['experiment']['name']}\n")

            results = main(config)
    else:
        for task in task_list:
            task_config = task.split(sep='_')
            config['experiment']['name'] = task
            config['experiment']['path'] = 'mof_experment/'+task

            config['data']['database'] = task_config[1]

            config['GNN']['in_channels'] = int(task_config[2])
            config['GNN']['edge_dim'] = int(task_config[3])

            config['data']['pe'] = task_config[4]

            if len(task_config[0]) > 5 and task_config[0][:5] == 'MLHGT':
                config['model'] = task_config[0][:5]
                config['Heter']['conv'] = task_config[0][6:]
            else:
                config['model'] = task_config[0]

            config['data']['label_index'] = int(task_config[5])
            
            if len(task_config) == 7:
                if task_config[6][:9] == 'attention':
                    config['GNN']['pool'] = task_config[6][:9]
                    config['GNN']['apool'] = int(task_config[6][9:])
                else:
                    config['GNN']['pool'] = task_config[6]
            
            if len(task_config) == 8:
                config['GNN']['mp'] = task_config[7]
                

            if not os.path.exists(config['experiment']['path']):
                os.mkdir(config['experiment']['path'])

            print(f"Model:{config['model']}")
            print(f"Train set:{1-(config['data']['valid_size']+config['data']['test_size'])}")
            print(f"Epoch:{config['epochs']}")
            print(f"Experiment:{config['experiment']['name']}\n")

            mae_list = []
            r2_list = []

            print('Generate datasets......\n')
            database_name = config['data']['database']
            with open(f'./Dataset/MOF/{database_name}.pkl', 'rb') as f:
                dataset = pickle.load(f)
            predata = MOF_dataset(config['path'],config['data']['database'],
                                 label_index=config['data']['label_index'],data_list=dataset)

            mae,r2 = main(config,predata=predata)
            mae_list.append(mae)
            r2_list.append(r2)

            df1  = pd.DataFrame(mae_list)
            df1.to_csv(
                config['experiment']['path']+'/mae.csv',
                mode='a',index=False, header=False
            )

            df2  = pd.DataFrame(r2_list)
            df2.to_csv(
                config['experiment']['path']+'/r2.csv',
                mode='a',index=False, header=False
            )