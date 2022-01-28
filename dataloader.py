import string
import numpy as np
from general_processor import Utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale, LabelEncoder
import torch
from torch.utils.data import DataLoader, TensorDataset

def MIEEGdataloader(path: string ,
                    channel: string = "a",
                    batch_size : int = 10 ,
                    augment : bool = True,
                    k_neighbors : int = 5,
                    random_state : int = 42):    
    # Load data
    if channel not in ["a","b","c","d","e","F"]:
        print(f"Channel {channel} does not exists")
        return False
    channels = Utils.combinations[channel] #["FC1", "FC2"], ["FC3", "FC4"], ["FC5", "FC6"]]

    exclude =  [38, 88, 89, 92, 100, 104]
    subjects = [n for n in np.arange(1,110) if n not in exclude]
    #Load data
    x, y = Utils.load(channels, subjects, base_path = path)
    #Transform y to Label
    le = LabelEncoder()
    y_lab = le.fit_transform(y)
    #Reshape for scaling
    reshaped_x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
    #Grab a test set before SMOTE
    x_train_raw, x_valid_test_raw, y_train_raw, y_valid_test_raw = train_test_split(reshaped_x,
                                                                                y_lab,
                                                                                stratify=y_lab,
                                                                                test_size=0.20,
                                                                                random_state=random_state)

    #Scale indipendently train/test
    x_train_scaled_raw = minmax_scale(x_train_raw, axis=1)
    x_test_valid_scaled_raw = minmax_scale(x_valid_test_raw, axis=1)

    #Create Validation/test
    x_valid_raw, x_test_raw, y_valid, y_test = train_test_split(x_test_valid_scaled_raw,
                                                        y_valid_test_raw,
                                                        stratify=y_valid_test_raw,
                                                        test_size=0.50,
                                                        random_state=42)

    x_valid = x_valid_raw.reshape(x_valid_raw.shape[0], int(x_valid_raw.shape[1]/2),2).astype(np.float64)
    x_test = x_test_raw.reshape(x_test_raw.shape[0], int(x_test_raw.shape[1]/2),2).astype(np.float64)

    #apply smote to train data
    if augment:        
        print('classes count')
        print ('before oversampling = {}'.format(np.unique(y_train_raw, return_counts=True)[1]))
        # smote
        from imblearn.over_sampling import SMOTE
        sm = SMOTE(k_neighbors= k_neighbors, random_state=random_state)
        x_train_smote_raw, y_train = sm.fit_resample(x_train_scaled_raw, y_train_raw)
        print('classes count')
        print ('after oversampling = {}'.format(np.unique(y_train, return_counts=True)[1]))
    else:
        print ('classes count = {}'.format(np.unique(y_train_raw, return_counts=True)[1]))
        x_train_smote_raw = x_train_scaled_raw
        y_train = y_train_raw
    x_train = x_train_smote_raw.reshape(x_train_smote_raw.shape[0], int(x_train_smote_raw.shape[1]/2), 2).astype(np.float64)

    train_dataset = TensorDataset( torch.tensor(x_train).permute(0, 2, 1), torch.tensor(y_train) )
    valid_dataset = TensorDataset( torch.tensor(x_valid).permute(0, 2, 1), torch.tensor(y_valid) )
    test_dataset = TensorDataset( torch.tensor(x_test).permute(0, 2, 1), torch.tensor(y_test) )
    train_loader = DataLoader(train_dataset, batch_size= batch_size, shuffle= True)
    valid_loader = DataLoader(valid_dataset, batch_size= batch_size, shuffle= False)
    test_loader = DataLoader(test_dataset, batch_size = 1, shuffle= False)
    
    return train_loader, valid_loader, test_loader


if __name__ == '__main__':   
    print(True)