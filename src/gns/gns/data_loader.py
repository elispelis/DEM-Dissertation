import torch
import numpy as np
import re
import glob
import os.path as osp
import pandas as pd

def load_npz_data(path):
    """Load data stored in npz format.
    
    The file format for Python 3.9 or less supports ragged arrays and Python 3.10
    requires a structured array. This function supports both formats.

    Args:
        path (str): Path to npz file.
    
    Returns:
        data (list): List of tuples of the form (positions, particle_type).
    """
    with np.load(path, allow_pickle=True) as data_file:
        if 'gns_data' in data_file:
            data = data_file['gns_data']
        else:
            data = [item for _, item in data_file.items()]
    return data

class SamplesDataset(torch.utils.data.Dataset):
    """Dataset of samples of trajectories.
    
    Each sample is a tuple of the form (positions, particle_type).
    positions is a numpy array of shape (sequence_length, n_particles, dimension).
    particle_type is an integer.

    Args:
        path (str): Path to dataset.
        input_length_sequence (int): Length of input sequence.

    Attributes:
        _data (list): List of tuples of the form (positions, particle_type).
        _dimension (int): Dimension of the data.
        _input_length_sequence (int): Length of input sequence.
        _data_lengths (list): List of lengths of trajectories in the dataset.
        _length (int): Total number of samples in the dataset.
        _precompute_cumlengths (np.array): Precomputed cumulative lengths of trajectories in the dataset.
    """
    def __init__(self, path, input_length_sequence):
        super().__init__()
        # load dataset stored in npz format
        # data is loaded as dict of tuples
        # of the form (positions, particle_type)
        # convert to list of tuples
        # TODO: allow_pickle=True is potential security risk. See docs.
        self._data = load_npz_data(path)
        
        # length of each trajectory in the dataset
        # excluding the input_length_sequence
        # may (and likely is) variable between data
        self._dimension = self._data[0][0].shape[-1]
        self._input_length_sequence = input_length_sequence
        self._data_lengths = [x.shape[0] - self._input_length_sequence for x, _ in self._data]
        self._length = sum(self._data_lengths)

        # pre-compute cumulative lengths
        # to allow fast indexing in __getitem__
        self._precompute_cumlengths = [sum(self._data_lengths[:x]) for x in range(1, len(self._data_lengths)+1)]
        self._precompute_cumlengths = np.array(self._precompute_cumlengths, dtype=int)

    def __len__(self):
        """Return length of dataset.
        
        Returns:
            int: Length of dataset.
        """
        return self._length

    def __getitem__(self, idx):
        """Returns a training example from the dataset.
        
        Args:
            idx (int): Index of training example.

        Returns:
            tuple: Tuple of the form ((positions, particle_type, n_particles_per_example), label).
        """
        # Select the trajectory immediately before
        # the one that exceeds the idx
        # (i.e., the one in which idx resides).
        trajectory_idx = np.searchsorted(self._precompute_cumlengths - 1, idx, side="left")

        # Compute index of pick along time-dimension of trajectory.
        start_of_selected_trajectory = self._precompute_cumlengths[trajectory_idx-1] if trajectory_idx != 0 else 0
        time_idx = self._input_length_sequence + (idx - start_of_selected_trajectory)

        # Prepare training data.
        positions = self._data[trajectory_idx][0][time_idx - self._input_length_sequence:time_idx]
        positions = np.transpose(positions, (1, 0, 2)) # nparticles, input_sequence_length, dimension
        particle_type = np.full(positions.shape[0], self._data[trajectory_idx][1], dtype=int)
        n_particles_per_example = positions.shape[0]
        label = self._data[trajectory_idx][0][time_idx]

        return ((positions, particle_type, n_particles_per_example), label)

def collate_fn(data):
    """Collate function for SamplesDataset.
    Args:
        data (list): List of tuples of the form ((positions, particle_type, n_particles_per_example), label).

    Returns:
        tuple: Tuple of the form ((positions, particle_type, n_particles_per_example), label).    
    """
    position_list = []
    particle_type_list = []
    n_particles_per_example_list = []
    label_list = []

    for ((positions, particle_type, n_particles_per_example), label) in data:
        position_list.append(positions)
        particle_type_list.append(particle_type)
        n_particles_per_example_list.append(n_particles_per_example)
        label_list.append(label)

    return ((
        torch.tensor(np.vstack(position_list)).to(torch.float32).contiguous(), 
        torch.tensor(np.concatenate(particle_type_list)).contiguous(),
        torch.tensor(n_particles_per_example_list).contiguous(),
        ),
        torch.tensor(np.vstack(label_list)).to(torch.float32).contiguous()
        )
    
def collate_fn_baseline(data):
    """Collate function for SamplesDataset.
    Args:
        data (list): List of tuples of the form ((positions, particle_type, n_particles_per_example), label).

    Returns:
        tuple: Tuple of the form ((positions, particle_type, n_particles_per_example), label).    
    """
    position_list = []
    particle_type_list = []
    n_particles_per_example_list = []
    edge_index_list = []
    label_list = []

    for ((positions, particle_type, n_particles_per_example, edge_index), label) in data:
        position_list.append(positions)
        particle_type_list.append(particle_type)
        n_particles_per_example_list.append(n_particles_per_example)
        edge_index_list.append(edge_index)
        label_list.append(label)

    return ((
        torch.tensor(np.vstack(position_list)).to(torch.float32).contiguous(), 
        torch.tensor(np.concatenate(particle_type_list)).contiguous(),
        torch.tensor(n_particles_per_example_list).contiguous(),
        torch.tensor(np.vstack(edge_index_list)).contiguous(),
        ),
        torch.tensor(np.vstack(label_list)).to(torch.float32).contiguous()
        )


class TrajectoriesDataset(torch.utils.data.Dataset):
    """Dataset of trajectories.

    Each trajectory is a tuple of the form (positions, particle_type).
    positions is a numpy array of shape (sequence_length, n_particles, dimension).
    """
    def __init__(self, path):
        super().__init__()
        # load dataset stored in npz format
        # data is loaded as dict of tuples
        # of the form (positions, particle_type)
        # convert to list of tuples
        # TODO (jpv): allow_pickle=True is potential security risk. See docs.
        self._data = load_npz_data(path)
        self._dimension = self._data[0][0].shape[-1]
        self._length = len(self._data)

    def __len__(self):
        """Return length of dataset.
        
        Returns:
            int: Length of dataset.
        """
        return self._length

    def __getitem__(self, idx):
        """Returns a training example from the dataset.
        
        Args:
            idx (int): Index of training example.
            
        Returns:
            tuple: Tuple of the form (positions, particle_type).
        """
        positions, _particle_type = self._data[idx]
        positions = np.transpose(positions, (1, 0, 2))
        particle_type = np.full(positions.shape[0], _particle_type, dtype=int)
        n_particles_per_example = positions.shape[0]
        return (
            torch.tensor(positions).to(torch.float32).contiguous(), 
            torch.tensor(particle_type).contiguous(), 
            n_particles_per_example
        )
        
class SAGMillTrajectoriesDataset_Baseline(torch.utils.data.Dataset):
    def __init__(self, path, input_length_sequence, valid_ratio=0.7):
        super().__init__()
        # load SAG Mill particle dataset - p4p
        timestemp_list = glob.glob(osp.join(path, "SAG_Mill_*.p4p"))
        timestemp_list = list(map(lambda x: re.split(r'[/_.]', x)[-2], timestemp_list))
        timestemp_list = sorted(timestemp_list, key=int)
        timestemp_list = timestemp_list[4:]
        
        timestemp_list = timestemp_list[int(len(timestemp_list) * (1 - valid_ratio)) - input_length_sequence: ] 
        
        self._dimension = 3
        self._data_lengths = len(timestemp_list) - 1
            
        self._position = []
        self._edge_index = []
        self._particle_type = None

        columns = ['ID', 'GROUP', 'VOLUME', 'MASS', 'POS_X', 'POS_Y','POS_Z', 'VEL_X', 'VEL_Y',
                   'VEL_Z', 'AVG_VEL_X', 'AVG_VEL_Y', 'AVG_VEL_Z', 'AVG_ACC_X', 'AVG_ACC_Y', 'AVG_ACC_Z',
                   'ANG_VEL_X', 'ANG_VEL_Y', 'ANG_VEL_Z', 'TORQ_X', 'TORQ_Y', 'TORQ_Z']
        for i in timestemp_list:
            data = pd.read_csv(osp.join(path, "SAG_Mill_{}.p4p".format(i)), skiprows=3, sep=" ", header=None)
            data.columns = columns
            self._position.append(data[['POS_X', 'POS_Y', 'POS_Z']].to_numpy())
            
        data = pd.read_csv(osp.join(path, "SAG_Mill_{}.p4p".format(timestemp_list[-1])), skiprows=3, sep=" ", header=None)
        data.columns = columns
        self._particle_type = data['GROUP'].to_numpy()
              
        columns = ['P1', 'P2', 'CPOS_X', 'CPOS_Y', 'CPOS_Z', 'F_X', 'F_Y', 'F_Z', 'FN_X',
                   'FN_Y', 'FN_Z', 'FT_X', 'FT_Y', 'FT_Z', 'OVLP_N']
        for i in timestemp_list:
            data = pd.read_csv(osp.join(path, "SAG_Mill_{}.p4c".format(i)), skiprows=3, sep=" ", header=None)
            data.columns = columns
            data['P1'] = data['P1'] - 1
            data['P2'] = data['P2'] - 1
            self._edge_index.append(data[['P1', 'P2']].to_numpy().T)
            
    def data(self):
        positions = np.stack(self._position, axis=0)
        positions = np.transpose(positions, (1, 0, 2)) # nparticles, input_sequence_length, dimension
        
        particle_type = self._particle_type
        n_particles_per_example = len(particle_type)
        
        edge_index = self._edge_index
        
        return torch.tensor(positions).to(torch.float32).contiguous(), \
            torch.tensor(particle_type).contiguous(), \
            edge_index, n_particles_per_example
        
    def __len__(self):
        return self._data_lengths
    
    def __getitem__(self, idx):
        positions = self._position[idx]
        positions = positions[np.newaxis, :]
        positions = np.transpose(positions, (1, 0, 2)) # nparticles, input_sequence_length, dimension
        particle_type = self._particle_type
        n_particles_per_example = len(particle_type)
        edge_index = self._edge_index[idx]
        
        return (torch.tensor(positions).to(torch.float32).contiguous(), 
            torch.tensor(particle_type).contiguous(), 
            n_particles_per_example, torch.tensor(edge_index).contiguous())

class SAGMillSamplesDataset_Baseline(torch.utils.data.Dataset):
    def __init__(self, path, input_length_sequence, train_ratio=0.3):
        super().__init__()
        # load SAG Mill particle dataset - p4p
        timestemp_list = glob.glob(osp.join(path, "SAG_Mill_*.p4p"))
        timestemp_list = list(map(lambda x: re.split(r'[/_.]', x)[-2], timestemp_list))
        timestemp_list = sorted(timestemp_list, key=int)
        timestemp_list = timestemp_list[4:]
        
        timestemp_list = timestemp_list[:int(len(timestemp_list) * train_ratio)] 
        
        self._dimension = 3
        self._input_length_sequence = input_length_sequence
        self._data_lengths = len(timestemp_list) - self._input_length_sequence - 1
            
        self._position = []
        self._edge_index = []
        self._particle_type = None

        columns = ['ID', 'GROUP', 'VOLUME', 'MASS', 'POS_X', 'POS_Y','POS_Z', 'VEL_X', 'VEL_Y',
                   'VEL_Z', 'AVG_VEL_X', 'AVG_VEL_Y', 'AVG_VEL_Z', 'AVG_ACC_X', 'AVG_ACC_Y', 'AVG_ACC_Z',
                   'ANG_VEL_X', 'ANG_VEL_Y', 'ANG_VEL_Z', 'TORQ_X', 'TORQ_Y', 'TORQ_Z']
        for i in timestemp_list:
            data = pd.read_csv(osp.join(path, "SAG_Mill_{}.p4p".format(i)), skiprows=3, sep=" ", header=None)
            data.columns = columns
            self._position.append(data[['POS_X', 'POS_Y', 'POS_Z']].to_numpy())
            
        data = pd.read_csv(osp.join(path, "SAG_Mill_{}.p4p".format(timestemp_list[-1])), skiprows=3, sep=" ", header=None)
        data.columns = columns
        self._particle_type = data['GROUP'].to_numpy()
              
        columns = ['P1', 'P2', 'CPOS_X', 'CPOS_Y', 'CPOS_Z', 'F_X', 'F_Y', 'F_Z', 'FN_X',
                   'FN_Y', 'FN_Z', 'FT_X', 'FT_Y', 'FT_Z', 'OVLP_N']
        for i in timestemp_list:
            data = pd.read_csv(osp.join(path, "SAG_Mill_{}.p4c".format(i)), skiprows=3, sep=" ", header=None)
            data.columns = columns
            data['P1'] = data['P1'] - 1
            data['P2'] = data['P2'] - 1
            self._edge_index.append(data[['P1', 'P2']].to_numpy().T)
            
        
    def __len__(self):
        return self._data_lengths
    
    def __getitem__(self, idx):
        positions = self._position[idx:idx+self._input_length_sequence]
        positions = np.stack(positions, axis=0)
        positions = np.transpose(positions, (1, 0, 2)) # nparticles, input_sequence_length, dimension
        particle_type = self._particle_type
        n_particles_per_example = len(particle_type)
        label = self._position[idx+self._input_length_sequence]
        edge_index = self._edge_index[idx+self._input_length_sequence-1]
        
        return ((positions, particle_type, n_particles_per_example, edge_index), label)
    
def get_data_loader_SAG_Mill_baseline(path, input_length_sequence, batch_size, train_ratio, shuffle=True):
    """Returns a data loader for the dataset.
    
    Args:
        path (str): Path to dataset.
        input_length_sequence (int): Length of input sequence.
        batch_size (int): Batch size.
        shuffle (bool, optional): Whether to shuffle the dataset. Defaults to True.
        
    Returns:
        torch.utils.data.DataLoader: Data loader for the dataset.
    """
    dataset = SAGMillSamplesDataset_Baseline(path, input_length_sequence, train_ratio)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                       pin_memory=True, collate_fn=collate_fn_baseline)
        
def get_data_loader_by_samples(path, input_length_sequence, batch_size, shuffle=True):
    """Returns a data loader for the dataset.
    
    Args:
        path (str): Path to dataset.
        input_length_sequence (int): Length of input sequence.
        batch_size (int): Batch size.
        shuffle (bool, optional): Whether to shuffle the dataset. Defaults to True.
        
    Returns:
        torch.utils.data.DataLoader: Data loader for the dataset.
    """
    dataset = SamplesDataset(path, input_length_sequence)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                       pin_memory=True, collate_fn=collate_fn)

def get_data_loader_by_trajectories(path):
    """Returns a data loader for the dataset.
    
    Args:
        path (str): Path to dataset.

    Returns:
        torch.utils.data.DataLoader: Data loader for the dataset.
    """
    dataset = TrajectoriesDataset(path)
    return torch.utils.data.DataLoader(dataset, batch_size=None, shuffle=False,
                                       pin_memory=True)
    
def get_data_loader_SAG_Mill_by_trajectories_baseline(path, input_length_sequence, valid_ratio=0.1):
    """Returns a data loader for the dataset.
    
    Args:
        path (str): Path to dataset.
        input_length_sequence (int): Length of input sequence.
        batch_size (int): Batch size.
        shuffle (bool, optional): Whether to shuffle the dataset. Defaults to True.
        
    Returns:
        torch.utils.data.DataLoader: Data loader for the dataset.
    """
    dataset = SAGMillTrajectoriesDataset_Baseline(path, input_length_sequence, valid_ratio=valid_ratio)
    # return torch.utils.data.DataLoader(dataset, batch_size=None, shuffle=False,
    #                                    pin_memory=True)
    return dataset