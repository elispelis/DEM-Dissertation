import collections
import json
import os
import pickle
import glob
import re
import sys
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

import argparse

import logging
logger = logging.getLogger(__name__)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from gns import learned_simulator_baseline as learned_simulator
from gns import noise_utils
from gns import reading_utils
from gns import data_loader

import utils.utils as utils

INPUT_SEQUENCE_LENGTH = 6  # So we can calculate the last 5 velocities.
NUM_PARTICLE_TYPES = 9
KINEMATIC_PARTICLE_ID = 3

def rollout(
        simulator: learned_simulator.LearnedSimulator,
        position: torch.Tensor,
        particle_types: torch.Tensor,
        edge_index: torch.Tensor,
        device):
    """Rolls out a trajectory by applying the model in sequence.

    Args:
        simulator: Learned simulator.
        features: Torch tensor features.
        nsteps: Number of steps.
    """
    position = position.to(device)
    initial_positions = position[:, :INPUT_SEQUENCE_LENGTH]
    edge_index = edge_index[INPUT_SEQUENCE_LENGTH-1:]
    ground_truth_positions = position[:, INPUT_SEQUENCE_LENGTH:]

    current_positions = initial_positions
    predictions = []
    particle_types = particle_types.to(device)
    nsteps = ground_truth_positions.shape[1]
    for step in range(nsteps):
        # Get next position with shape (nnodes, dim)
        next_position = simulator.predict_positions(
            current_positions,
            particle_types=particle_types,
            edge_index=torch.tensor(edge_index[step]).to(device).contiguous(),
        )

        # Update kinematic particles from prescribed trajectory.
        kinematic_mask = (particle_types == KINEMATIC_PARTICLE_ID).clone().detach().to(device)
        next_position_ground_truth = ground_truth_positions[:, step]
        kinematic_mask = kinematic_mask.bool()[:, None].expand(-1, current_positions.shape[-1])
        next_position = torch.where(
            kinematic_mask, next_position_ground_truth, next_position)
        predictions.append(next_position)

        # Shift `current_positions`, removing the oldest position in the sequence
        # and appending the next position at the end.
        current_positions = torch.cat(
            [current_positions[:, 1:], next_position[:, None, :]], dim=1)

    # Predictions with shape (time, nnodes, dim)
    predictions = torch.stack(predictions)
    ground_truth_positions = ground_truth_positions.permute(1, 0, 2)

    loss = (predictions - ground_truth_positions) ** 2

    output_dict = {
        'initial_positions': initial_positions.permute(1, 0, 2).cpu().numpy(),
        'predicted_rollout': predictions.cpu().numpy(),
        'ground_truth_rollout': ground_truth_positions.cpu().numpy(),
        'particle_types': particle_types.cpu().numpy(),
    }

    return output_dict, loss

def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)
            
def _get_simulator(
        metadata: json,
        acc_noise_std: float,
        vel_noise_std: float,
        device: str) -> learned_simulator.LearnedSimulator:
    """Instantiates the simulator.

    Args:
        metadata: JSON object with metadata.
        acc_noise_std: Acceleration noise std deviation.
        vel_noise_std: Velocity noise std deviation.
        device: PyTorch device 'cpu' or 'cuda'.
    """

    # Normalization stats
    normalization_stats = {
        'acceleration': {
            'mean': torch.FloatTensor(metadata['acc_mean']).to(device),
            'std': torch.sqrt(torch.FloatTensor(metadata['acc_std'])**2 +
                                acc_noise_std**2).to(device),
        },
        'velocity': {
            'mean': torch.FloatTensor(metadata['vel_mean']).to(device),
            'std': torch.sqrt(torch.FloatTensor(metadata['vel_std'])**2 +
                                vel_noise_std**2).to(device),
        },
    }

    simulator = learned_simulator.LearnedSimulator(
        particle_dimensions=metadata['dim'],
        nnode_in=37 if metadata['dim'] == 3 else 30,
        nedge_in=metadata['dim'] + 1,
        latent_dim=128,
        nmessage_passing_steps=5,
        nmlp_layers=2,
        mlp_hidden_dim=128,
        boundaries=np.array(metadata['bounds']),
        normalization_stats=normalization_stats,
        nparticle_types=NUM_PARTICLE_TYPES,
        particle_type_embedding_size=16,
        device=device)

    return simulator

def eval_on_step(device: str, flags):
    """Predict rollouts.

    Args:
        simulator: Trained simulator if not will undergo training.

    """
    logger = utils.init_logger(is_main=True, is_distributed=False, filename=f'{flags["log_path"]}valid_{flags["exp_id"]}.log')
    metadata = reading_utils.read_metadata(flags["data_path"])
    simulator = _get_simulator(metadata, flags["noise_std"], flags["noise_std"], device)

    max_loss = torch.tensor(float('inf'))
    best_checkpoint_step = 0
    ds = data_loader.get_data_loader_SAG_Mill_by_trajectories_baseline(path=f'{flags["data_path"]}', input_length_sequence=INPUT_SEQUENCE_LENGTH, valid_ratio=flags["valid_ratio"])
    sequence_len = len(ds)
    
    for checkpoint_step in range(flags["start_checkpoint"], flags["end_checkpoint"], flags["chechpoint_step"]):
        # Load simulator
        logger.info("Load {}-th step checkpoint".format(checkpoint_step))
        if os.path.exists(flags["model_path"] + f'{flags["exp_id"]}-model-{checkpoint_step}.pt'):
            simulator.load(flags["model_path"] + f'{flags["exp_id"]}-model-{checkpoint_step}.pt')
        else:
            logger.error('Model file {} not found.'.format(flags["model_path"] + f'{flags["exp_id"]}-model-{checkpoint_step}.pt'))
            raise FileNotFoundError('Model file {} not found.'.format(flags["model_path"] + f'{flags["exp_id"]}-model-{checkpoint_step}.pt'))
        simulator.to(device)
        simulator.eval()

        eval_loss = []
        with torch.no_grad():

            position, particle_types, edge_index, n_particles_per_example = ds.data()
            # Predict example rollout
            example_rollout, loss = rollout(simulator, position, particle_types, edge_index, device)

            example_rollout['metadata'] = metadata
            loss = torch.sqrt(loss[:, :, 0]**2 + loss[:, :, 1]**2 + loss[:, :, 2]**2)
            logger.info("Predicting loss: {:.6f}".format(loss.mean()))
            eval_loss.append(torch.flatten(loss))

            # Save rollout in testing
            if flags["save_rollout"]:
                example_rollout['metadata'] = metadata
                filename = f'{flags["exp_id"]}_rollout_{checkpoint_step}_{loss.mean():.4f}.pkl'
                filename = os.path.join(flags["output_path"], filename)
                with open(filename, 'wb') as f:
                    pickle.dump(example_rollout, f)

        logger.info("Mean loss on rollout prediction with {}-th step checkpoint: {:.6f}".format(
            checkpoint_step, loss.mean()))
        if loss.mean() < max_loss:
            max_loss = loss.mean()
            best_checkpoint_step = checkpoint_step
            logger.info("Best checkpoint step: {}".format(best_checkpoint_step))
    
    logger.info("Save best checkpoint step: {}".format(best_checkpoint_step))
    os.system(f'cp {flags["model_path"]}{flags["exp_id"]}-model-{best_checkpoint_step}.pt {flags["model_path"]}{flags["exp_id"]}-model-{best_checkpoint_step}-best.pt')
    os.system(f'cp {flags["model_path"]}{flags["exp_id"]}-train-state-{best_checkpoint_step}.pt {flags["model_path"]}{flags["exp_id"]}-train-state-{best_checkpoint_step}-best.pt')
    
    simulator.load(f'{flags["model_path"]}{flags["exp_id"]}-model-{best_checkpoint_step}-best.pt')
    simulator.to(device)
    simulator.eval()
    with torch.no_grad():
        position, particle_types, edge_index, n_particles_per_example = ds.data()
        # Predict example rollout
        example_rollout, loss = rollout(simulator, position, particle_types, edge_index, device)

        loss = torch.sqrt(loss[:, :, 0]**2 + loss[:, :, 1]**2 + loss[:, :, 2]**2)
        logger.info("Predicting loss: {:.6f}".format(loss.mean()))
        error_trend = torch.mean(loss, dim=1).cpu().numpy()
        np.save(f'{flags["output_path"]}{flags["exp_id"]}-error-trend.npy', error_trend)
    
if __name__ == '__main__':
    default_data_path = "/mnt/raid0sata1/hcj/SAG_Mill/Train_Data"

    default_ds_name = "SAGMill"
    default_res_dir="./results"

    default_model_path=f"{default_res_dir}/models/{default_ds_name}/"
    default_rollouts_path=f"{default_res_dir}/rollouts/{default_ds_name}/"

    myflags = {}
    myflags["data_path"] = default_data_path
    myflags["noise_std"] = 6.7e-4
    myflags["model_path"] = default_model_path
    myflags["exp_id"] = 0
    myflags["save_rollout"] = True
    myflags["start_checkpoint"] = 0
    myflags["end_checkpoint"] = 60000+1
    myflags["chechpoint_step"] = 2000
    myflags["output_path"] = default_rollouts_path
    myflags["log_path"] = default_res_dir
    myflags["valid_ratio"] = 0.3
    
    eval_on_step('cuda:7', myflags)