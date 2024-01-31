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
from gns import learned_simulator
from gns import noise_utils
from gns import reading_utils
from gns import data_loader
from gns import distribute

import utils.utils as utils

INPUT_SEQUENCE_LENGTH = 6  # So we can calculate the last 5 velocities.
NUM_PARTICLE_TYPES = 9
KINEMATIC_PARTICLE_ID = 3

def rollout(
        simulator: learned_simulator.LearnedSimulator,
        position: torch.tensor,
        particle_types: torch.tensor,
        n_particles_per_example: torch.tensor,
        nsteps: int,
        device):
    """Rolls out a trajectory by applying the model in sequence.

    Args:
        simulator: Learned simulator.
        features: Torch tensor features.
        nsteps: Number of steps.
    """
    initial_positions = position[:, :INPUT_SEQUENCE_LENGTH]
    ground_truth_positions = position[:, INPUT_SEQUENCE_LENGTH:]

    current_positions = initial_positions
    predictions = []

    for step in range(nsteps):
        # Get next position with shape (nnodes, dim)
        next_position = simulator.predict_positions(
            current_positions,
            nparticles_per_example=[n_particles_per_example],
            particle_types=particle_types,
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
        nmessage_passing_steps=10,
        nmlp_layers=2,
        mlp_hidden_dim=128,
        connectivity_radius=metadata['default_connectivity_radius'],
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

        ds = data_loader.get_data_loader_by_trajectories(path=f'{flags["data_path"]}valid.npz')

        eval_loss = []
        with torch.no_grad():
            for example_i, (positions, particle_type, n_particles_per_example) in enumerate(ds):
                positions.to(device)
                particle_type.to(device)
                n_particles_per_example = torch.tensor([int(n_particles_per_example)], dtype=torch.int32).to(device)

                nsteps = metadata['sequence_length'] - INPUT_SEQUENCE_LENGTH
                # Predict example rollout
                example_rollout, loss = rollout(simulator, positions.to(device), particle_type.to(device),
                                                n_particles_per_example.to(device), nsteps, device)

                example_rollout['metadata'] = metadata
                logger.info("Predicting example {} loss: {:.6f}".format(example_i, loss.mean()))
                eval_loss.append(torch.flatten(loss))

                # Save rollout in testing
                if flags["save_rollout"]:
                    example_rollout['metadata'] = metadata
                    filename = f'{flags["exp_id"]}_rollout_{example_i}_{checkpoint_step}_{loss.mean():.4f}.pkl'
                    filename = os.path.join(flags["output_path"], filename)
                    with open(filename, 'wb') as f:
                        pickle.dump(example_rollout, f)

        logger.info("Mean loss on rollout prediction with {}-th step checkpoint: {}".format(
            checkpoint_step, torch.mean(torch.cat(eval_loss))))
        if torch.mean(torch.cat(eval_loss)) < max_loss:
            loss = torch.mean(torch.cat(eval_loss))
            best_checkpoint_step = checkpoint_step
            logger.info("Best checkpoint step: {}".format(best_checkpoint_step))
    
    logger.info("Save best checkpoint step: {}".format(best_checkpoint_step))
    os.system(f'cp {flags["model_path"]}{flags["exp_id"]}-model-{best_checkpoint_step}.pt {flags["model_path"]}{flags["exp_id"]}-model-{best_checkpoint_step}-best.pt')
    os.system(f'cp {flags["model_path"]}{flags["exp_id"]}-train-state-{best_checkpoint_step}.pt {flags["model_path"]}{flags["exp_id"]}-train-state-{best_checkpoint_step}-best.pt')
    
if __name__ == '__main__':
    dataset = 'WaterDropSample'
    myflags = {}
    myflags["data_path"] = f"/mnt/data/hcj/GNS_TrainingData/{dataset}/dataset/"
    myflags["noise_std"] = 6.7e-4
    myflags["model_path"] = f"../results/models/{dataset}/"
    myflags["exp_id"] = 0
    myflags["save_rollout"] = True
    myflags["start_checkpoint"] = 0
    myflags["end_checkpoint"] = 32000+1
    myflags["chechpoint_step"] = 2000
    myflags["output_path"] = f"../results/rollouts/{dataset}/"
    myflags["log_path"] = "../results/"
    
    eval_on_step('cuda:0', myflags)