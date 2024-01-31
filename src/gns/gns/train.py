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

parser = argparse.ArgumentParser(description='GNS simulator training script')

parser.add_argument(
    '--mode', type=str, default='train', choices=['train', 'valid', 'rollout'],
    help='Train model, validation or rollout evaluation.')
parser.add_argument('--batch_size', type=int, default=32, help='The batch size.')
parser.add_argument('--noise_std', type=float, default=6.7e-4, help='The std deviation of the noise.')
parser.add_argument('--data_path', type=str, default=None, help='The dataset directory.')
parser.add_argument('--model_path', type=str, default='models/', help=('The path for saving checkpoints of the model.'))
parser.add_argument('--output_path', type=str, default='rollouts/', help='The path for saving outputs (e.g. rollouts).')
parser.add_argument('--model_file', type=str, default=None, help=('Model filename (.pt) to resume from. Can also use "latest" to default to newest file.'))
parser.add_argument('--train_state_file', type=str, default=None, help=('Train state filename (.pt) to resume from. Can also use "latest" to default to newest file.'))
parser.add_argument('--exp_id', type=str, default='exp-test', help='Experiment ID.')
parser.add_argument('--ntraining_steps', type=int, default=int(2E7), help='Number of training steps.')
parser.add_argument('--nvalid_steps', type=int, default=int(2000), help='Number of steps at which to valid the model.')
parser.add_argument('--nsave_steps', type=int, default=int(2000), help='Number of steps at which to save the model.')
parser.add_argument('--nlog_steps', type=int, default=int(100), help='Number of steps at which to log the model.')
# Learning rate parameters
parser.add_argument('--lr_init', type=float, default=1e-4, help='Initial learning rate.')
parser.add_argument('--lr_decay', type=float, default=0.1, help='Learning rate decay.')
parser.add_argument('--lr_decay_steps', type=int, default=int(5e6), help='Learning rate decay steps.')

args = parser.parse_args()

Stats = collections.namedtuple('Stats', ['mean', 'std'])

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


def predict(flags):
  """Predict rollouts.

  Args:
    simulator: Trained simulator if not will undergo training.

  """
  device = flags["device"]
  metadata = reading_utils.read_metadata(flags["data_path"])
  simulator = _get_simulator(metadata, flags["noise_std"], flags["noise_std"], device)

  # Load simulator
  if os.path.exists(flags["model_path"] + flags["model_file"]):
    simulator.load(flags["model_path"] + flags["model_file"])
  else:
    raise FileNotFoundError(f'Model file {flags["model_path"] + flags["model_file"]} not found.')
  simulator.to(device)
  simulator.eval()

  # Output path
  if not os.path.exists(flags["output_path"]):
    os.makedirs(flags["output_path"])

  # Use `valid`` set for eval mode if not use `test`
  split = 'test' if flags["mode"] == 'rollout' else 'valid'

  ds = data_loader.get_data_loader_by_trajectories(path=f'{flags["data_path"]}{split}.npz')

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
      print("Predicting example {} loss: {}".format(example_i, loss.mean()))
      eval_loss.append(torch.flatten(loss))

      # Save rollout in testing
      if flags["mode"] == 'rollout':
        example_rollout['metadata'] = metadata
        filename = f'rollout_{example_i}.pkl'
        filename = os.path.join(flags["output_path"], filename)
        with open(filename, 'wb') as f:
          pickle.dump(example_rollout, f)

  print("Mean loss on rollout prediction: {}".format(
      torch.mean(torch.cat(eval_loss))))

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

def train(flags):
  """Train the model.

  Args:
    rank: local rank
    world_size: total number of ranks
  """   
  is_cuda = flags["is_cuda"]
  is_main = flags["is_main"]
  is_distributed = flags["is_distributed"]
  device = flags["device"]
  
  rank = flags["local_rank"]
  world_size = flags["world_size"]
  
  if is_cuda:
    logger = utils.init_logger(is_main=is_main, is_distributed=is_distributed, filename=f'{flags["log_path"]}run_{flags["exp_id"]}.log')
    logger.info(f"Running on GPU {rank}.")
  else:
    logger = utils.init_logger(is_main=True, is_distributed=False, filename=f'{flags["log_path"]}run_{flags["exp_id"]}.log')
    logger.info(f"Running on CPU.")

  metadata = reading_utils.read_metadata(flags["data_path"])

  if is_cuda:
    serial_simulator = _get_simulator(metadata, flags["noise_std"], flags["noise_std"], device)
    simulator = DDP(serial_simulator.to(device), device_ids=[rank], output_device=device)
    optimizer = torch.optim.Adam(simulator.parameters(), lr=flags["lr_init"] * world_size)
  else:
    simulator = _get_simulator(metadata, flags["noise_std"], flags["noise_std"], device)
    optimizer = torch.optim.Adam(simulator.parameters(), lr=flags["lr_init"] * world_size)
  step = 0

  # If model_path does exist and model_file and train_state_file exist continue training.
  if flags["model_file"] is not None:

    if flags["model_file"] == "latest" and flags["train_state_file"] == "latest":
      # find the latest model, assumes model and train_state files are in step.
      fnames = glob.glob(f'{flags["model_path"]}{flags["exp_id"]}-model-*pt')
      max_model_number = 0
      expr = re.compile(f'.{flags["exp_id"]}-model-(\d+).pt')
      for fname in fnames:
        model_num = int(expr.search(fname).groups()[0])
        if model_num > max_model_number:
          max_model_number = model_num
      # reset names to point to the latest.
      flags["model_file"] = f'{flags["exp_id"]}-model-{max_model_number}.pt'
      flags["train_state_file"] = f"{flags['exp_id']}-train-state-{max_model_number}.pt"

    if os.path.exists(flags["model_path"] + flags["model_file"]) and os.path.exists(flags["model_path"] + flags["train_state_file"]):
      # load model
      simulator.module.load(flags["model_path"] + flags["model_file"])

      # load train state
      train_state = torch.load(flags["model_path"] + flags["train_state_file"])
      # set optimizer state
      optimizer = torch.optim.Adam(simulator.module.parameters())
      optimizer.load_state_dict(train_state["optimizer_state"])
      optimizer_to(optimizer, rank)
      # set global train state
      step = train_state["global_train_state"].pop("step")

    else:
      msg = f'Specified model_file {flags["model_path"] + flags["model_file"]} and train_state_file {flags["model_path"] + flags["train_state_file"]} not found.'
      # raise FileNotFoundError(msg)
      logger.info(msg)
      logger.info("Starting training from scratch.")

  simulator.train()
  simulator.to(device)

  if is_cuda:
    dl = distribute.get_data_distributed_dataloader_by_samples(path=f'{flags["data_path"]}train.npz',
                                                               input_length_sequence=INPUT_SEQUENCE_LENGTH,
                                                               batch_size=flags["batch_size"],
                                                               )
  else:
    dl = data_loader.get_data_loader_by_samples(path=f'{flags["data_path"]}train.npz',
                                                input_length_sequence=INPUT_SEQUENCE_LENGTH,
                                                batch_size=flags["batch_size"],
                                                )

  print(f"rank = {rank}, cuda = {torch.cuda.is_available()}")
  not_reached_nsteps = True
  try:
    while not_reached_nsteps:
      if is_cuda:
        torch.distributed.barrier()
      else:
        pass
      for ((position, particle_type, n_particles_per_example), labels) in dl:
        position.to(device)
        particle_type.to(device)
        n_particles_per_example.to(device)
        labels.to(device)

        # TODO (jpv): Move noise addition to data_loader
        # Sample the noise to add to the inputs to the model during training.
        sampled_noise = noise_utils.get_random_walk_noise_for_position_sequence(position, noise_std_last_step=flags["noise_std"]).to(device)
        non_kinematic_mask = (particle_type != KINEMATIC_PARTICLE_ID).clone().detach().to(device)
        sampled_noise *= non_kinematic_mask.view(-1, 1, 1)

        # Get the predictions and target accelerations.
        if is_cuda:
          pred_acc, target_acc = simulator.module.predict_accelerations(
              next_positions=labels.to(device),
              position_sequence_noise=sampled_noise.to(device),
              position_sequence=position.to(device),
              nparticles_per_example=n_particles_per_example.to(device),
              particle_types=particle_type.to(device))
        else:
          pred_acc, target_acc = simulator.predict_accelerations(
            next_positions=labels.to(device),
            position_sequence_noise=sampled_noise.to(device),
            position_sequence=position.to(device),
            nparticles_per_example=n_particles_per_example.to(device),
            particle_types=particle_type.to(device))

        # Calculate the loss and mask out loss on kinematic particles
        loss = (pred_acc - target_acc) ** 2
        loss = loss.sum(dim=-1)
        num_non_kinematic = non_kinematic_mask.sum()
        loss = torch.where(non_kinematic_mask.bool(),
                         loss, torch.zeros_like(loss))
        loss = loss.sum() / num_non_kinematic

        # Computes the gradient of loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update learning rate
        lr_new = flags["lr_init"] * (flags["lr_decay"] ** (step/flags["lr_decay_steps"])) * world_size
        for param in optimizer.param_groups:
          param['lr'] = lr_new

        if is_main:
          if step % flags["nlog_steps"] == 0:
            torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.SUM)
            loss /= world_size
            logger.info(f'Training step: {step}/{flags["ntraining_steps"]}. Loss: {loss}.')
          # Save model state
          if step % flags["nsave_steps"] == 0:
            if not is_cuda:
              simulator.save(flags["model_path"] + f'{flags["exp_id"]}-model-'+str(step)+'.pt')
            else:
              simulator.module.save(flags["model_path"] + f'{flags["exp_id"]}-model-'+str(step)+'.pt')
            train_state = dict(optimizer_state=optimizer.state_dict(), global_train_state={"step":step})
            torch.save(train_state, f'{flags["model_path"]}{flags["exp_id"]}-train-state-{step}.pt')

        # Complete training
        if (step >= flags["ntraining_steps"]):
          not_reached_nsteps = False
          break

        step += 1

  except KeyboardInterrupt:
    pass

  # if is_main:
  #   if not is_cuda:
  #     simulator.save(flags["model_path"] + 'model-'+str(step)+'.pt')
  #   else:
  #     simulator.module.save(flags["model_path"] + 'model-'+str(step)+'.pt')
  #   train_state = dict(optimizer_state=optimizer.state_dict(), global_train_state={"step":step})
  #   torch.save(train_state, f'{flags["model_path"]}train_state-{step}.pt')

  if is_cuda:
    distribute.cleanup()


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


def main():
  """Train or evaluates the model.

  """
  myflags = {}
  myflags["data_path"] = args.data_path
  myflags["noise_std"] = args.noise_std
  myflags["lr_init"] = args.lr_init
  myflags["lr_decay"] = args.lr_decay
  myflags["lr_decay_steps"] = args.lr_decay_steps
  myflags["batch_size"] = args.batch_size
  myflags["ntraining_steps"] = args.ntraining_steps
  myflags["nvalid_steps"] = args.nvalid_steps
  myflags["nsave_steps"] = args.nsave_steps
  myflags["model_file"] = args.model_file
  myflags["model_path"] = args.model_path
  myflags["train_state_file"] = args.train_state_file
  myflags["mode"] = args.mode
  myflags["output_path"] = args.output_path
  myflags["log_path"] = "./results/"
  myflags["exp_id"] = args.exp_id
  myflags["nlog_steps"] = args.nlog_steps
  
  myflags = utils.init_distritubed_mode(myflags)

  # Read metadata
  if args.mode == 'train':
    # If model_path does not exist create new directory.
    if not os.path.exists(myflags["model_path"]):
      os.makedirs(myflags["model_path"])

    torch.distributed.barrier()
    train(myflags)

  elif args.mode in ['valid', 'rollout']:
    torch.distributed.barrier()
    predict(myflags)

if __name__ == '__main__':
  main()