import pathlib
import os
import json 
from dataclasses import dataclass
import numpy as np

def read_metadata(data_path: str):
  """Read metadata of datasets

  Args:
    data_path: Path to metadata JSON file

  Returns:
    metadata json object
  """
  with open(os.path.join(data_path, 'metadata.json'), 'rt') as fp:
    return json.load(fp)
  

def get_p4_files_list(datapath, ext='*.p4p'):
    """
    Return sorted list of p4 files in a directory.

    Files are sorted based on timestep and list is a `pathlib.Path`.
    """
    directory = pathlib.Path(datapath)  
    file_list = list(directory.rglob(ext))
    file_list.sort(key=lambda fname: int(fname.name.split('_')[-1].split('.')[0]))

    return file_list


@dataclass
class ParticleData:
    timestep: np.float64
    num_particles: int
    id: np.ndarray
    group: np.ndarray
    volume: np.ndarray
    mass: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    avg_velocity: np.ndarray
    avg_acceleration: np.ndarray
    ang_velocity: np.ndarray
    torque: np.ndarray

    def __post_init__(self):
         self.radius = ((3 * self.volume) / (4 * np.pi))**(1/3)
         self.diameter = self.radius * 2
         self.density = self.mass / self.volume


def read_gns_file(gns_particle_file, version='timestep'):
    if version == 'timestep':
        data = np.loadtxt(gns_particle_file, skiprows=3)

        with open(gns_particle_file) as file:
            head = [next(file) for _ in range(2)]
            timestep, num_p = head[1].rstrip().split()

        if data.size > 0:
           return ParticleData(float(timestep), int(num_p), data[:, 0], data[:, 1], data[:, 2], data[:, 3],
                        data[:, 4:7], data[:, 7:10], data[:, 10:13],data[:, 13:16],
                        data[:, 16:19], data[:, 19:22])
        else:
            return ParticleData(float(timestep), int(num_p), np.array([]), np.array([]), np.array([]), np.array([]),
                np.array([]), np.array([]), np.array([]), np.array([]),
                np.array([]), np.array([]))


class Cylinder():
    """
    Cylinder Primitive

    """
    def __init__(self, point1, point2, radius):
        """
        Constructor for Cylinder Primitive.

        Args:
            point1 (list): First point defining CylinderBin [x, y, z].
            point2 (list): Second point defining CylinderBin [x, y, z].
            radius (float): Radius of CylinderBin [m].
        """
        self.radius = radius
        
        # Points
        self.point1 = np.array(point1)
        self.point2 = np.array(point2)
        
        # Vectors
        self.vector = self.point2 - self.point1

        # Vector Norm
        self.cylinder_height = np.linalg.norm(self.vector)
        self.vector_norm = self.vector / self.cylinder_height