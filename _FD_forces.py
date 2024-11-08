import torch
import torch_geometric
from Data_processing import transform_data
import numpy as np


class FinDiffForces:
    def __init__(
        self,
        model,  # torch.nn.Module,
        data,
        delta: float = 1e-4,
    ):
        self.modelt = model
        self.delta = delta
        self.data = data

        self.num_atoms = len(data.entry)
        self.forces = np.zeros((self.num_atoms, 3))
        self.initial_positions = data.entry.positions.copy()

    def get_energy(
        self,
        new_positions: np.ndarray,
    ):
        _data = self.data.clone()
        ndata = transform_data(_data, new_positions)

        self.modelt.to("cpu")
        self.modelt.eval()
        with torch.inference_mode():
            en, _ = self.modelt(ndata)  # total energy, energy per atom
        return en

    def get_forces(self) -> np.ndarray:
        for i in range(self.num_atoms):  # atomic iterator
            for j in range(3):  # directional iterator

                # Initialize perturbed positions
                pos_plus = self.initial_positions.copy()
                pos_minus = self.initial_positions.copy()

                # Apply small displacement delta in the positive and negative direction
                pos_plus[i, j] += self.delta
                pos_minus[i, j] -= self.delta

                # Calculate the energy at the perturbed positions
                energy_plus = self.get_energy(pos_plus)
                energy_minus = self.get_energy(pos_minus)

                # Finite difference
                dE_dX = (energy_plus - energy_minus) / (2 * self.delta)

                # Compute force
                _force = -dE_dX

                # Store the force
                self.forces[i, j] = _force

        return self.forces
