
# -*- coding: utf-8 -*-
"""
High level TES models for HVAC-oriented studies.

TESsimple: 
    - Basic approach. Tracks energy stored based on inputs, outputs, and losses.
    - Calculates state of charge (SOC) and checks energy balance.

TES_PCM_ideal:
    - Incorporates further details of PCM behavior. 


Created on:     11/23/2021
Last update:    06/01/2026
@author: conrado ermel

"""


from scipy.optimize import fsolve
import numpy as np
from matplotlib import pyplot as plt
from CoolProp.CoolProp import PropsSI
import math




class TESsimple:
    # Constructor method to initialize the TESsimple object with properties of a PCM (phase change material) thermal energy storage system
    def __init__(self, PCM, Qin, Qout, SOC_old):
        
        # Extract key parameters from the PCM dictionary
        Scapacity = PCM['Scapacity']  # Storage capacity of the PCM
        UA = PCM['UA']                # Heat transfer coefficient (U) times area (A), representing heat loss
        Ttrans = PCM['Ttrans']        # Transition temperature of the PCM (e.g., melting point)
        Tamb = PCM['Tamb']            # Ambient temperature

        # Calculate the energy stored in the TES based on the state of charge (SOC_old), inputs, and outputs
        self.Energy = (SOC_old * Scapacity) + Qin - Qout - UA * (Ttrans - Tamb)

        # Ensure that energy stays within valid bounds [0, Scapacity]
        self.Energy = 0 if self.Energy < 0 else self.Energy         # Prevent energy from going below zero
        self.Energy = Scapacity if self.Energy > Scapacity else self.Energy  # Cap energy at storage capacity

        # Update the state of charge (SOC) based on the current energy level
        self.SOC = self.Energy / Scapacity  # SOC is the ratio of stored energy to total capacity

        # Calculate an energy balance check to validate the energy calculation
        # This checks if the energy change is consistent with inputs and losses
        self.Echeck = -Scapacity * (self.SOC - SOC_old) + Qin - Qout - UA * (Ttrans - Tamb)


class TES_PCM_ideal:
    """
    High-level PCM TES model for HVAC-oriented studies.

    SI units:
    - Temperatures [C]
    - Energy [J]
    - Power [W]
    - Time [s]
    """

    def __init__(
        self,
        transition_temp_C,
        total_capacity_J,
        heat_transfer_effectiveness=0.9,
        soc_init=0.5,
        min_temp_C=-20.0,
        max_temp_C=80.0,
        latent_fraction=0.85,
        charge_efficiency=0.95,
        discharge_efficiency=0.95,
        standby_UA_W_per_K=0.0,
        ambient_temp_C=25.0,
        max_charge_power_W=None,
        max_discharge_power_W=None,
    ):
        if max_temp_C <= min_temp_C:
            raise ValueError("max_temp_C must be greater than min_temp_C.")
        if not (min_temp_C <= transition_temp_C <= max_temp_C):
            raise ValueError("transition_temp_C must be within [min_temp_C, max_temp_C].")
        if total_capacity_J <= 0:
            raise ValueError("total_capacity_J must be > 0.")
        if not (0 < heat_transfer_effectiveness <= 1):
            raise ValueError("heat_transfer_effectiveness must be in (0, 1].")
        if not (0 <= soc_init <= 1):
            raise ValueError("soc_init must be in [0, 1].")
        if not (0 <= latent_fraction <= 1):
            raise ValueError("latent_fraction must be in [0, 1].")
        if not (0 < charge_efficiency <= 1):
            raise ValueError("charge_efficiency must be in (0, 1].")
        if not (0 < discharge_efficiency <= 1):
            raise ValueError("discharge_efficiency must be in (0, 1].")
        if standby_UA_W_per_K < 0:
            raise ValueError("standby_UA_W_per_K must be >= 0.")
        if max_charge_power_W is not None and max_charge_power_W < 0:
            raise ValueError("max_charge_power_W must be >= 0 or None.")
        if max_discharge_power_W is not None and max_discharge_power_W < 0:
            raise ValueError("max_discharge_power_W must be >= 0 or None.")

        self.transition_temp_C = float(transition_temp_C)
        self.total_capacity_J = float(total_capacity_J)
        self.heat_transfer_effectiveness = float(heat_transfer_effectiveness)
        self.min_temp_C = float(min_temp_C)
        self.max_temp_C = float(max_temp_C)
        self.latent_fraction = float(latent_fraction)
        self.charge_efficiency = float(charge_efficiency)
        self.discharge_efficiency = float(discharge_efficiency)
        self.standby_UA_W_per_K = float(standby_UA_W_per_K)
        self.ambient_temp_C = float(ambient_temp_C)
        self.max_charge_power_W = max_charge_power_W
        self.max_discharge_power_W = max_discharge_power_W

        self.latent_capacity_J = self.total_capacity_J * self.latent_fraction
        self.sensible_capacity_total_J = self.total_capacity_J - self.latent_capacity_J
        self.sensible_capacity_low_J = 0.5 * self.sensible_capacity_total_J
        self.sensible_capacity_high_J = 0.5 * self.sensible_capacity_total_J

        self.energy_stored_J = soc_init * self.total_capacity_J
        self.soc = soc_init
        self.stored_capacity_J = self.energy_stored_J
        self.available_capacity_J = self.total_capacity_J - self.energy_stored_J
        self.phase_region = "solid"
        self.state_temperature_C = self.min_temp_C
        self.last_step = {}

        self._update_state_from_energy()

    def _state_temperature_from_energy(self, energy_J):
        e_low = self.sensible_capacity_low_J
        e_lat = self.latent_capacity_J
        t_low_span = max(self.transition_temp_C - self.min_temp_C, 1e-9)
        t_high_span = max(self.max_temp_C - self.transition_temp_C, 1e-9)

        if energy_J <= e_low:
            if e_low <= 0:
                return self.transition_temp_C, "phase_change"
            temp = self.min_temp_C + t_low_span * (energy_J / e_low)
            return temp, "solid"

        if energy_J <= e_low + e_lat:
            return self.transition_temp_C, "phase_change"

        if self.sensible_capacity_high_J <= 0:
            return self.transition_temp_C, "phase_change"

        e_high = energy_J - (e_low + e_lat)
        temp = self.transition_temp_C + t_high_span * (e_high / self.sensible_capacity_high_J)
        return min(temp, self.max_temp_C), "liquid"

    def _update_state_from_energy(self):
        self.energy_stored_J = max(0.0, min(self.energy_stored_J, self.total_capacity_J))
        self.soc = self.energy_stored_J / self.total_capacity_J
        self.stored_capacity_J = self.energy_stored_J
        self.available_capacity_J = self.total_capacity_J - self.energy_stored_J
        self.state_temperature_C, self.phase_region = self._state_temperature_from_energy(self.energy_stored_J)

    def step(self, q_in_J=0.0, q_out_J=0.0, dt_s=3600.0, ambient_temp_C=None):
        """
        Advance the TES state by one timestep.

        Parameters
        ----------
        q_in_J : float
            External thermal energy sent to TES during the timestep.
        q_out_J : float
            Useful thermal energy requested from TES during the timestep.
        dt_s : float
            Timestep duration in seconds.
        ambient_temp_C : float or None
            Optional ambient temperature override for this timestep.

        Returns
        -------
        dict
            Step results including SOC, stored energy, losses, and unmet load.
        """
        if dt_s <= 0:
            raise ValueError("dt_s must be > 0.")
        if q_in_J < 0 or q_out_J < 0:
            raise ValueError("q_in_J and q_out_J must be >= 0.")

        ambient = self.ambient_temp_C if ambient_temp_C is None else float(ambient_temp_C)

        q_in_effective = q_in_J * self.heat_transfer_effectiveness * self.charge_efficiency
        if self.max_charge_power_W is not None:
            q_in_effective = min(q_in_effective, self.max_charge_power_W * dt_s)

        q_out_from_storage = q_out_J / (self.heat_transfer_effectiveness * self.discharge_efficiency)
        if self.max_discharge_power_W is not None:
            q_out_from_storage = min(q_out_from_storage, self.max_discharge_power_W * dt_s)

        estimated_temp_C, _ = self._state_temperature_from_energy(self.energy_stored_J)
        q_standby_loss_J = self.standby_UA_W_per_K * (estimated_temp_C - ambient) * dt_s

        e_before = self.energy_stored_J
        e_after_raw = e_before + q_in_effective - q_out_from_storage - q_standby_loss_J
        e_after = max(0.0, min(e_after_raw, self.total_capacity_J))

        spill_charge_J = max(0.0, e_after_raw - self.total_capacity_J)
        unmet_discharge_J = max(0.0, -e_after_raw)

        self.energy_stored_J = e_after
        self._update_state_from_energy()

        self.last_step = {
            "dt_s": dt_s,
            "ambient_temp_C": ambient,
            "q_in_requested_J": q_in_J,
            "q_out_requested_J": q_out_J,
            "q_in_effective_J": q_in_effective,
            "q_out_from_storage_J": q_out_from_storage,
            "q_standby_loss_J": q_standby_loss_J,
            "spill_charge_J": spill_charge_J,
            "unmet_discharge_J": unmet_discharge_J,
            "energy_before_J": e_before,
            "energy_after_J": self.energy_stored_J,
            "soc": self.soc,
            "stored_capacity_J": self.stored_capacity_J,
            "available_capacity_J": self.available_capacity_J,
            "state_temperature_C": self.state_temperature_C,
            "phase_region": self.phase_region,
        }
        return self.last_step

    def state(self):
        """Return current TES state variables as a dictionary."""
        return {
            "soc": self.soc,
            "stored_capacity_J": self.stored_capacity_J,
            "available_capacity_J": self.available_capacity_J,
            "state_temperature_C": self.state_temperature_C,
            "phase_region": self.phase_region,
            "total_capacity_J": self.total_capacity_J,
            "latent_capacity_J": self.latent_capacity_J,
        }






if __name__ == "__main__":

    #=======================================================================================================
    # Example 001 - TESsimple class

    # pcm_params = {
    #             'Scapacity': 36e6,  # ~10 kWh thermal
    #             'UA': 2.2,          # W/K, representing heat loss to ambient
    #             'Ttrans': 24.0,     # degC, transition temperature    
    #             'Tamb': 23.0,       # degC, ambient temperature
    #             }
    
    # pcm = TESsimple(PCM=pcm_params, Qin=1.8e6, Qout=1.5e6, SOC_old=0.35)
    # print(f"SOC: {pcm.SOC:.3f}, Energy: {pcm.Energy/1e6:.3f} MJ, Energy check: {pcm.Echeck/1e6:.3f} MJ")



    #=======================================================================================================
    # Example 002 - TES_PCM_ideal class

    # Paraffin-like PCM parameters for HVAC-scale storage
    pcm = TES_PCM_ideal(
        transition_temp_C=24.0,
        total_capacity_J=36e6,            # ~10 kWh thermal
        heat_transfer_effectiveness=0.88,
        soc_init=0.35,
        min_temp_C=5.0,
        max_temp_C=40.0,
        latent_fraction=0.82,
        charge_efficiency=0.95,
        discharge_efficiency=0.94,
        standby_UA_W_per_K=2.2,
        ambient_temp_C=23.0,
        max_charge_power_W=2500.0,
        max_discharge_power_W=2200.0,
    )


    dt_s = 3600.0
    n_hours = 24
    hours = np.arange(n_hours + 1)

    soc_hist = [pcm.soc]
    energy_hist_MJ = [pcm.stored_capacity_J / 1e6]
    temp_hist_C = [pcm.state_temperature_C]
    phase_hist = [pcm.phase_region]

    for h in range(n_hours):
        hod = h % 24

        # Example HVAC operation:
        # charge in daytime (e.g., solar-assisted heat pump), discharge in evening/night
        if 10 <= hod <= 16:
            q_in_J = 1.8e6   # 0.5 kWh equivalent thermal per hour before efficiencies
        else:
            q_in_J = 0.0

        if 18 <= hod <= 23 or 0 <= hod <= 6:
            q_out_J = 1.5e6  # requested useful load from storage
        else:
            q_out_J = 0.3e6  # low background load

        ambient = 21.0 + 4.0 * np.sin((2.0 * np.pi / 24.0) * hod)

        step_out = pcm.step(q_in_J=q_in_J, q_out_J=q_out_J, dt_s=dt_s, ambient_temp_C=ambient)

        soc_hist.append(step_out["soc"])
        energy_hist_MJ.append(step_out["stored_capacity_J"] / 1e6)
        temp_hist_C.append(step_out["state_temperature_C"])
        phase_hist.append(step_out["phase_region"])


    print("Final TES state:")
    print(f"SOC: {soc_hist[-1]:.3f}")
    print(f"Stored capacity: {energy_hist_MJ[-1]:.3f} MJ")
    print(f"State temperature: {temp_hist_C[-1]:.2f} C")
    print(f"Phase region: {phase_hist[-1]}")

    fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    ax[0].plot(hours, soc_hist, color="tab:blue", lw=2)
    ax[0].set_ylabel("SOC [-]")
    ax[0].set_ylim(0.0, 1.05)
    ax[0].grid(True, alpha=0.3)

    ax[1].plot(hours, energy_hist_MJ, color="tab:orange", lw=2)
    ax[1].set_ylabel("Stored Energy [MJ]")
    ax[1].grid(True, alpha=0.3)

    ax[2].plot(hours, temp_hist_C, color="tab:green", lw=2)
    ax[2].axhline(24.0, color="k", ls="--", lw=1, label="Transition T")
    ax[2].set_ylabel("State Temp [C]")
    ax[2].set_xlabel("Time [h]")
    ax[2].grid(True, alpha=0.3)
    ax[2].legend(loc="best")

    fig.suptitle("TES_PCM_ideal Example (Paraffin-like PCM)")
    fig.tight_layout()
    # fig.savefig("ThermalStorage/pcm_example_profile.png", dpi=150)