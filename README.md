# Simulation platform for air source heat pump integrated with thermal energy storage

This is an open-source library. Users are welcome to use, reproduce, and improve the code.
For a detailed explanation of the physics considered, please refer to the article:

**Open-source simulation platform for air source heat pump integrated with thermal energy storage,
Science and Technology for the Built Environment**, DOI:10.1080/23744731.2024.2366134

Link the article: https://doi.org/10.1080/23744731.2024.2366134

- _Main_program_: Allows for hourly simulation of ASHP-TES systems over an entire year.
- _ASHP_Performance_: Class that calculates the performance of an air-source heat pump
- _HX_: Class to model the evaporator and the condenser. This class is called by the ASHP_Performance
- _TES_: Class to model a simple TES based on energy balance calculations.
- _ThermalLoad_: Class that calculates the envelope heating load of a given building according to the input characteristics.
- _Climateimport_: Class used to import year-round weather data files from databases such as Copernicus or Energy Plus.

## Installation
See [Setup.md](Setup.md) for installation (`environment.yml` / `requirements.txt`) and how to run the simulation.
