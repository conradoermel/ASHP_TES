"""
OPEN-SOURCE SIMULATION PLATFORM FOR AIR SOURCE HEAT PUMP INTEGRATED WITH THERMAL ENERGY STORAGE

authors: Conrado Ermel, Marcus V.A. Bianchi & Paulo S. Schneider

To cite this article: Conrado Ermel, Marcus V.A. Bianchi & Paulo S. Schneider (04 Jul
2024): Open-source simulation platform for air source heat pump integrated with
thermal energy storage, Science and Technology for the Built Environment, 
DOI:10.1080/23744731.2024.2366134

To link to this article: https://doi.org/10.1080/23744731.2024.2366134


=== MAIN PROGRAM ===

    Use this main code for calculating the HP performance in an hourly basis for one year round. 
    This code calls the other classes available in the library.
    Please refer to the article for detailed description.
"""

# importing packages
#------------------------------------------------------------------------------
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from statistics import mean
from Climateimport import copernicus2
from ASHP_Performance import HeatpumpAIR
from ThermalLoad import TL_5R1C





#####################################################################################################################
#####################################################################################################################
# Simulation Parameters
#####################################################################################################################
#####################################################################################################################

N   =           8760                        # Length of x axis (time series)
SP  =           19                          # Indoors temperature set point

season =        {'HS1_init':    0,          # Heating season 1 start                                            0
                  'HS1_end':    100,       # Heating season 1 end                                              2880
                  'HS2_init':   8600,       # Heating season 2 start    (if south hemisphere copy 'HS1_init')   7400
                  'HS2_end':    8000}       # Heating season 2 end      (if south hemisphere copy 'HS2_end')    8760


SP =            273+19                      # Indoor temperature [K]
timestep=       3600                        # Simulation time step [s]
TinCondenser =  25 + 273                    # Indoor temperature (used to determine HP performance)

#####################################################################################################################
#####################################################################################################################






# Climate data importing
#------------------------------------------------------------------------------

fp = "DENVER_2022.nc"
 
a=          copernicus2(fp)
TairDB =    a.read('t2m')['t2m']            # Dry bulb temperature [K]
Tgr =       a.read('stl4')['stl4']          # Soil temperature (4m deep) [K]
Wind =      a.read('Wind')['Wind']          # Wind speed [m/s]
 



# ASHP    
#------------------------------------------------------------------------------

#Compressor, Evaporator, and Condenser
Comp =          {'Comp: rpm': 3600,'Comp: eta_C_iso':0.63, 'Comp: Disp':3e-5, 'Comp: eta_C_vol':0.95, 'Comp: eta_C_mec':0.88}
Evap =          {'Evap: TubeL':10, 'Evap: TubeN':5, "Evap: TubeID": 0.011, 'Evap: FinEff': 0.75, 'Evap: FinRatio': 7.2, 'Evap: Fan':120}
Cond =          {'Cond: TubeL':10, 'Cond: TubeN':5, "Cond: TubeID": 0.011, 'Cond: FinEff': 0.75, 'Cond: FinRatio': 7.2, 'Cond: Fan':300}
System =        {'Sys: Superheat': 8.5, 'Sys: Subcool': 6.5}
SysParams =     dict(Comp, **Evap,**Cond, **System)   # add the dicts together

#Condenser
m_dot_w_b =     0.1
m_dot_w_c =     0.1

#Evaporator
m_dot_air =     1.18     # 1.18/PropsSI('D', 'P', 101325, 'T', 281, 'air')




#DWELLING PARAMETERS (Thermal load 5R1C)
#------------------------------------------------------------------------------

Dwelling =      {'Af':      6*8,            # Floor area, [m2]
                  'Am':     139.91,         # Effective mass area - for a light class building, [m2]
                  'Atot':   216,            # whole area of all the surfaces which face the air-conditioned environment, [m2]
                  'Vbuild': 129.6,          # Building volume, [m3]
                  'Cm':     3.00e6}         # Thermal capacity of the envelope, [J/k]
Coefficients=   {'his':     3.45,           # thermal exchange coefficient, [W/m2K]
                  'Hve':    20.86,          # Thermal coeff. ventilation, [W/K]
                  'Htris':  500,            # Thermal coeff. indoor air, [W/K]
                  'Htrw':   50,             # Thermal coeff. windowed area, [W/K]
                  'Htrms':  1200,           # Thermal coeff. opaque, [W/K]
                  'Htrem':  60.89}          # Thermal coeff. opaque, [W/K]
RESParams =     dict(Dwelling, **Coefficients)   # add the dicts together

A =             TL_5R1C(RESParams)
Ti_sp =         SP-273





#Creating Arrays
#-------------------------------------------------------------------------------
arraylist = []
# heat pump
arraylist.extend(['W_c','W_hp','COP','Qload','Qcond','Qevap','Energy'])
# Thermal Load
arraylist.extend(['Tm', 'Qload5R1C'])

for x in arraylist:
    globals()[x] = np.empty(N)
    globals()[x][:] = np.nan
    





#####################################################################################################################
# Thermal load calculation
#####################################################################################################################

for i in tqdm((range(N))):
    if (i >= season['HS1_init'] and i <= season['HS1_end']) or (i >= season['HS2_init'] and i <= season['HS2_end']): 

        # Thermal load - 5R1C:
        if i==season['HS1_init'] or i==season['HS2_init']:
            Tmold = 19
        else:
            Tmold = Tm[i-1]
            
        B=A.TL(Tmold,Ti_sp,TairDB[i]-273 + Tgr[i]-273,TairDB[i]-273,0,0)
        Qload5R1C[i] =  B[0]
        Tm[i] =         B[1]

for i in range(N):
    if Qload5R1C[i] > 0:
        Qload[i] = Qload5R1C[i]
print('Thermal load:    ok')




#####################################################################################################################
# Heat pump performance calculation
#####################################################################################################################

for i in tqdm((range(N))):
    HP = HeatpumpAIR(SysParams, m_dot_w_b, TinCondenser, m_dot_air, TairDB[i], 'air', 'R134a')
    COP[i] = HP.COP
    W_c[i] = HP.W_c
    Qcond[i] = HP.Q_cond





#####################################################################################################################
# POST PROCESSING
#####################################################################################################################

# Print outputs
#------------------------------------------------------------------------------
def outsprint():
    linelength = 60
    print('')
    print(linelength*'-')
    print('Weather info:')
    print('Temp. avg.:                  {:.2f} [K]'.format(mean(TairDB)))
    print('Temp. amplitude:             {:.2f} to {:.2f} [K]'.format(min(TairDB), max(TairDB)))
    
    print(linelength*'-')
    print('System info:')
    # print('Storage capacity:            {} [kWh]'.format(PCM['Scapacity']/1e3))
    # print('Control logic:               {}'.format(goal))
outsprint()
    


x_array =       np.arange(1,N+1,1)


# Thermal load Year results
#------------------------------------------------------------------------------
def plotTL(l):
    x_array = np.arange(0,len(TairDB),1)
    
    font = {'size'   : 12}
    plt.rc('font', **font)
    fig, axs = plt.subplots(2,1, sharex=True, dpi=400,figsize=(10,6))
    
    axs[0].plot(x_array, Qload/1e3, c='olive', label='5R1C', alpha=0.8)#, linewidth=0.8)
    axs[0].tick_params(axis="x", direction="in", length=0.5*font['size'], width=0.1*font['size'], top=True)
    axs[0].set_ylabel("Thermal load [kW]")
    axs[0].legend()
    axs[0].set_xlim(l)
    axs[0].xaxis.set_ticks(np.arange(0, 8760, 730))
    # font = {'size'   : 8}
    
    axs[1].plot(x_array, TairDB, c='gray', label='TairDB', alpha=0.8, linewidth=0.8)
    axs[1].tick_params(axis="x", direction="in", length=0.5*font['size'], width=0.1*font['size'], top=True)
    axs[1].set_ylabel("Temperature [K]")
    axs[1].legend()
    axs[1].set_xlim(l)
    axs[1].xaxis.set_ticks(np.arange(0, 8760, 730))
    font = {'size'   : 8}
    axs[1].xaxis.set_major_formatter(mticker.FuncFormatter(update_ticks))
       
    fig.set_tight_layout(True)
    # plt.savefig('TLcomparison.pdf', transparent=True)

plotTL([0,8760])
    


# Heat pump performance - year results
#------------------------------------------------------------------------------
def plotENERGY(l,file):
    font = {'size'   : 16}
    plt.rc('font', **font)

    fig = plt.figure(figsize=(12, 6), dpi=400)
    axs1 = plt.subplot2grid((5, 6), (0, 0), colspan=5, rowspan=2)
    axs2 = plt.subplot2grid((5, 6), (2, 0), colspan=5, rowspan=6)

    # First axis
    axs1.plot(x_array, COP,     c='gray',   label='$COP$', alpha=0.9,linewidth=2)
    axs1.tick_params(axis="x", direction="in", length=0.5*font['size'], width=0.1*font['size'], top=True)
    axs1.set_ylabel("[COP]")
    axs1.set_xlim(l)
    axs1.set_axisbelow(True)
    axs1.xaxis.set_ticklabels([])
    
    axs2.plot(x_array, Qcond,     c='red',   label='Capacity', alpha=0.9,linewidth=2)
    axs2.tick_params(axis="x", direction="in", length=0.5*font['size'], width=0.1*font['size'], top=True)
    axs2.grid(color='gray', linestyle='--', linewidth=0.2)
    axs2.set_ylabel("Thermal power [W]")
    axs2.set_xlabel("Hour in the year")
    axs2.legend(loc=2)
    axs2.set_xlim(l)

    plt.tight_layout()
    
plotENERGY([0,8760],'HP_performance')



