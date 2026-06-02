"""
OPEN-SOURCE SIMULATION PLATFORM FOR AIR SOURCE HEAT PUMP INTEGRATED WITH THERMAL ENERGY STORAGE

authors: Conrado Ermel, Marcus V.A. Bianchi & Paulo S. Schneider

To cite this article: Conrado Ermel, Marcus V.A. Bianchi & Paulo S. Schneider (04 Jul
2024): Open-source simulation platform for air source heat pump integrated with
thermal energy storage, Science and Technology for the Built Environment, 
DOI:10.1080/23744731.2024.2366134

To link to this article: https://doi.org/10.1080/23744731.2024.2366134


=== THERMAL LOAD ===
    This class icalculates the envelope thermal load of a building given the outdoor conditions. 
    The user may add internal loads if required.
    Please refer to the article for detailed description.

"""




import numpy as np
import math
from scipy.optimize import fsolve



class TL_5R1C:
    """"
        SP must contain the building parameters:
            'Af':       Floor area, [m2]
            'Am':       Effective mass area - for a light class building, [m2]
            'Atot':     Whole area of all the surfaces which face the air-conditioned environment, [m2]
            'Vbuild':   Building volume, [m3]
            'Cm':       Thermal capacity of the envelope, [J/k]
            'his':      Thermal exchange coefficient, [W/m2K]
            'Hve':      Thermal coeff. ventilation, [W/K]
            'Htris':    Thermal coeff. indoor air, [W/K]
            'Htrw':     Thermal coeff. windowed area, [W/K]
            'Htrms':    Thermal coeff. opaque, [W/K]
            'Htrem':    Thermal coeff. opaque, [W/K]
                    
        TL calculates the energy balance for the 5R1C model:
            inputs:
                - Building mass temperature [C]
                - Room temperature setpoint [C]
                - Outdoor temperature [C]
                - Air supply temperature [C]
                - Internal heat gain (people, appliances, etc.) [W]
                - Solar radiation hourly average [W]
            Outputs:
                - Thermal load
                - Building mass temperature [C]
                - Building mass heat exchanged [C]
                - Walls surface temperature [C]
    """
    def __init__(self, SP):
        self.SP = SP
   
    def TL(self,Tmold, Ti, Te, Tsup, PHIint, PHIsol):
        # Tgr = 16
        # Ti = Ti - Tgr
        # Te = Te + Tgr
        # Tsup = Tsup + Tgr
        
        
        Htr1 =      1/((1/self.SP['Hve']) + (1/self.SP['Htris']))
        Htr2 =      Htr1 + self.SP['Htrw']
        Htr3 =      1/((1/Htr2) + (1/self.SP['Htrms']))
        PHIia =     0.5*PHIint
        PHIm =      (self.SP['Am']/self.SP['Atot'])*(0.5*PHIint + PHIsol)  
        PHIst =     (1 - (self.SP['Am']/self.SP['Atot']) - (self.SP['Htrw']/(9.1*self.SP['Atot']))) * (0.5*PHIint + PHIsol)
 
        def RCload(p):
            """Solve the xRyC energy balance"""
            PHIhc, Tmt, Tm, PHImtot, Ts = p
            return((Tmold * ((self.SP['Cm']/3600)-0.5*(Htr3 + self.SP['Htrem'])) + PHImtot) / ((self.SP['Cm']/3600) + 0.5*(Htr3 + self.SP['Htrem'])) - (Tmt ),
                    (Tmt  + Tmold)/2 - (Tm ),
                    PHIm + self.SP['Htrem']*Te + (Htr3*(PHIst + self.SP['Htrw']*Te + Htr1*(((PHIia + PHIhc)/self.SP['Hve'])) + Tsup))/(Htr2) - PHImtot,
                    ((self.SP['Htrms']*(Tm ) + PHIst + self.SP['Htrw']*Te) + Htr1*(Tsup + (PHIia + PHIhc)/self.SP['Hve']))/(self.SP['Htrms'] + self.SP['Htrw'] + Htr1) - (Ts ),
                    (self.SP['Htris']*(Ts ) + self.SP['Hve']*Tsup + PHIia + PHIhc)/(self.SP['Htris'] + self.SP['Hve']) - Ti)

        PHIhc, Tmt, Tm, PHImtot, Ts= fsolve(RCload, [1,1,1,1,1])
        return PHIhc, Tmt, Tm, PHImtot, Ts
        



          
    
class TL:
    """"""
    
    def __init__(self, Tamb, SP, V_inf, timestep):
        self.Tamb =     Tamb
        self.SP =       SP
        self.signal=    False
        self.VwindEx=   V_inf #1 #m/s
        self.TS =       timestep
        
        
        if self.Tamb < self.SP:
            self.signal=True
            
            Wh=             6
            Lh=             8
            Hh=             2.8
            Aceiling =      Wh * Lh                     # Ceiling area [m2]
            Awall =         2*(Hh*Wh) + 2*(Hh*Lh)       # walls area, no windows considered [m2]
            U_roof =        0.18                        # U: ceiling.   Chapter 29 - Table 23
            U_wall =        0.51                        # U: walls.     Chapter 29 - Table 23
            Al=             Awall * 2.8                 # Average leakeage area. Chapter 20, Eq.9 - Table 3
            alphac=         0.92                        # Alpha ceiling asphalt shingles (ASHRAE 2005)
            
            """
            #Opaque surfaces
            CF_cei = U_roof * (1*(Tamb - self.SP) + (39*alphac - 6.8) )
            self.Q_ceiling= Aceiling * CF_cei

            CF_wall = U_wall * (1*(Tamb - self.SP) + 7.9)
            self.Q_wall = Awall * CF_wall
            
            #infiltration:     
            V_inf = (Al/1e3) * math.sqrt(0.000145*(self.SP - Tamb) + 0.000246*self.VwindEx**2)
            self.Q_inf =  V_inf * 1.2 * 1006 *(Tamb - self.SP)
            
            self.Qloadold = ((Tamb - self.SP)*364.)/5
            self.Qload = self.Q_ceiling + self.Q_wall + self.Q_inf
            """
            
            #Ceiling
            self.Q_ceiling = U_roof * Aceiling * (Tamb - self.SP)
            #Walls
            self.Q_wall = U_wall * Awall * (Tamb - self.SP)
            #infiltration:     
            V_inf = (Al/1e3) * math.sqrt(0.000145*(self.SP - Tamb) + 0.000246*self.VwindEx**2)
            self.Q_inf =  V_inf * 1.2 * 1006 *(Tamb - self.SP)
            
            self.Qloadold = ((Tamb - self.SP)*364.)/5
            self.Qload = (self.Q_ceiling + self.Q_wall + self.Q_inf)*(self.TS/3600)
            

        else:
            self.signal=    False
            self.Qloadold = 0
            self.Qload =    0
            
            