"""
ASHP Energy Model

Author:         Conrado Ermel
Date:           01/04/2022
Last update:    05/04/2023
Description:
            - 3 zones model for the evaporator and condenser
            - Scipy minimize function
"""


from scipy import optimize
from CoolProp.CoolProp import PropsSI
import numpy as np
import time
from matplotlib import pyplot as plt
import math
from HX import condenser, evaporator
from tqdm import tqdm
from Diagrams import TSdiagSIMPLE, TSdiag, PHdiag



class HeatpumpAIR:
    """"Heat pump class:
        This Heat pump considers a 3 zone model for the condenser and the evaporator.
        It is suitable for air-air heat pumps.
        If water is selected as condenser fluid, the refrigerant must be changed to 'R410a'.
        
        Inputs:
            rpm:                rpm, compressor rotation
            eta_C_iso:          isoentropic efficiency
            Disp:               m3,displacement
            eta_C_vol:          volumetric efficiency
            m_dot_w:            kg/s, indoor fluid mass flow rate
            T_8:                K, indoor fluid temperature
            m_dot_air:          kg/s, outdoor fluid mass flow rate
            T_air_in:           K, outdoor fluid temperature
            Condfluid='water':  indoor fluid (water or air)
            fluid='R134a':      refrigerant fluid
            
        Methods:
            state_:             [Pressure, Temperature, Enthalpy, Quality]
            Q_cond:             W, heat transfered in the condenser
            T_5:                K, fluid temperature exiting condenser
            Q_evap:             W, heat transfered in the evaporator
            T_air_out:          K, fluid temperature exiting evaporator
            P_1:                Pa, pressure in the compressor inlet
            P_2:                Pa, pressure in the compressor outlet
            W_c:                W, compressor work
            R_c:                compression rate
            COP:                real Coefficient of Performance
            E_balance:          residual from calculation
            Capacity:           Btu/h, refrigeration capacity            
            """
    
    def __init__(self, SysParams, m_dot_w, T_8, m_dot_air, T_air_in, Condfluid, fluid):
        self.rpm =          SysParams['Comp: rpm']
        self.eta_C_iso =    SysParams['Comp: eta_C_iso']
        self.Disp =         SysParams['Comp: Disp']
        self.eta_C_vol =    SysParams['Comp: eta_C_vol']
        self.eta_C_mec =    SysParams['Comp: eta_C_mec']
        
        self.superheating=  SysParams['Sys: Superheat']
        self.subcooling=    SysParams['Sys: Subcool']
        
        self.W_FanCOND =    SysParams['Cond: Fan']
        self.W_FanEVAP =    SysParams['Evap: Fan']
        
        self.m_dot_w =      m_dot_w
        self.T_8 =          T_8
        self.m_dot_air =    m_dot_air
        self.T_air_in =     T_air_in
        self.fluid =        fluid
        self.Condfluid =    Condfluid
        
        dec=                10          # decimal for some PropsSI functions
        
        self.CD_HTC_air =   100         # W/m2K, heat transfer coefficient of Condenser: air side
        self.EV_HTC_air =   100         # W/m2K, heat transfer coefficient of Evaporator: air side
        
        
        def march(Pguess):
            if Pguess[0] < 0 or Pguess[1] < 0:  
                self.flag1 = True                                               # Flag True if the pressure guesses are negative 
                return (0,0)
       
            else:
                self.C_min_cond=    PropsSI('C', 'P', 101325, 'T', T_8, self.Condfluid) * m_dot_w
                self.C_min_evap=    PropsSI('C', 'P', 101325, 'T', T_air_in, 'air') * m_dot_air    
                
                #---------------------------------------------------------
                # Point 1
                self.T_1 =          round(PropsSI('T', 'P', Pguess[0], 'Q', 1, fluid) + self.superheating,   dec)
                self.h_1 =          PropsSI('H', 'P', Pguess[0], 'T', self.T_1, fluid)
                self.x_1 =          PropsSI('Q', 'P', Pguess[0], 'T', self.T_1, fluid)
                self.s_1 =          PropsSI('S', 'P', Pguess[0], 'T', self.T_1, fluid)
                rho_1 =             PropsSI('D', 'P', Pguess[0], 'T', self.T_1, fluid)
                
                self.m_dot_r=       rho_1 * self.Disp * (self.rpm/60) * self.eta_C_vol
                            
                self.state_1=       [Pguess[0], self.T_1, self.h_1, self.x_1]
                #---------------------------------------------------------
                # Point 2
                self.s_2 =          self.s_1
                self.h_2s =         PropsSI('H', 'P', Pguess[1], 'S', self.s_2, fluid)
                self.h_2 =          (self.h_2s - self.h_1)/self.eta_C_iso + self.h_1
                self.T_2 =          round(PropsSI('T', 'P', Pguess[1], 'H', self.h_2, fluid),   dec)
                
                #---------------------------------------------------------
                # Point 3
                self.T_3 =          round(PropsSI('T', 'P', Pguess[1], 'Q', 0, fluid) - self.subcooling,    dec) #8.4
                self.h_3=                PropsSI('H', 'P', Pguess[1], 'T', self.T_3, fluid)
                
                #Condenser 3 zones
                self.CD=            condenser(self.fluid, self.T_8, self.m_dot_r, self.m_dot_w,
                                              round(Pguess[1], 2),round(self.T_2, 2), self.Condfluid, self.subcooling, self.CD_HTC_air)
                self.Q_cond =       self.CD.Q
                h_3calc =           self.Q_cond/self.m_dot_r + self.h_2
               
                self.T_5 =          -self.Q_cond/self.C_min_cond + T_8
                self.Taircomparison =      self.CD.T_out
                
                #---------------------------------------------------------
                # Point 4
                self.h_4 =          self.h_3
                self.T_4 =          round(PropsSI('T', 'P', Pguess[0], 'H', self.h_4, fluid),   dec)
                self.x_4 =          PropsSI('Q', 'P', Pguess[0], 'H', self.h_4, fluid)
                
                #Evaporator 2 zones
                self.EV=            evaporator(self.fluid, self.T_air_in, self.m_dot_r, self.m_dot_air,
                                                round(Pguess[0], 2), self.h_4, self.superheating, SysParams, self.EV_HTC_air)
                
                self.Q_evap =       self.EV.Q
                h_1calc =           self.Q_evap/self.m_dot_r + self.h_4
                self.T_air_out =    -self.Q_evap/self.C_min_evap + T_air_in
                #---------------------------------------------------------
                
               
                errP1=              (self.h_1 - h_1calc)
                errP2=              (self.h_3 - h_3calc)                
                return (errP1, errP2)
    
    
        #---------------------------------------------------------
        #Initial guess considering the saturation temperatures with superheating and subcooling
        self.Pguess=        [int(PropsSI('P', 'Q', 1, 'T', self.T_air_in    -self.subcooling, self.fluid)),
                             int(PropsSI('P', 'Q', 1, 'T', self.T_8         +self.superheating, self.fluid))]

        
        
        self.flag1 = False                                                  # error flag when guess is P<0
        result = optimize.root(march, self.Pguess, method='hybr')           # Minimization function
        self.P_1=           result.x[0]
        self.P_2=           result.x[1]
        
        
        # Post Processing
        # ---------------------------------------------------------
        def postproc():
            W_c=           self.m_dot_r * (self.h_2 - self.h_1)
            R_c=           self.P_2/self.P_1
            COP =          -self.Q_cond / ((W_c / self.eta_C_mec) + self.W_FanCOND + self.W_FanEVAP)
            E_balance =    self.Q_cond + self.Q_evap + W_c
            Capacity =     -self.Q_cond*3.41
            
            return {'W_c': W_c, 'R_c':R_c, 'COP':COP, 'E_balance':E_balance, 'Capacity':Capacity}


        # Linear interpolation to predict values in singularity points
        # ---------------------------------------------------------
        if self.flag1 == False:
            a = postproc()
            self.W_c =          a['W_c']
            self.R_c =          a['R_c']
            self.COP =          a['COP']
            self.E_balance =    a['E_balance']
            self.Capacity =     a['Capacity']

        else:
            dif = 1                                                     # value away from the singularity to perform the interpolation
            # Simulation with Tamb - 1 K
            self.Pguess=        [int(PropsSI('P', 'Q', 1, 'T', self.T_air_in -dif   -self.subcooling, self.fluid)),
                                  int(PropsSI('P', 'Q', 1, 'T', self.T_8            +self.superheating, self.fluid))]
            optimize.root(march, self.Pguess, method='hybr')           # Minimization function
            b = postproc()
            
            # Simulation with Tamb + 1 K
            self.Pguess=        [int(PropsSI('P', 'Q', 1, 'T', self.T_air_in +dif   -self.subcooling, self.fluid)),
                                  int(PropsSI('P', 'Q', 1, 'T', self.T_8            +self.superheating, self.fluid))]
            optimize.root(march, self.Pguess, method='hybr')           # Minimization function
            c = postproc()

            
            self.W_c =          (b['W_c']+c['W_c'])/2
            self.R_c =          (b['R_c']+c['R_c'])/2
            self.COP =          (b['COP']+c['COP'])/2
            self.E_balance =    (b['E_balance']+c['E_balance'])/2
            self.Capacity =     (b['Capacity']+c['Capacity'])/2

                           
        
        
        
class HeatpumpWATER:
    """"Heat pump class:
        This Heat pump considers a 3 zone model for the condenser and the evaporator.
        It is suitable for air-water heat pumps.
        
        Inputs:
            rpm:                rpm, compressor rotation
            eta_C_iso:          isoentropic efficiency
            Disp:               m3,displacement
            eta_C_vol:          volumetric efficiency
            m_dot_w:            kg/s, indoor fluid mass flow rate
            T_8:                K, indoor fluid temperature
            m_dot_air:          kg/s, outdoor fluid mass flow rate
            T_air_in:           K, outdoor fluid temperature
            Condfluid='water':  indoor fluid (water or air)
            fluid='R134a':      refrigerant fluid
            
        Methods:
            state_:             [Pressure, Temperature, Enthalpy, Quality]
            Q_cond:             W, heat transfered in the condenser
            T_5:                K, fluid temperature exiting condenser
            Q_evap:             W, heat transfered in the evaporator
            T_air_out:          K, fluid temperature exiting evaporator
            P_1:                Pa, pressure in the compressor inlet
            P_2:                Pa, pressure in the compressor outlet
            W_c:                W, compressor work
            R_c:                compression rate
            COP:                real Coefficient of Performance
            E_balance:          residual from calculation
            Capacity:           Btu/h, refrigeration capacity            
            """
    
    def __init__(self, SysParams, m_dot_w, T_8, m_dot_air, T_air_in, Condfluid, fluid):
        self.rpm =          SysParams['Comp: rpm']
        self.eta_C_iso =    SysParams['Comp: eta_C_iso']
        self.Disp =         SysParams['Comp: Disp']
        self.eta_C_vol =    SysParams['Comp: eta_C_vol']
        self.eta_C_mec =    SysParams['Comp: eta_C_mec']
        
        self.superheating=  SysParams['Sys: Superheat']
        self.subcooling=    SysParams['Sys: Subcool']
        
        self.W_FanCOND =    SysParams['Cond: Fan']
        self.W_FanEVAP =    SysParams['Evap: Fan']
        
        self.m_dot_w =      m_dot_w
        self.T_8 =          T_8
        self.m_dot_air =    m_dot_air
        self.T_air_in =     T_air_in
        self.fluid =        fluid
        self.Condfluid =    Condfluid

        dec=                10          # decimal for some PropsSI functions
        
        self.EV_HTC_air =   100         # W/m2K, heat transfer coefficient of Evaporator: air side
        
        
        def march(Pguess):
            if Pguess[0] < 0 or Pguess[1] < 0:  
                self.flag1 = True                                               # Flag True if the pressure guesses are negative 
                return (0,0)
       
            else:
                self.C_min_cond=    PropsSI('C', 'P', 101325, 'T', T_8, self.Condfluid) * m_dot_w
                self.C_min_evap=    PropsSI('C', 'P', 101325, 'T', T_air_in, 'air') * m_dot_air    
                
                #---------------------------------------------------------
                # Point 1
                self.T_1 =          round(PropsSI('T', 'P', Pguess[0], 'Q', 1, fluid) + self.superheating,   dec)
                self.h_1 =          PropsSI('H', 'P', Pguess[0], 'T', self.T_1, fluid)
                self.x_1 =          PropsSI('Q', 'P', Pguess[0], 'T', self.T_1, fluid)
                self.s_1 =          PropsSI('S', 'P', Pguess[0], 'T', self.T_1, fluid)
                rho_1 =             PropsSI('D', 'P', Pguess[0], 'T', self.T_1, fluid)
                
                self.m_dot_r=       rho_1 * self.Disp * (self.rpm/60) * self.eta_C_vol
                            
                self.state_1=       [Pguess[0], self.T_1, self.h_1, self.x_1]
                #---------------------------------------------------------
                # Point 2
                self.s_2 =          self.s_1
                self.h_2s =         PropsSI('H', 'P', Pguess[1], 'S', self.s_2, fluid)
                self.h_2 =          (self.h_2s - self.h_1)/self.eta_C_iso + self.h_1
                self.T_2 =          round(PropsSI('T', 'P', Pguess[1], 'H', self.h_2, fluid),   dec)
                
                #---------------------------------------------------------
                # Point 3
                self.T_3 =          round(PropsSI('T', 'P', Pguess[1], 'Q', 0, fluid) - self.subcooling,    dec) #8.4
                self.h_3=                PropsSI('H', 'P', Pguess[1], 'T', self.T_3, fluid)
                
                # Condenser fixed effectiveness
                self.Q_cond =       0.7* self.C_min_cond * (T_8 - self.T_2)
                
                h_3calc =           self.Q_cond/self.m_dot_r + self.h_2
                self.T_5 =          -self.Q_cond/self.C_min_cond + T_8
                
                #---------------------------------------------------------
                # Point 4
                self.h_4 =          self.h_3
                self.T_4 =          round(PropsSI('T', 'P', Pguess[0], 'H', self.h_4, fluid),   dec)
                self.x_4 =          PropsSI('Q', 'P', Pguess[0], 'H', self.h_4, fluid)
                
                #Evaporator 2 zones
                self.EV=            evaporator(self.fluid, self.T_air_in, self.m_dot_r, self.m_dot_air,
                                                round(Pguess[0], 2), self.h_4, self.superheating, SysParams, self.EV_HTC_air)
                
                self.Q_evap =       self.EV.Q
                h_1calc =           self.Q_evap/self.m_dot_r + self.h_4
                self.T_air_out =    -self.Q_evap/self.C_min_evap + T_air_in
                #---------------------------------------------------------
                
               
                errP1=              (self.h_1 - h_1calc)
                errP2=              (self.h_3 - h_3calc)                
                return (errP1, errP2)
    
    
        #---------------------------------------------------------
        #Initial guess considering the saturation temperatures with superheating and subcooling
        self.Pguess=        [int(PropsSI('P', 'Q', 1, 'T', self.T_air_in    -3*self.subcooling, self.fluid)),
                              int(PropsSI('P', 'Q', 1, 'T', self.T_8         +0.1*self.superheating, self.fluid))]

        
        
        self.flag1 = False                                                  # error flag when guess is P<0
        result = optimize.root(march, self.Pguess, method='hybr')           # Minimization function
        self.P_1=           result.x[0]
        self.P_2=           result.x[1]
        
        
        # Post Processing
        # ---------------------------------------------------------
        def postproc():
            W_c=            self.m_dot_r * (self.h_2 - self.h_1)
            W_cTOT=         self.m_dot_r * (self.h_2 - self.h_1) + self.W_FanCOND + self.W_FanEVAP
            R_c=            self.P_2/self.P_1
            COP =           -self.Q_cond / ((W_c / self.eta_C_mec) + self.W_FanCOND + self.W_FanEVAP)
            E_balance =     self.Q_cond + self.Q_evap + W_c
            Capacity =      -self.Q_cond*3.41
            
            return {'W_c': W_c, 'W_cTOT': W_cTOT, 'R_c':R_c, 'COP':COP, 'E_balance':E_balance, 'Capacity':Capacity}


        # Linear interpolation to predict values in singularity points
        # ---------------------------------------------------------
        if self.flag1 == False:
            a = postproc()
            self.W_c =          a['W_c']
            self.W_cTOT =       a['W_cTOT']
            self.R_c =          a['R_c']
            self.COP =          a['COP']
            self.E_balance =    a['E_balance']
            self.Capacity =     a['Capacity']

        else:
            dif = 1                                                     # value away from the singularity to perform the interpolation
            # Simulation with Tamb - 1 K
            self.Pguess=        [int(PropsSI('P', 'Q', 1, 'T', self.T_air_in -dif   -3*self.subcooling, self.fluid)),
                                  int(PropsSI('P', 'Q', 1, 'T', self.T_8            +self.superheating, self.fluid))]
            optimize.root(march, self.Pguess, method='hybr')           # Minimization function
            b = postproc()
            
            # Simulation with Tamb + 1 K
            self.Pguess=        [int(PropsSI('P', 'Q', 1, 'T', self.T_air_in +dif   -3*self.subcooling, self.fluid)),
                                  int(PropsSI('P', 'Q', 1, 'T', self.T_8            +self.superheating, self.fluid))]
            optimize.root(march, self.Pguess, method='hybr')           # Minimization function
            c = postproc()

            
            self.W_c =          (b['W_c']+c['W_c'])/2
            self.W_cTOT =       (b['W_c']+c['W_c'])/2 + self.W_FanCOND + self.W_FanEVAP
            self.R_c =          (b['R_c']+c['R_c'])/2
            self.COP =          (b['COP']+c['COP'])/2
            self.E_balance =    (b['E_balance']+c['E_balance'])/2
            self.Capacity =     (b['Capacity']+c['Capacity'])/2
            


