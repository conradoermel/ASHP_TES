"""
OPEN-SOURCE SIMULATION PLATFORM FOR AIR SOURCE HEAT PUMP INTEGRATED WITH THERMAL ENERGY STORAGE

authors: Conrado Ermel, Marcus V.A. Bianchi & Paulo S. Schneider

To cite this article: Conrado Ermel, Marcus V.A. Bianchi & Paulo S. Schneider (04 Jul
2024): Open-source simulation platform for air source heat pump integrated with
thermal energy storage, Science and Technology for the Built Environment, 
DOI:10.1080/23744731.2024.2366134

To link to this article: https://doi.org/10.1080/23744731.2024.2366134


=== CONDENSER AND EVAPORATOR MODEL ===
    This class is called by the ASHP_Performance file.
    Please refer to the article for detailed description.

"""


from CoolProp.CoolProp import PropsSI
import math
import numpy as np
from matplotlib import pyplot as plt

    

class condenser:
    """
    Class of condenser.
    Inputs:
        refName:    Refrigerant fluid
        T_air:      K,      Outdoor (ambient) air temperature
        m_dot:      kg/s,   refrigerant flowrate
        m_dot_air:  kg/s,   air side flowrate
        P_in:       Pq,     fluid pressure in the inlet
        T_in:       K,      fluid temperature in the inlet
        subcooling  K,      Degree of subcooling exiting the condenser
    """
    
    def __init__(self, refName, T_air, m_dot, m_dot_air,P_in,T_in,Condfluid,subcooling, HTC_air):
        
        self.T_air=         T_air           # K, inlet air temperature              273
        self.m_dot=         m_dot           # kg/s, refrigerant mass flow rate      0.02
        self.m_dot_air=     m_dot_air       # kg/s, air mass flow rate              0.4
        self.refName =      refName
        
        self.P_in =         P_in            # Pa, inlet pressure
        self.T_in =         T_in            # K, inlet temperature
        self.Condfluid  =   Condfluid       # air or water
        self.subcooling=    -subcooling     # -8.4      # K, level of refrigerant subcooling 8.4
        HTC_air =           HTC_air         # W/m2K, heat transfer coefficient: refrigerant liquid phase
        
        
        h_in=               PropsSI('H', 'P', self.P_in, 'T', T_in, self.refName)
        rho_in=             PropsSI('D', 'P', self.P_in, 'T', T_in, self.refName)
        
        L_tube =    10      # m, tube length
        Nparallel=  5       # number of tubes
        ID=         0.01    # m, tube inner diameter    0.01
        eta_fin=    0.8     # fin efficiency
        finRatio=   7.2     # fin ratio
        
        
        HTC_liq=    600     # W/m2K, heat transfer coefficient: refrigerant liquid phase
        HTC_2ph=    3000    # W/m2K, heat transfer coefficient: refrigerant two phases
        HTC_vap=    600     # W/m2K, heat transfer coefficient: refrigerant vapor phase
    
      
        #------------------------------------------------------------------------------
        mfr_channel =   self.m_dot/Nparallel
            
        #obtaining U for each zone
        U_vap =         1/(1/(HTC_vap) + 1/(HTC_air*eta_fin*finRatio))
        U_2ph =         1/(1/(HTC_2ph) + 1/(HTC_air*eta_fin*finRatio))
        U_liq =         1/(1/(HTC_liq) + 1/(HTC_air*eta_fin*finRatio))
                
        #Pressure drop estimation
        fri=            0.0015
        v=              4*mfr_channel / (math.pi*ID**2 * rho_in)
        PD=             v**2 * L_tube * fri / (2*ID)
        P_out=          self.P_in# - PD
        
        cp_1=           PropsSI('C', 'P', P_out, 'H', h_in, self.refName)
        U_2=            U_2ph
        
        #3 zones energy balance
        U_1=            U_vap
        U_3=            U_liq
        hg=             PropsSI('H', 'P', self.P_in,    'Q', 1, self.refName)
        hf=             PropsSI('H', 'P', P_out,        'Q', 0, self.refName)
        h_sat_first=    hg
        T_sat_first=    PropsSI('T', 'P', self.P_in,    'Q', 1, self.refName)
        T_sat_second=   PropsSI('T', 'P', P_out,        'Q', 0, self.refName)
        cp_3=           PropsSI('C', 'P', P_out,        'T', T_sat_second+self.subcooling, self.refName)
        
               

        if T_sat_first - self.T_air < 0:
            self.L_1=   L_tube
            self.T_out= getT(self.L_1, U_1, self.T_in, cp_1, self.T_air, mfr_channel, ID)
            self.h_out= PropsSI('H', 'P', P_out,        'T', self.T_out, self.refName)
            self.Q=         self.m_dot*(self.h_out - h_in)
        
        else:
            Tchar1 =        (self.T_in + T_sat_first)/2
            self.L_1=       getL(Tchar1, abs(h_in-h_sat_first), U_1, self.T_air, mfr_channel, ID)
            self.L_2=       L_tube - self.L_1
            
            if self.L_2 < 0:     # 1 zone only
                self.L_1=   L_tube
                self.T_out= getT(self.L_1, U_1, self.T_in, cp_1, self.T_air, mfr_channel, ID)
                self.h_out= PropsSI('H', 'P', P_out,        'T', self.T_out, self.refName)
                self.Q=         self.m_dot*(self.h_out - h_in)
            
            else:
                Tchar2 =        (T_sat_first + T_sat_second)/2
                self.L_2=       getL(Tchar2, abs(hg-hf), U_2, self.T_air, mfr_channel, ID)
                self.L_3=       L_tube - self.L_1 - self.L_2
            
                if self.L_3 < 0:
                    self.L_3 =      0
                    self.L_2 =      L_tube - self.L_1
                    h_2 =           getHout(Tchar2, self.L_2, U_2, self.T_air, mfr_channel,ID,h_in)
                    T_2 =           PropsSI('T', 'P', P_out, 'H', h_2, self.refName)
                    self.Q=         self.m_dot*(h_2 - h_in)
                    self.T_out=     T_2
                    self.h_out=     h_2
                    self.h_inlet=   h_in
           
                else:
                    T_3=            getT(self.L_3, U_3, T_sat_second, cp_3, self.T_air, mfr_channel, ID)
                    h_3=            PropsSI('H', 'P', P_out, 'T', T_3, self.refName)
                    self.Q=         self.m_dot*(h_3 - h_in)
                    self.T_out=     T_3
                    self.h_out=     h_3
                    self.h_inlet=   h_in
    
            cp_a=                   PropsSI('C', 'P', 101325, 'T', self.T_air, self.Condfluid)
            C_a=                    cp_a*self.m_dot_air 
            self.T_air_out=         self.T_air - self.Q/C_a
            
            self.UA =               self.L_3/L_tube*(U_3) + self.L_2/L_tube*(U_2) + self.L_1/L_tube*(U_1)
            self.effectiveness=     self.Q/(self.m_dot_air*cp_a*(self.T_air - self.T_in))





class evaporator:
    """
    Class of evaporator
    .
    Inputs:
        refName:        Refrigerant fluid
        T_air:          K,      indoor (room) air temperature
        m_dot:          kg/s,   refrigerant flowrate
        m_dot_air:      kg/s,   air side flowrate
        P_in:           Pq,     fluid pressure in the inlet
        T_in:           K,      fluid temperature in the inlet
        superheating    K,      Degree of superheating exiting the condenser
    """
    
    def __init__(self, refName, T_air, m_dot, m_dot_air,P_in,h_in,superheating,SysParams, HTC_air):
            
        self.T_air=         T_air           # K, inlet air temperature              273
        self.m_dot=         m_dot           # kg/s, refrigerant mass flow rate      0.02
        self.m_dot_air=     m_dot_air       # kg/s, air mass flow rate              0.4
        self.refName =      refName
        
        self.P_in=          P_in            # Pa, inlet pressure
        self.h_in=          h_in
        self.T_in=          PropsSI('T', 'P', self.P_in, 'H', h_in, refName)            # K, inlet temperature
        self.X_in=          PropsSI('Q', 'P', self.P_in, 'H', h_in, refName)
        
        self.superheating=  superheating    #11     #11.2 K, level of refrigerant superheating
        HTC_air =           HTC_air
        
        rho_in=             PropsSI('D', 'P', self.P_in, 'H', h_in, refName)
        self.L_tube =       SysParams['Evap: TubeL']        # m, tube length        10
        self.Nparallel=     SysParams['Evap: TubeN']        # number of tubes       5
        self.ID=            SysParams['Evap: TubeID']       # m, tube inner diameter 0.007
        self.eta_fin=       SysParams['Evap: FinEff']        # fin efficiency 0.7
        self.finRatio=      SysParams['Evap: FinRatio']        # fin ratio 7.2
        
        # HTC_air=            100     # W/m2K, heat transfer coefficient: refrigerant liquid phase    100
        HTC_liq=            600     # W/m2K, heat transfer coefficient: refrigerant liquid phase    600
        HTC_2ph=            3000    # W/m2K, heat transfer coefficient: refrigerant two phases      3000
        HTC_vap=            600     # W/m2K, heat transfer coefficient: refrigerant vapor phase     600
    
    
        #------------------------------------------------------------------------------
        mfr_channel = self.m_dot/self.Nparallel
            
        #obtaining U for each zone
        self.U_vap = 1/(1/(HTC_vap) + 1/(HTC_air*self.eta_fin*self.finRatio))
        self.U_2ph = 1/(1/(HTC_2ph) + 1/(HTC_air*self.eta_fin*self.finRatio))
                
        #Pressure drop estimation
        fri=        0.0015
        v=          4*mfr_channel / (math.pi*self.ID**2 * rho_in)
        PD=         v**2 * self.L_tube * fri / (2*self.ID)
        P_out=      self.P_in - PD
                
        # cp_1=       PropsSI('C', 'P', self.P_in, 'H', h_in, self.refName)
        
        U_2=        self.U_2ph
        
        

        #Evaporator
        U_3=                self.U_vap
        h_sat_second=       PropsSI('H', 'P', self.P_in, 'Q', 1., self.refName)
        T_sat_second=       PropsSI('T', 'P', self.P_in, 'Q', 1., self.refName)
        cp_3=               PropsSI('C', 'P', P_in, 'T', T_sat_second+superheating, self.refName)
        
        Tchar2 =            (self.T_in + T_sat_second)/2
        self.L_2=           getL(Tchar2, abs(h_in - h_sat_second), U_2, self.T_air, mfr_channel, self.ID)
        self.L_3=           self.L_tube - self.L_2
        
        if self.L_3 <= 0:      # 1 zone model
            self.L_3=       0
            self.L_2=       self.L_tube
            h_2=            getHout(Tchar2, self.L_2, U_2, self.T_air, mfr_channel, self.ID,h_in)
            T_2=            PropsSI('T', 'P', P_out, 'H', h_2, self.refName)
            self.Q=         self.m_dot*(h_2 - h_in)
            
            cp_a=           PropsSI('C', 'P', 101325, 'T', self.T_air, 'Air')
            C_a=            cp_a*self.m_dot_air 
            self.T_air_out= self.T_air - self.Q/C_a
            self.T_out=     T_2
            self.h_out=     h_2
        
        else:               # 2 zones model
            T_3=            getTevap(self.L_3, U_3, T_sat_second, cp_3, self.T_air, mfr_channel, self.ID)
            h_3=            PropsSI('H', 'P', P_out, 'T', T_3, self.refName)
            self.Q=         self.m_dot*(h_3 - h_in)
        
            cp_a=           PropsSI('C', 'P', 101325, 'T', self.T_air, 'Air')
            C_a=            cp_a*self.m_dot_air
            self.T_air_out= self.T_air - self.Q/C_a
            self.T_out=     T_3
            self.h_out=     h_3

        self.UA =           self.L_3/self.L_tube*(U_3) + self.L_2/self.L_tube*(U_2)
        self.effectiveness= self.Q/(self.m_dot_air*cp_a*(self.T_air - self.T_in))






#Functions to retrieve the zone length, enthalpy and temperature
    
def getL(Tchar, dh, U, T_air,mfr_channel,ID):
    dT= abs(Tchar - T_air)
    L= (mfr_channel * dh) / (dT*U*math.pi*ID)
    return L

def getHout(Tchar, length, U, T_air,mfr_channel,ID,h_in):
    UA = U*(math.pi*ID*length)
    dT= (Tchar - T_air)
    h= h_in - UA*dT/mfr_channel
    return h
    
def getT(length, U, Tin, cp,T_air,mfr_channel,ID):
    UA= U*(math.pi*ID*length)
    T=  (UA*(T_air - 0.5*Tin) + mfr_channel*cp*Tin) / (mfr_channel*cp + 0.5*UA)
    
    if (T - T_air) < 0:
        T = T_air
    return T    

def getTevap(length, U, Tin, cp,T_air,mfr_channel,ID):
    UA= U*(math.pi*ID*length)
    T=  (UA*(T_air - 0.5*Tin) + mfr_channel*cp*Tin) / (mfr_channel*cp + 0.5*UA)
    
    if (T_air - T) < 0:
        T = T_air
    return T    


