"""
OPEN-SOURCE SIMULATION PLATFORM FOR AIR SOURCE HEAT PUMP INTEGRATED WITH THERMAL ENERGY STORAGE

authors: Conrado Ermel, Marcus V.A. Bianchi & Paulo S. Schneider

To cite this article: Conrado Ermel, Marcus V.A. Bianchi & Paulo S. Schneider (04 Jul
2024): Open-source simulation platform for air source heat pump integrated with
thermal energy storage, Science and Technology for the Built Environment, 
DOI:10.1080/23744731.2024.2366134

To link to this article: https://doi.org/10.1080/23744731.2024.2366134


=== CLIMATE DATA IMPORTING ===

    Class to import the climate data from:
    - Copernicus
    - Energy plus weather files
    Please refer to the article for detailed description.    
"""

import pandas as pd
import csv
import xarray as xr
import math
import numpy as np




class copernicus2():
    """
    Enter a '.nc' file from Copernicus data base
    
    Read functino Syntax:
    Dry bulb temp. [K] =        a.read('t2m')['t2m']
    Dew temp. [k] =             a.read('d2m')['d2m']
    Relative humidity [%] =     a.read('RH')['RH']
    Soil temp. [K] =            a.read('stl4')['stl4']
    Solar radiation [W/m2]=     a.read('ssr')['ssr']
    Wind speed [m/s] =          a.read('Wind')['Wind']
    """
    def __init__(self,fp):
        """
        """
        self.DS = xr.open_dataset(fp)
        self.dicto={}

    
    def read(self, VR):
        if VR == 't2m':
            TairDB=     self.DS.t2m.values[:,0,0,0]     if  self.DS.t2m.ndim > 3    else   self.DS.t2m.values[:,0,0]
            self.dicto['t2m'] = TairDB
        
        if VR == 'd2m':
            Tdew=       self.DS.d2m.values[:,0,0,0]     if  self.DS.d2m.ndim > 3    else   self.DS.d2m.values[:,0,0]
            self.dicto['d2m'] = Tdew
        
        if VR == 'RH':
            TairDB=     self.DS.t2m.values[:,0,0,0]     if  self.DS.t2m.ndim > 3    else   self.DS.t2m.values[:,0,0]
            Tdew=       self.DS.d2m.values[:,0,0,0]     if  self.DS.d2m.ndim > 3    else   self.DS.d2m.values[:,0,0]
            RH =        [100 * math.exp(17.625 * (Tdew[i]-273)/(243.04 + (Tdew[i]-273))) / math.exp(17.625 * (TairDB[i]-273)/(243.04 + (TairDB[i]-273))) for i in range(len(TairDB))]
            self.dicto['RH'] =      RH
            
        if VR == 'stl4':
            Tgr=        self.DS.stl4.values[:,0,0,0]    if  self.DS.stl4.ndim > 3   else   self.DS.stl4.values[:,0,0]
            self.dicto['stl4'] = Tgr
        
        if VR == 'ssr':
            Solar=      self.DS.ssr.values[:,0,0,0]     if  self.DS.ssr.ndim > 3    else   self.DS.ssr.values[:,0,0]
            Solar=      np.where(Solar>0 ,Solar/(3*3600), 0)
            self.dicto['ssr'] = Solar
        
        if VR == 'Wind':
            uWind=      self.DS.u10.values[:,0,0,0]     if  self.DS.u10.ndim > 3    else   self.DS.u10.values[:,0,0]
            vWind=      self.DS.v10.values[:,0,0,0]     if  self.DS.v10.ndim > 3    else   self.DS.v10.values[:,0,0]
            VwindEx =   np.sqrt(uWind**2 + vWind**2)
            self.dicto['Wind'] = VwindEx
        if VR == 'time':
            time=     self.DS.time.data[:]
            self.dicto['time'] = time

        return self.dicto
        



class epw():
    """A class which represents an EnergyPlus weather (epw) file
        Template from: https://github.com/building-energy/epw
    """
    
    def __init__(self):
        """
        """
        self.headers={}
        self.dataframe=pd.DataFrame()
            
    
    def read(self,fp):
        """Reads an epw file 
        
        Arguments:
            - fp (str): the file path of the epw file   
        
        """
        
        self.headers=self._read_headers(fp)
        self.dataframe=self._read_data(fp)
                
        
    def _read_headers(self,fp):
        """Reads the headers of an epw file
        
        Arguments:
            - fp (str): the file path of the epw file   
            
        Return value:
            - d (dict): a dictionary containing the header rows 
            
        """
        
        d={}
        with open(fp, newline='') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',', quotechar='"')
            for row in csvreader:
                if row[0].isdigit():
                    break
                else:
                    d[row[0]]=row[1:]
        return d
    
    
    def _read_data(self,fp):
        """Reads the climate data of an epw file
        
        Arguments:
            - fp (str): the file path of the epw file   
            
        Return value:
            - df (pd.DataFrame): a DataFrame comtaining the climate data
            
        """
        
        names=['Year',
               'Month',
               'Day',
               'Hour',
               'Minute',
               'Data Source and Uncertainty Flags',
               'Dry Bulb Temperature',
               'Dew Point Temperature',
               'Relative Humidity',
               'Atmospheric Station Pressure',
               'Extraterrestrial Horizontal Radiation',
               'Extraterrestrial Direct Normal Radiation',
               'Horizontal Infrared Radiation Intensity',
               'Global Horizontal Radiation',
               'Direct Normal Radiation',
               'Diffuse Horizontal Radiation',
               'Global Horizontal Illuminance',
               'Direct Normal Illuminance',
               'Diffuse Horizontal Illuminance',
               'Zenith Luminance',
               'Wind Direction',
               'Wind Speed',
               'Total Sky Cover',
               'Opaque Sky Cover (used if Horizontal IR Intensity missing)',
               'Visibility',
               'Ceiling Height',
               'Present Weather Observation',
               'Present Weather Codes',
               'Precipitable Water',
               'Aerosol Optical Depth',
               'Snow Depth',
               'Days Since Last Snowfall',
               'Albedo',
               'Liquid Precipitation Depth',
               'Liquid Precipitation Quantity']
        
        first_row=self._first_row_with_climate_data(fp)
        df=pd.read_csv(fp,
                       skiprows=first_row,
                       header=None,
                       names=names)
        return df
        
        
    def _first_row_with_climate_data(self,fp):
        """Finds the first row with the climate data of an epw file
        
        Arguments:
            - fp (str): the file path of the epw file   
            
        Return value:
            - i (int): the row number
            
        """
        
        with open(fp, newline='') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',', quotechar='"')
            for i,row in enumerate(csvreader):
                if row[0].isdigit():
                    break
        return i
        
        

    def Temp(self,fp):
        """ returns the temperature array from the EPW file
        fp (str): the file path of the new epw file
        """
        
        self.read(fp)
        arr = self.dataframe[["Dry Bulb Temperature"]].to_numpy()
        arr2=[]
        [arr2.append(i[0]) for i in arr]
        return arr2
    
    
    def RH(self,fp):
        """ returns the Relative Humidity array from the EPW file
        fp (str): the file path of the new epw file
        """
        
        self.read(fp)
        arr = self.dataframe[["Relative Humidity"]].to_numpy()
        arr2=[]
        [arr2.append(i[0]) for i in arr]
        return arr2
    
    
    def wind(self,fp):
        """ returns the wind speed array from the EPW file
        fp (str): the file path of the new epw file
        """
        
        self.read(fp)
        arr = self.dataframe[["Wind Speed"]].to_numpy()
        arr2=[]
        [arr2.append(i[0]) for i in arr]
        return arr2
    
    