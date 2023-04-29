import itertools
import numpy as np
import sys as syst
import Kwant_Class_ReduceMem as KC

W_g_list = [100, 200, 300, 400, 500, 600]
S_g_list = [200, 300, 400, 500, 600]
combWG = list(itertools.product(W_g_list, S_g_list))
DavidPot = True


RefName = '/mnt/d/OneDrive/Desktop2/iCloud_Desktop/NN_backup/Reference/ReferData.xlsx'

mu_N_list = [2e-3,4e-3,8e-3]
mu_SC_list = [2e-3,4e-3,8e-3]
# E_excited_list = [0.023,0.024]
# E_excited_list = [0]
E_excited_list = np.round(np.arange(-2e-3,2e-3,10e-5),6)
# TeV_list = [1]
# TeV_T_list = [0.5]
TeV_list = [5e-3,7e-3]
TeV_T_list = [5e-3,7e-3]
# mu_N_list = [1e-3,2e-3]
# mu_SC_list = [1e-3,2e-3]
# # E_excited_list = [0.023,0.024]
# E_excited_list = [0.01]
# TeV_list = [3e-3]


PeriBC_list = [0]
# TStrength_list = np.round(np.arange(0,2,0.04),5)
TStrength_list = [0]
SNjunc_list = ['SNS']
ProximityOn_list = [1]

lenswp = len(mu_SC_list) * len(mu_N_list) * len(E_excited_list) * len(TeV_list) * len(PeriBC_list) * len(
    TStrength_list) * len(SNjunc_list) * len(ProximityOn_list)

if ~DavidPot & lenswp > 1:
    ShowDensity = False
elif DavidPot & lenswp * len(W_g_list) * len(S_g_list) > 1:
    ShowDensity = False
else:
    ShowDensity = True

# np.round(np.arange(0.5,-1.2,-0.01),3),
delta_list = [6.5e-4] # in eV
delta_list = [6.5e-3,6.5e-4,6.5e-5] # in eV
# delta_list = [6.4e-6]
# delta_list = [0.1] # in eV
VGate_shift_list = [0]



if DavidPot:
  
    NName = ''
else:
    NName = '/mnt/d/OneDrive/Desktop2/iCloud_Desktop/NN_backup/UpdateSiDopedLayerThickness/2023Y03M23D-17h59m57s'
   
for DELTA in delta_list:
    for Vg_s in VGate_shift_list:
        E_excited_list = np.arange(-8*DELTA,8*DELTA, 8*DELTA/100)
        B = KC.Kwant_SSeS(NextNanoName=NName, W_r=400, DavidPot=DavidPot, W_g=500, S_g=300, D_2DEG=250,
                          V_A=[0], TStrength=TStrength_list,
                          PeriBC=PeriBC_list, Tev=TeV_list, Tev_Tunnel=TeV_T_list,
                          E_excited=E_excited_list, SNjunc=SNjunc_list,
                          ProOn=ProximityOn_list, BField=[0],
                          ShowDensity=ShowDensity, Phase=[np.pi / 4],
                          SaveNameNote='', SeriesR=500,TeV_Normal=True,
                          muN=mu_N_list, DefectAmp=0, CombineMu=False, CombineTev=True,
                          muSC=mu_SC_list, delta=DELTA, VGate_shift=Vg_s, SwpID="E", PlotbeforeFigures=1)
        B = KC.Kwant_SSeS(NextNanoName=NName,W_r = 600, DavidPot=DavidPot, W_g=500, S_g=300, D_2DEG=250,
                       V_A=[0], TStrength=TStrength_list,
                       PeriBC=PeriBC_list, Tev=TeV_list,Tev_Tunnel=TeV_T_list,
                       E_excited=E_excited_list, SNjunc=SNjunc_list,
                       ProOn=ProximityOn_list,BField=[0],
                       ShowDensity=ShowDensity,Phase=[np.pi/4],
                       SaveNameNote='',SeriesR = 500,TeV_Normal=True,
                       muN=mu_N_list, DefectAmp=0,CombineMu=False,CombineTev=True,
                       muSC=mu_SC_list, delta=DELTA, VGate_shift=Vg_s,SwpID = "E",PlotbeforeFigures=1)



