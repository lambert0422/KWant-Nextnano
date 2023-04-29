import itertools
import numpy as np
import sys as syst
import Kwant_Class_latest_ROG as KC

W_g_list = [100, 200, 300, 400, 500, 600]
S_g_list = [200, 300, 400, 500, 600]
combWG = list(itertools.product(W_g_list, S_g_list))
DavidPot = True


RefName = '/mnt/d/OneDrive/Desktop2/iCloud_Desktop/NN_backup/Reference/ReferData.xlsx'

mu_N_list = [5e-3,10e-3]
mu_SC_list = np.arange(0,20e-3,100e-5)
# E_excited_list = [0.023,0.024]
# E_excited_list = [0]
E_excited_list = np.round(np.arange(-2e-3,2e-3,10e-5),6)
# TeV_list = [1]
# TeV_T_list = [0.5]
TeV_list = [7e-3,10e-3,15e-3]
TeV_T_list = [7e-3,10e-3,15e-3]
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
delta_list = [6.5e-4] # in eV
# delta_list = [6.4e-6]
# delta_list = [0.1] # in eV
VGate_shift_list = [0]

#
syst.stdout.write("\r{0}".format('--------------------------- Loading Poisson Result -----------------------------------'))
syst.stdout.flush()

if DavidPot:
    Dict, VgList = [], []
    NName = ''
else:
    NName = '/mnt/d/OneDrive/Desktop2/iCloud_Desktop/NN_backup/UpdateSiDopedLayerThickness/2023Y03M23D-17h59m57s'
    ccccccccccc
for DELTA in delta_list:
    for Vg_s in VGate_shift_list:
        E_excited_list = np.arange(-20*DELTA,20*DELTA, 20*DELTA/100)

        B = KC.Kwant_SSeS(NextNanoName=NName,W_r = 600, DavidPot=DavidPot, W_g=500, S_g=300, D_2DEG=250,
                       V_A=[0], TStrength=TStrength_list,
                       PeriBC=PeriBC_list, Tev=TeV_list,Tev_Tunnel=TeV_T_list,
                       E_excited=E_excited_list, SNjunc=SNjunc_list,
                       ProximityOn=ProximityOn_list,BField=[0],Dict = Dict, VgList = VgList,
                       ShowDensity=ShowDensity,phi=[np.pi/4],
                       SaveNameNote='',SeriesR = 500,
                       mu_N=mu_N_list, DefectAmp=0,CombineMu=False,CombineTev=True,
                       mu_SC=mu_SC_list, delta=DELTA, VGate_shift=Vg_s,SwpID = "E",PlotbeforeFigures=20)



