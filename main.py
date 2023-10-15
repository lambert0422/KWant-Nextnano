import itertools
import numpy as np
import sys as syst
import Kwant_Class_ReduceMem as KC
from datetime import datetime

W_g_list = [100, 200, 300, 400, 500, 600]
S_g_list = [200, 300, 400, 500, 600]
combWG = list(itertools.product(W_g_list, S_g_list))
DavidPot = True
MMR = True
now = datetime.now()
Date = now.strftime("%YY%mM%dD")
Time = now.strftime("%Hh%Mm%Ss")
master_file_path = __file__
RefName = None
# onedrivepath = '/mnt/c/Users/li244/OneDrive/'


onedrivepath = '/mnt/d/OneDrive/'


# mu_N_list = [0.1e-3,1e-3,1.5e-3,2e-3,5e-3,10e-3,15e-3]
# mu_SC_list = [0.1e-3,1e-3,1.5e-3,2e-3,5e-3,10e-3,15e-3]
E_excited_list = [0.035]
# # E_excited_list = [0]
# E_excited_list = np.round(np.arange(-2e-3,2e-3,10e-5),6)
# TeV_list = [1]
# TeV_T_list = [0.5]
TeV_list = [1]
TeV_T_list = [1]
# mu_N_list = [1e-3,2e-3]
# mu_SC_list = [1e-3,2e-3]
# # E_excited_list = [0.023,0.024]
# E_excited_list = [0.01]

# TeV_list = [3e-3]


PeriBC_list = [1]
# TStrength_list = np.round(np.arange(0,2,0.04),5)
TStrength_list = [0]
SNjunc_list = ['SNS']
ProximityOn_list = [1]


ShowDensity = False


# np.round(np.arange(0.5,-1.2,-0.01),3),
# delta_list = [6.5e-4] # in eV
delta_list = [0.125] # in eV
# delta_list = [0] # in eV
# delta_list = [6.4e-6]
# delta_list = [0.1] # in eV
Vg_s = [0]
alphaList = [1]
#
# syst.stdout.write("\r{0}".format('--------------------------- Loading Poisson Result -----------------------------------'))
# syst.stdout.flush()
BTest_list = [1]
if DavidPot:
    NName = ''
else:
    NName = onedrivepath+'Desktop2/iCloud_Desktop/NN_backup/UpdateSiDopedLayerThickness/2023Y03M23D-17h59m57s'
for DELTA in delta_list:
    for alphaTest in alphaList:
        for BTest in BTest_list:
            Estep = 0.001
            # Estep = 0.03
            Emin = -0.25
            Emax = 0.25
            E_excited_list = np.round(np.arange(Emin*100, Emax*100+Estep*100,Estep*100)/100,14) # the unit is in t(normalised)
            # alphaTest = 1# in the unit of t
            # BTest = [1]
            mu_N_list = [0.25] # in the univt of t
            mu_SC_list = [0.25]# in the unit of t
            mu_Lead_list = [0.25]
            GetConductance = False
            BTest = [BTest]

            B = KC.Kwant_SSeS(NextNanoName=NName, Masterfilepath=master_file_path, ReferenceData=RefName, W_r=81, WSC=161,
                              DavidPot=DavidPot, W_g=500, S_g=300, D_2DEG=250, gn=30, L_r=4001, L_s=4001,
                              alpha=alphaTest, beta=0,
                              V_A=[0], TStrength=[0], TeV_Normal=True,
                              AddOrbitEffect=False, AddZeemanField=True, AddRashbaSOI=True, AddDresselhausSOI=True,
                              PeriBC=[0], Tev=TeV_list, Tev_Tunnel=TeV_T_list,
                              E_excited=E_excited_list, SNjunc=SNjunc_list, B_theta=[np.pi / 2], B_phi=[0],
                              ProOn=[0], constantDelta=True, BField=BTest, a=20,
                              ShowDensity=ShowDensity, ShowCurrent=False, GetLDOS=True, Swave=False,
                              FieldDependentGap=False, deltaPairingMatrix="sigma_0", deltaPairingMatrix_sign="+",
                              Phase=np.round(np.arange(0, 101) / 50, 5) * np.pi,
                              CloseSystem=True, k_Num=50, mode_Num=2000,
                              SaveNameNote=NName, SeriesR=0, DateT=Date, TimeT=Time, MasterMultiRun=MMR,
                              muN=mu_N_list, muLead=mu_Lead_list, DefectAmp=0, DefectNumPer=0, CombineMu=True,
                              CombineTev=False, showBands=False,
                              NumBands=1, Mapping=True, GetConductance=GetConductance,
                              muSC=mu_SC_list, delta=DELTA, delta_real=0.58e-3, VGate_shift=Vg_s, SwpID="E",
                              PlotbeforeFigures=200,
                              PlotbeforeFigures_Ana=20).Run_sweep()
