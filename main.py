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
mu_N_list = [3.2e-3]
mu_SC_list = [3.25e-3]

# mu_N_list = [0.1e-3,1e-3,1.5e-3,2e-3,5e-3,10e-3,15e-3]
# mu_SC_list = [0.1e-3,1e-3,1.5e-3,2e-3,5e-3,10e-3,15e-3]
E_excited_list = [0.035]
# # E_excited_list = [0]
# E_excited_list = np.round(np.arange(-2e-3,2e-3,10e-5),6)
# TeV_list = [1]
# TeV_T_list = [0.5]
TeV_list = [4.5e-3]
TeV_T_list = [4.5e-3]
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

lenswp = len(mu_SC_list) * len(mu_N_list) * len(E_excited_list) * len(TeV_list) * len(PeriBC_list) * len(
    TStrength_list) * len(SNjunc_list) * len(ProximityOn_list)

ShowDensity = False


# np.round(np.arange(0.5,-1.2,-0.01),3),
# delta_list = [6.5e-4] # in eV
delta_list = [5.5e-4] # in eV
# delta_list = [0] # in eV
# delta_list = [6.4e-6]
# delta_list = [0.1] # in eV
VGate_shift_list = [0]

#
# syst.stdout.write("\r{0}".format('--------------------------- Loading Poisson Result -----------------------------------'))
# syst.stdout.flush()

if DavidPot:
    NName = ''
else:
    NName = onedrivepath+'Desktop2/iCloud_Desktop/NN_backup/UpdateSiDopedLayerThickness/2023Y03M23D-17h59m57s'
for DELTA in delta_list:
    for Vg_s in VGate_shift_list:
        E_excited_list = np.arange(-0.5, 0.5, 1/50) # the unit is in t(normalised)
        # B = KC.Kwant_SSeS(NextNanoName=NName, Masterfilepath=master_file_path, ReferenceData=RefName, W_r=400,
        #                   DavidPot=DavidPot, W_g=500, S_g=300, D_2DEG=250,
        #                   V_A=[0], TStrength=TStrength_list,TeV_Normal=True,AddOrbitEffect=True,
        #                   PeriBC=PeriBC_list, Tev=TeV_list, Tev_Tunnel=TeV_T_list,
        #                   E_excited=E_excited_list, SNjunc=SNjunc_list,B_theta = [np.pi/2],B_phi = [0],
        #                   ProOn=ProximityOn_list, BField=[0.1], a=25,
        #                   ShowDensity=ShowDensity, Phase=[3*np.pi/4],
        #                   SaveNameNote=NName, SeriesR=500, DateT=Date, TimeT=Time, MasterMultiRun=MMR,
        #                   muN=mu_N_list, DefectAmp=0, DefectNumPer=0, CombineMu=True, CombineTev=False,
        #                   muSC=mu_SC_list, delta=DELTA, VGate_shift=Vg_s, SwpID="E", PlotbeforeFigures=1,
        #                   PlotbeforeFigures_Ana=20)
        B = KC.Kwant_SSeS(NextNanoName=NName, Masterfilepath=master_file_path, ReferenceData=RefName, W_r=400,
                          DavidPot=DavidPot, W_g=500, S_g=300, D_2DEG=250, gn=-30,L_s = 4000,
                          V_A=[0], TStrength=TStrength_list, TeV_Normal=True,
                          AddOrbitEffect=True, AddZeemanField=True, AddRashbaSOI=True, AddDresselhausSOI=True,
                          PeriBC=PeriBC_list, Tev=TeV_list, Tev_Tunnel=TeV_T_list,
                          E_excited=E_excited_list, SNjunc=SNjunc_list, B_theta=[0], B_phi=[0],
                          ProOn=ProximityOn_list, BField=np.round(np.arange(0, 50) * 0.1, 4), a=20,
                          ShowDensity=ShowDensity, ShowCurrent=False,GetLDOS=True, Phase=[np.pi],
                          SaveNameNote=NName, SeriesR=500, DateT=Date, TimeT=Time, MasterMultiRun=MMR,
                          muN=mu_N_list, DefectAmp=0, DefectNumPer=0, CombineMu=True, CombineTev=False, showBands=False,
                          NumBands=1,
                          muSC=mu_SC_list, delta=DELTA, VGate_shift=Vg_s, SwpID="E", PlotbeforeFigures=1,
                          PlotbeforeFigures_Ana=20)
        B = KC.Kwant_SSeS(NextNanoName=NName, Masterfilepath=master_file_path, ReferenceData=RefName, W_r=400,
                          DavidPot=DavidPot, W_g=500, S_g=300, D_2DEG=250, gn=-30,L_s = 4000,
                          V_A=[0], TStrength=TStrength_list, TeV_Normal=True,
                          AddOrbitEffect=True, AddZeemanField=True, AddRashbaSOI=True, AddDresselhausSOI=True,
                          PeriBC=PeriBC_list, Tev=TeV_list, Tev_Tunnel=TeV_T_list,
                          E_excited=E_excited_list, SNjunc=SNjunc_list, B_theta=[0], B_phi=[0],
                          ProOn=ProximityOn_list, BField=np.round(np.arange(0, 50) * 0.1, 4), a=20,
                          ShowDensity=ShowDensity, ShowCurrent=False,GetLDOS=True, Phase=[np.pi/4],
                          SaveNameNote=NName, SeriesR=500, DateT=Date, TimeT=Time, MasterMultiRun=MMR,
                          muN=mu_N_list, DefectAmp=0, DefectNumPer=0, CombineMu=True, CombineTev=False, showBands=False,
                          NumBands=1,
                          muSC=mu_SC_list, delta=DELTA, VGate_shift=Vg_s, SwpID="E", PlotbeforeFigures=1,
                          PlotbeforeFigures_Ana=20)
        B = KC.Kwant_SSeS(NextNanoName=NName, Masterfilepath=master_file_path, ReferenceData=RefName, W_r=400,
                          DavidPot=DavidPot, W_g=500, S_g=300, D_2DEG=250, gn=-30,L_s = 4000,
                          V_A=[0], TStrength=TStrength_list, TeV_Normal=True,
                          AddOrbitEffect=True, AddZeemanField=True, AddRashbaSOI=True, AddDresselhausSOI=True,
                          PeriBC=PeriBC_list, Tev=TeV_list, Tev_Tunnel=TeV_T_list,
                          E_excited=E_excited_list, SNjunc=SNjunc_list, B_theta=[0], B_phi=[0],
                          ProOn=[0], BField=np.round(np.arange(0, 50) * 0.1, 4), a=20,
                          ShowDensity=ShowDensity, ShowCurrent=False,GetLDOS=True, Phase=[np.pi],
                          SaveNameNote=NName, SeriesR=500, DateT=Date, TimeT=Time, MasterMultiRun=MMR,
                          muN=mu_N_list, DefectAmp=0, DefectNumPer=0, CombineMu=True, CombineTev=False, showBands=False,
                          NumBands=1,
                          muSC=mu_SC_list, delta=DELTA, VGate_shift=Vg_s, SwpID="E", PlotbeforeFigures=1,
                          PlotbeforeFigures_Ana=20)

        B = KC.Kwant_SSeS(NextNanoName=NName, Masterfilepath=master_file_path, ReferenceData=RefName, W_r=400,
                          DavidPot=DavidPot, W_g=500, S_g=300, D_2DEG=250, gn=-30,L_s = 4000,
                          V_A=[0], TStrength=TStrength_list, TeV_Normal=True,
                          AddOrbitEffect=True, AddZeemanField=True, AddRashbaSOI=True, AddDresselhausSOI=True,
                          PeriBC=PeriBC_list, Tev=TeV_list, Tev_Tunnel=TeV_T_list,
                          E_excited=E_excited_list, SNjunc=SNjunc_list, B_theta=[0], B_phi=[0],
                          ProOn=[0], BField=np.round(np.arange(0, 50) * 0.1, 4), a=20,
                          ShowDensity=ShowDensity, ShowCurrent=False,GetLDOS=True, Phase=[np.pi/4],
                          SaveNameNote=NName, SeriesR=500, DateT=Date, TimeT=Time, MasterMultiRun=MMR,
                          muN=mu_N_list, DefectAmp=0, DefectNumPer=0, CombineMu=True, CombineTev=False, showBands=False,
                          NumBands=1,
                          muSC=mu_SC_list, delta=DELTA, VGate_shift=Vg_s, SwpID="E", PlotbeforeFigures=1,
                          PlotbeforeFigures_Ana=20)



