import itertools
import numpy as np
import sys as syst
import Kwant_Class_ReduceMem as KC
from datetime import datetime
# import matplotlib.pyplot as plt
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
# alphaList = [1,2]
#
# syst.stdout.write("\r{0}".format('--------------------------- Loading Poisson Result -----------------------------------'))
# syst.stdout.flush()


# W_r_list = [41]
# WSC_list = [161]
if DavidPot:
    NName = ''
else:
    NName = onedrivepath+'Desktop2/iCloud_Desktop/NN_backup/UpdateSiDopedLayerThickness/2023Y03M23D-17h59m57s'

for DELTA in delta_list:
    for alphaTest in alphaList:
        # Estep = 0.005
        # Estep = 0.001
        Estep = 0.005
        Emin = -0.25
        Emax = 0.25
        E_excited_list = np.round(np.arange(Emin*100, Emax*100+Estep*100,Estep*100)/100,14) # the unit is in t(normalised)
        # alphaTest = 1# in the unit of t
        # BTest = [1]
        mu_N_list = [0.25] # in the univt of t
        mu_SC_list = [0.25]# in the unit of t
        mu_Lead_list = [0.25]
        GetConductance = False
        numTheta = 10
        theta_list = np.round(np.arange(0,numTheta+1)/(2*numTheta),13)*np.pi
        PhiAngle = 0
        Bmag = 1
        PhaseNum = 101
        # BTest_list =np.round( [(Bmag*np.sin(angle)*np.cos(PhiAngle), Bmag*np.sin(angle)*np.sin(PhiAngle), Bmag*np.cos(angle)) for angle in theta_list],13)
        BTest_list = [(0, 0, np.round(x, 13)) for x in np.linspace(0, 5, 101)]
        # BTest_list = [(0,0,1)]
        # plt.plot([t[0] for t in BTest_list],[t[2] for t in BTest_list])
        # plt.show()
        # B = KC.Kwant_SSeS(NextNanoName=NName, Masterfilepath=master_file_path, ReferenceData=RefName, W_r=120, WSC=160,
        #                   DavidPot=DavidPot, W_g=500, S_g=300, D_2DEG=250, gn=30, L_r=4000, L_s=4000,
        #                   alpha=alphaTest, beta=0,V_A=[0], TStrength=TStrength_list, TeV_Normal=True,Surface2DEG = True,
        #                   AddOrbitEffect=False, AddZeemanField=True, AddRashbaSOI=True, AddDresselhausSOI=True,
        #                   PeriBC=[0], Tev=TeV_list, Tev_Tunnel=TeV_T_list,TunnelLength = 1,
        #                   E_excited=E_excited_list, SNjunc=SNjunc_list,
        #                   ProOn=[0], constantDelta=False, BField=[(0,0,0.1)], a=20,
        #                   ShowDensity=ShowDensity, ShowCurrent=False, GetLDOS=True, Swave=False,
        #                   FieldDependentGap=False, deltaPairingMatrix="sigma_0", deltaPairingMatrix_sign="+",
        #                   Phase=2*np.pi*np.round(np.arange(0,PhaseNum+1)/PhaseNum,5),
        #                   CloseSystem=True,
        #                   SaveNameNote=NName, SeriesR=0, DateT=Date, TimeT=Time, MasterMultiRun=MMR,
        #                   muN=mu_N_list, muLead=mu_Lead_list, DefectAmp=0, DefectNumPer=0, CombineMu=True,
        #                   CombineTev=False, showBands=False, LockFieldAngle= False,
        #                   NumBands=1, Mapping=True, GetConductance=GetConductance,
        #                   muSC=mu_SC_list, delta=DELTA, delta_real=0.58e-3, VGate_shift=Vg_s, SwpID="E",
        #                   PlotbeforeFigures=100,
        #                   PlotbeforeFigures_Ana=20).Run_sweep()
        # W_r should be 120 WSC can be 160
        B = KC.Kwant_SSeS(NextNanoName=NName, Masterfilepath=master_file_path, ReferenceData=RefName, W_r=200, WSC=200,
                          DavidPot=DavidPot, W_g=500, S_g=300, D_2DEG=250, gn=30, L_r=2000, L_s=2000,
                          alpha=alphaTest, beta=0, V_A=[0], TStrength=TStrength_list, TeV_Normal=True, Surface2DEG=True,
                          AddOrbitEffect=False, AddZeemanField=True, AddRashbaSOI=True, AddDresselhausSOI=False,
                          PeriBC=[0], Tev=TeV_list, Tev_Tunnel=TeV_T_list, TunnelLength=1,
                          E_excited=E_excited_list, SNjunc=SNjunc_list,
                          ProOn=[0], constantDelta=True, BField=BTest_list, a=20,
                          ShowDensity=ShowDensity, ShowCurrent=False, GetLDOS=True, Swave=False,
                          FieldDependentGap=False, deltaPairingMatrix="sigma_0", deltaPairingMatrix_sign="+",
                          Phase=[np.pi],
                          CloseSystem=True,
                          SaveNameNote=NName, SeriesR=0, DateT=Date, TimeT=Time, MasterMultiRun=MMR,
                          muN=mu_N_list, muLead=mu_Lead_list, DefectAmp=0, DefectNumPer=0, CombineMu=True,
                          CombineTev=False, showBands=False, LockFieldAngle=False,
                          NumBands=1, Mapping=True, GetConductance=False,
                          muSC=mu_SC_list, delta=DELTA, delta_real=0.58e-3, VGate_shift=Vg_s, SwpID="E",
                          PlotbeforeFigures=100,
                          PlotbeforeFigures_Ana=20).Run_sweep()
        B = []
        # B = KC.Kwant_SSeS(NextNanoName=NName, Masterfilepath=master_file_path, ReferenceData=RefName, W_r=120, WSC=160,
        #                   DavidPot=DavidPot, W_g=500, S_g=300, D_2DEG=250, gn=30, L_r=4000, L_s=4000,
        #                   alpha=alphaTest, beta=0, V_A=[0], TStrength=TStrength_list, TeV_Normal=True, Surface2DEG=True,
        #                   AddOrbitEffect=False, AddZeemanField=True, AddRashbaSOI=True, AddDresselhausSOI=True,
        #                   PeriBC=[0], Tev=TeV_list, Tev_Tunnel=TeV_T_list, TunnelLength=1,
        #                   E_excited=E_excited_list, SNjunc=SNjunc_list,
        #                   ProOn=[0], constantDelta=False, BField=BTest_list, a=20,
        #                   ShowDensity=ShowDensity, ShowCurrent=False, GetLDOS=True, Swave=False,
        #                   FieldDependentGap=False, deltaPairingMatrix="sigma_0", deltaPairingMatrix_sign="+",
        #                   Phase=[np.pi],
        #                   CloseSystem=True,
        #                   SaveNameNote=NName, SeriesR=0, DateT=Date, TimeT=Time, MasterMultiRun=MMR,
        #                   muN=mu_N_list, muLead=mu_Lead_list, DefectAmp=0, DefectNumPer=0, CombineMu=True,
        #                   CombineTev=False, showBands=False, LockFieldAngle=False,
        #                   NumBands=1, Mapping=True, GetConductance=GetConductance,
        #                   muSC=mu_SC_list, delta=DELTA, delta_real=0.58e-3, VGate_shift=Vg_s, SwpID="E",
        #                   PlotbeforeFigures=100,
        #                   PlotbeforeFigures_Ana=20).Run_sweep()
        # B = []
