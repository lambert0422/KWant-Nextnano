import itertools
import numpy as np
import sys as syst
import Kwant_Class_ReduceMem as KC
from datetime import datetime
now = datetime.now()
Date = now.strftime("%YY%mM%dD")
Time = now.strftime("%Hh%Mm%Ss")

W_g_list = [100, 200, 300, 400, 500, 600]
S_g_list = [200, 300, 400, 500, 600]
combWG = list(itertools.product(W_g_list, S_g_list))
DavidPot = False
MMR = False
ShowDensity = False

master_file_path = __file__
onedrivepath = '/mnt/c/Users/li244/OneDrive/'
onedrivepath = '/mnt/d/OneDrive/'
# RefName = '/mnt/d/OneDrive/Desktop2/iCloud_Desktop/NN_backup/Reference/ReferData.xlsx'
# RefName = onedrivepath+'Desktop2/iCloud_Desktop/NN_backup/Reference/ReferData.xlsx'

mu_N_list = [3.5e-3]
mu_SC_list =   [3.5e-3]

AC = 5e-6
TeV_list =  np.array([1])
TeV_T_list = np.array([1])

E_excited_list =AC/TeV_list # get around 5uV excitation energy which fit to the measurement
# TeV_list = [2.44e-3]
# TeV_T_list = [2.44e-3]

PeriBC_list = [0]
# TStrength_list = np.round(np.arange(0,2,0.04),5)
TStrength_list = [0]
SNjunc_list = ['SNS']
ProximityOn_list = [1]

lenswp = len(mu_SC_list) * len(mu_N_list) * len(E_excited_list) * len(TeV_list) * len(PeriBC_list) * len(
    TStrength_list) * len(SNjunc_list) * len(ProximityOn_list)





# np.round(np.arange(0.5,-1.2,-0.01),3),
# delta_list = [6.5e-4] # in eV
# delta_list = [6.5e-4] # in eV
delta_list = [0.125]
# delta_list = [0.1] # in eV
VGate_shift_list = [0]

#
# syst.stdout.write("\r{0}".format('--------------------------- Loading Poisson Result -----------------------------------'))
# syst.stdout.flush()
NName_list = []
if DavidPot:
    NName_list = ['']
else:
    # NName_list.append(onedrivepath + 'Desktop2/iCloud_Desktop/NN_backup/NN_2QPCs/2023Y11M09D-01h10m57s')  # No surface charge
    NName_list.append(
        onedrivepath + 'Desktop2/iCloud_Desktop/NN_backup/NN_ABMF/2023Y12M06D-05h20m31s')  # No surface charge

for DELTA in delta_list:
    for Vg_s in VGate_shift_list:
        for NName in NName_list:
            Estep = 0.005
            # Estep = 0.1
            Emin = -0.25
            Emax = 0.25

            E_excited_list = np.round(np.arange(Emin * 100, Emax * 100 + Estep * 100, Estep * 100) / 100,
                                      14)  # the unit is in t(normalised)

            B = KC.Kwant_SSeS(NextNanoName=NName, Masterfilepath=master_file_path, W_r=200,
                              WSC=100, DavidPot=DavidPot, W_g=50, S_g=250, D_2DEG=250, gn=30, L_r=2500, L_s=2000,
                              alpha=1, beta=0, VGate_x_f = 2,VGate_y_f =2,DAir = 0,
                              V_A=np.round(np.arange(0.5, -2, -0.02), 3), TStrength=TStrength_list, TeV_Normal=True,
                              AddOrbitEffect=False, AddZeemanField=True, AddRashbaSOI=True, AddDresselhausSOI=True,
                              PeriBC=[0], Tev=TeV_list, Tev_Tunnel=TeV_T_list,
                              E_excited=E_excited_list, SNjunc=SNjunc_list, Surface2DEG=False,
                              ProOn=[0], constantDelta=False, BField=[(0, 0, 0)], a=20,
                              ShowDensity=ShowDensity, ShowCurrent=False, Swave=False,
                              FieldDependentGap=False, deltaPairingMatrix="sigma_0", deltaPairingMatrix_sign="+",
                              Phase=[np.pi], CloseSystem=True,
                              SaveNameNote=NName, SeriesR=0, DateT=Date, TimeT=Time, MasterMultiRun=MMR,
                              muN=[0.4], muLead=[0.5], DefectAmp=0, DefectNumPer=0, CombineMu=True,
                              CombineTev=False, showBands=False, OhmicContact=True,
                              NumBands=1, Mapping=True, GetConductance=False, GetLDOS=True, Two_QPC=True,
                              muSC=[0.4], delta=DELTA, delta_real=0.58e-3, VGate_shift=Vg_s, SwpID="E",
                              PlotbeforeFigures=200, PlotbeforeFigures_Ana=1).Run_sweep()

            # B = KC.Kwant_SSeS(NextNanoName=NName,  Masterfilepath=master_file_path, W_r=100,
            #                   WSC=100, DavidPot=DavidPot, W_g=50, S_g=100, D_2DEG=250, gn=30, L_r=5000, L_s=4000,
            #                   alpha=1, beta=0,
            #                   V_A=np.round(np.arange(0.5, -2, -0.02), 3), TStrength=TStrength_list, TeV_Normal=True,
            #                   AddOrbitEffect=False, AddZeemanField=True, AddRashbaSOI=True, AddDresselhausSOI=True,
            #                   PeriBC=[0], Tev=TeV_list, Tev_Tunnel=TeV_T_list,
            #                   E_excited=E_excited_list, SNjunc=SNjunc_list, Surface2DEG=False,
            #                   ProOn=[1,0], constantDelta=False, BField=[(0, 0, 0)], a=20,
            #                   ShowDensity=ShowDensity, ShowCurrent=False, Swave=False,
            #                   FieldDependentGap=False, deltaPairingMatrix="sigma_0", deltaPairingMatrix_sign="+",
            #                   Phase=[np.pi], CloseSystem=True,
            #                   SaveNameNote=NName, SeriesR=0, DateT=Date, TimeT=Time, MasterMultiRun=MMR,
            #                   muN=[0.4], muLead=[0.5], DefectAmp=0, DefectNumPer=0, CombineMu=True,
            #                   CombineTev=False, showBands=False, OhmicContact=True,
            #                   NumBands=1, Mapping=True, GetConductance=False, GetLDOS=True, Two_QPC=True,
            #                   muSC=[0.4], delta=DELTA, delta_real=0.58e-3, VGate_shift=Vg_s, SwpID="E",
            #                   PlotbeforeFigures=200, PlotbeforeFigures_Ana=200).Run_sweep()




