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
RefName = onedrivepath+'Desktop2/iCloud_Desktop/NN_backup/Reference/ReferData.xlsx'

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
SNjunc_list = ['SN']
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
    NName = ''
else:
    # NName = '/mnt/d/OneDrive/Desktop2/iCloud_Desktop/NN_backup/UpdateSiDopedLayerThickness/2023Y05M18D-19h37m02s' # with 1000nm gate width
    # NName = '/mnt/d/OneDrive/Desktop2/iCloud_Desktop/NN_backup/UpdateSiDopedLayerThickness/2023Y03M23D-17h59m57s' # with 1200nm gate width


    # NName_list.append(onedrivepath+'Desktop2/iCloud_Desktop/NN_backup/UpdateSiDopedLayerThickness/2023Y03M23D-17h59m57s')  # with 0.5 / 1    surface charge
    #
    # NName_list.append(onedrivepath+'Desktop2/iCloud_Desktop/NN_backup/UpdateSiDopedLayerThickness/2023Y03M29D-03h35m53s')  # with 1.2 / 1.9  surface charge
    #
    # NName_list.append(onedrivepath+'Desktop2/iCloud_Desktop/NN_backup/UpdateSiDopedLayerThickness/2023Y03M31D-02h23m36s')  # with 1.3 / 1.3  surface charge
    # NName_list.append(onedrivepath+'Desktop2/iCloud_Desktop/NN_backup/UpdateSiDopedLayerThickness/2023Y03M28D-21h12m50s')  # with 1.3 / 1.8  surface charge
    # NName_list.append(onedrivepath+'Desktop2/iCloud_Desktop/NN_backup/UpdateSiDopedLayerThickness/2023Y03M30D-23h41m55s')  # with 1.3 / 1.9  surface charge
    # NName_list.append(onedrivepath+'Desktop2/iCloud_Desktop/NN_backup/UpdateSiDopedLayerThickness/2023Y04M01D-09h09m18s')  # with 1.3 / 2    surface charge
    #
    # NName_list.append(onedrivepath+'Desktop2/iCloud_Desktop/NN_backup/UpdateSiDopedLayerThickness/2023Y04M02D-23h36m23s')  # with 1.4 / 1.4  surface charge
    # NName_list.append(onedrivepath+'Desktop2/iCloud_Desktop/NN_backup/UpdateSiDopedLayerThickness/2023Y03M28D-00h09m17s')  # with 1.4 / 1.9  surface charge
    # #
    # NName_list.append(onedrivepath+'Desktop2/iCloud_Desktop/NN_backup/UpdateSiDopedLayerThickness/2023Y03M26D-23h52m07s')  # with 1.6 / 2.25 surface charge
    # NName_list.append(onedrivepath+'Desktop2/iCloud_Desktop/NN_backup/UpdateSiDopedLayerThickness/2023Y03M25D-03h03m39s')  # with 1.6 / 2.3  surface charge
    # NName_list.append(onedrivepath+'Desktop2/iCloud_Desktop/NN_backup/UpdateSiDopedLayerThickness/2023Y05M25D-00h19m43s')  # with 1.6 / 2.5 surface charge
    # NName_list.append(onedrivepath+'Desktop2/iCloud_Desktop/NN_backup/UpdateSiDopedLayerThickness/2023Y05M26D-04h12m19s')  # with 1.6 / 3 surface charge
    # NName_list.append(onedrivepath+'Desktop2/iCloud_Desktop/NN_backup/UpdateSiDopedLayerThickness/2023Y05M21D-12h03m03s')  # with 1.8 / 2.3 surface charge



    # NName_list.append(onedrivepath+'Desktop2/iCloud_Desktop/NN_backup/UpdateSiDopedLayerThickness/2023Y05M27D-08h34m13s')  # with 1.8 / 2.5 surface charge

    # NName_list.append(onedrivepath + 'Desktop2/iCloud_Desktop/NN_backup/UpdateSiDopedLayerThickness/2023Y07M26D-23h04m09s')  # with 1.8 / 3.5 surface charge with 360nm gate split
    # NName_list.append(onedrivepath + 'Desktop2/iCloud_Desktop/NN_backup/UpdateSiDopedLayerThickness/2023Y07M28D-16h54m39s')  # with 1.8 / 3.5 surface charge with 380nm gate split
    # NName_list.append(onedrivepath + 'Desktop2/iCloud_Desktop/NN_backup/UpdateSiDopedLayerThickness/2023Y07M27D-23h00m59s')  # with 1.8 / 3.5 surface charge with 400nm gate split

    # NName_list.append(
    #     onedrivepath + 'Desktop2/iCloud_Desktop/NN_backup/UpdateSiDopedLayerThickness/2023Y08M10D-00h28m37s')  # with 1.8 / 4 surface charge with 360nm gate split
    # NName_list.append(
    #     onedrivepath + 'Desktop2/iCloud_Desktop/NN_backup/UpdateSiDopedLayerThickness/2023Y07M28D-16h54m39s')  # with 1.8 / 4 surface charge with 380nm gate split
    # NName_list.append(
    #     onedrivepath + 'Desktop2/iCloud_Desktop/NN_backup/UpdateSiDopedLayerThickness/2023Y08M08D-11h31m59s')  # with 1.8 / 4 surface charge with 400nm gate split

    #
    # NName_list.append(onedrivepath+'Desktop2/iCloud_Desktop/NN_backup/UpdateSiDopedLayerThickness/2023Y05M28D-12h21m17s')  # with 1.8 / 3 surface charge
    #
    # NName_list.append(onedrivepath+'Desktop2/iCloud_Desktop/NN_backup/UpdateSiDopedLayerThickness/2023Y05M22D-16h26m01s')  # with 2 / 2.3 surface charge
    # NName_list.append(onedrivepath+'Desktop2/iCloud_Desktop/NN_backup/UpdateSiDopedLayerThickness/2023Y05M24D-09h15m55s')  # with 2 / 2.5 surface charge
    NName_list.append(onedrivepath + 'Desktop2/iCloud_Desktop/NN_backup/UpdateSiDopedLayerThickness/2023Y05M28D-12h21m17s')  # with 1.8 / 3 surface charge(Best)
# for DELTA in delta_list:
#     for Vg_s in VGate_shift_list:
#         for NName in NName_list:
#             B = KC.Kwant_SSeS(NextNanoName=NName,Masterfilepath = master_file_path,ReferenceData = RefName, W_r = 1500, DavidPot=DavidPot, W_g=500, S_g=300, D_2DEG=250,
#                            V_A=np.round(np.arange(0.5,-1.2,-0.01),3), TStrength=TStrength_list,
#                            PeriBC=PeriBC_list, Tev=TeV_list,Tev_Tunnel=TeV_T_list,beta=0,
#                            E_excited=E_excited_list, SNjunc=SNjunc_list,
#                            ProOn=ProximityOn_list,BField=[0],a = 15,
#                            ShowDensity=ShowDensity,Phase=[0],
#                            SaveNameNote=NName,SeriesR = 500,DateT=Date,TimeT = Time,MasterMultiRun=MMR,
#                            muN=mu_N_list, DefectAmp=0,DefectNumPer=0,CombineMu=True,CombineTev=True,
#                            muSC=mu_SC_list, delta=DELTA, VGate_shift=Vg_s,SwpID = "Vg",PlotbeforeFigures=1,PlotbeforeFigures_Ana=20)

for DELTA in delta_list:
    for Vg_s in VGate_shift_list:
        for NName in NName_list:

            B = KC.Kwant_SSeS(NextNanoName=NName, Masterfilepath=master_file_path, ReferenceData=RefName, W_r=1400,
                              WSC=200, DavidPot=DavidPot, W_g=500, S_g=300, D_2DEG=250, gn=30, L_r=5000, L_s=5000,
                              alpha=1, beta=0,
                              V_A=np.round(np.arange(0.5, -1.2, -0.02), 3), TStrength=TStrength_list, TeV_Normal=True,
                              AddOrbitEffect=False, AddZeemanField=True, AddRashbaSOI=True, AddDresselhausSOI=True,
                              PeriBC=[0], Tev=TeV_list, Tev_Tunnel=TeV_T_list,
                              E_excited=[0.006], SNjunc=SNjunc_list,
                              ProOn=[1], constantDelta=False, BField=[(0, 0, 0)], a=20,
                              ShowDensity=ShowDensity, ShowCurrent=False, GetLDOS=False, Swave=False,
                              FieldDependentGap=False, deltaPairingMatrix="sigma_0", deltaPairingMatrix_sign="+",
                              Phase=[np.pi / 2], CloseSystem=False, k_Num=50, mode_Num=2000,
                              SaveNameNote=NName, SeriesR=0, DateT=Date, TimeT=Time, MasterMultiRun=MMR,
                              muN=[0.35], muLead=[0.4], DefectAmp=0, DefectNumPer=0, CombineMu=True,
                              CombineTev=False, showBands=False,
                              NumBands=1, Mapping=False, GetConductance=True,
                              muSC=[0.35], delta=DELTA, delta_real=0.58e-3, VGate_shift=Vg_s, SwpID="Vg",
                              PlotbeforeFigures=1, PlotbeforeFigures_Ana=1).Run_sweep()