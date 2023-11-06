import os
import csv
from os import system, name

from prettytable import PrettyTable
# import colorama
# colorama.init()
import sys as syst
# from cmath import exp
import numpy as np
from matplotlib.transforms import Affine2D
import mpl_toolkits.axisartist.floating_axes as floating_axes
import numpy as np
# import mpl_toolkits.axisartist.angle_helper as angle_helper
# from matplotlib.projections import PolarAxes
from mpl_toolkits.axisartist.grid_finder import (FixedLocator, MaxNLocator,
                                                 DictFormatter)
import matplotlib.pyplot as plt
# import tinyarray
import warnings
import kwant
import scipy.sparse.linalg as sla

from scipy.optimize import fsolve
from scipy.linalg import ishermitian, eig, eigh
from scipy.sparse import diags,coo_matrix
import kwant.continuum
import pandas as pd
import itertools
# from itertools import chain, combinations
import nextnanopy as nn
# from scipy.interpolate import interp2d
from scipy.interpolate import LinearNDInterpolator
import re
import time
from datetime import datetime

def savedata(file_in,init,initdata = None,newcol = None):
    # Save the updated table to a file
    tempfile = 'temp.txt'
    if init:
        with open(file_in, 'w', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            for row in initdata:
                writer.writerow(row)
    else:
        with open(file_in, 'r', newline='') as infile, \
                open(tempfile, 'w', newline='') as outfile:
            reader = csv.reader(infile, delimiter='\t')
            writer = csv.writer(outfile, delimiter='\t')

            for i, row in enumerate(reader):
                new_data = newcol[i]
                row += new_data

                writer.writerow(row)

        with open(tempfile, 'r', newline='') as infile, \
                open(file_in, 'w', newline='') as outfile:
            reader = csv.reader(infile, delimiter='\t')
            writer = csv.writer(outfile, delimiter='\t')

            for i, row in enumerate(reader):
                writer.writerow(row)
def clear():
    # for windows the name is 'nt'
    if name == 'nt':
        _ = system('cls')

    # and for mac and linux, the os.name is 'posix'
    else:
        _ = system('clear')


def ExpRounded(angle):
    return 1j * np.round(np.sin(angle), 15) + np.round(np.cos(angle), 15)


def TimeFormat(sec):
    # Format the time display into h/m/s
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    if h > 0:
        result = f"{int(h)}h{int(m)}m{int(s)}s"
    elif m > 0:
        result = f"{int(m)}m{int(s)}s"
    else:
        result = f"{int(s)}s"
    return result


def SearchFolder(NextNanoName, filename, varname, Vg_target=None, xlim=None, ylim=None, v=None, PLOT=0):
    my_datafolder = nn.DataFolder(NextNanoName)  # search the folder for all the file
    FileList = my_datafolder.find(filename, deep=True)  # find the file with specific filename
    Dict_Potential = {}  # define potential dictionary
     # define gate voltage list
    VgCount = 0  # count the gate voltage number
    Vg_target = list(Vg_target)
    if 0.0 not in Vg_target:
        Vg_target.append(0.0)
    Vg_target.sort()
    VgList = np.zeros(len(Vg_target))
    # syst.stdout.write(
    #         "\r{0}".format('--------------------------- Loading Poisson Result -----------------------------------'))
    # syst.stdout.flush()
    print("--------------------------- Loading Poisson Result -----------------------------------", end='\r')
    for FolderName in FileList:
        FolderName_Prv = FolderName.replace('/' + filename, '')  # get the file path before file name
        FolderName_bias = FolderName_Prv + '/bias_points.log'  # search the bias point file from Nextnano
        my_datafolder_Prv = nn.DataFolder(FolderName_Prv)
        with open(FolderName_bias) as f:
            lines = f.readlines()  # read the bias point file

        LogList = re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", lines[1])  # only get the number
        Vg = LogList[1]

        if np.round(float(Vg), 5) in Vg_target:
            VgList[VgCount] = np.round(float(Vg), 5)


            TwoD_PotentialData = nn.DataFile(FolderName, product='nextnano++')  # read the potential data form nextnano
            y = TwoD_PotentialData.coords['x']
            x = TwoD_PotentialData.coords['y']
            z = TwoD_PotentialData.variables[varname]

            # Fv = interp2d(x.value, y.value, z.value, kind='cubic', copy=True, bounds_error=False, fill_value=None) # interpolate the 2D matrix
            xv, yv = np.meshgrid(x.value, y.value, indexing='xy')

            A = np.array(list(zip(xv, yv)))

            xv = np.reshape(xv, (np.size(xv, 0) * np.size(xv, 1)))
            yv = np.reshape(yv, (np.size(yv, 0) * np.size(yv, 1)))
            Fv = LinearNDInterpolator(list(zip(xv, yv)),
                                      np.reshape(z.value, (np.size(z.value, 0) * np.size(z.value, 1))))
            Dict_Potential[VgCount] = Fv  # store the 2D potential into the potential dict
            VgCount += 1
            if PLOT == 1:
                fig, ax = plt.subplots(1)
                if v != None:
                    pcolor = ax.pcolormesh(x.value, y.value, z.value, vmin=v[0], vmax=v[1])
                else:
                    pcolor = ax.pcolormesh(x.value, y.value, z.value)
                cbar = fig.colorbar(pcolor)
                cbar.set_label(z.label)
                ax.set_xlabel(x.label)
                ax.set_ylabel(y.label)
                fig.tight_layout()
                if xlim != None:
                    plt.xlim(xlim)
                if ylim != None:
                    plt.ylim(ylim)
                fig.show()
    return Dict_Potential, list(VgList)



def setup_axes(fig, rect):
    """
    A simple one.
    """
    tr = Affine2D().scale(2, 1).rotate_deg(90)

    grid_helper = floating_axes.GridHelperCurveLinear(
        tr, extremes=(-0.5, 3.5, 0, 4),
        grid_locator1=MaxNLocator(nbins=4),
        grid_locator2=MaxNLocator(nbins=4))

    ax1 = fig.add_subplot(
        rect, axes_class=floating_axes.FloatingAxes, grid_helper=grid_helper)

    aux_ax = ax1.get_aux_axes(tr)

    return ax1, aux_ax


def set_size(w, h, ax=None):
    """ w, h: width, height in inches """
    if not ax: ax = plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w) / (r - l)
    figh = float(h) / (t - b)
    ax.figure.set_size_inches(figw, figh)


class Kwant_SSeS():
    ##########################################################################################################################
    # alpha = 34e-3  # eVnm SOI
    # # alpha = 0 # eVnm SOI
    # # Digits = 5  # Digits to round the import voltage
    # gn = 10  # g-factor

    # alpha = 0 # eVnm SOI
    # Digits = 5  # Digits to round the import voltage
    # gn = 0  # g-factor
    ##########################################################################################################################
    # mu_N = 12.5e-3  # eV chemical potential at normal material side
    # # mu_SC = 50e-3 #eV in Nature paper
    # mu_SC = 20e-3 #eV in Nature paper

    # mu_N = -5e-3 # eV chemical potential at normal material side
    # mu_SC = mu_N  # eV in Nature paper
    # mu_SC = 50e-3  # eV in Nature paper
    # mu_SC = 20e-3 #eV For fit with 2DEG(Ohmic contact)
    # mu_SC = 80e-3 #eV For fit with 2DEG(Ohmic contact) chemical potential at supercondcutor material side
    ##########################################################################################################################
    # in the unit of hopping term

    kB = 1.380649e-23  # Boltzmann constant
    me = 9.10938356e-31  # kg electron mass
    m = 0.039 * me  # kg effective electron mass
    e = 1.60217662e-19  # electron charge
    c = 299792458.0
    hbar = 1.05471817e-34  # J.s # Planck constant
    mu_B = hbar / (2 * me) # magneto, without multiple e to reduce accuracy
    epsilon_r = 13.5 # InGaAs
    epsilon_0 = 8.8541878128e-12
    n_2DEG = 2.24e15  # in m^-2 electron density of 2DEG
    lambda_london = 40e-9 # London penetration depth of Nb is 39+-5nm
    # t_cal = 1000*hbar**2/(e*2*m*(a*1e-9)**2)
    t_cal = 100
    Hcb = 3.5# the bulk critical magnetic field of Nb is 3.5T

    a_B = epsilon_r * epsilon_0 * hbar ** 2 / (m * e ** 2)
    kF = (4 * np.pi * n_2DEG) ** 0.5
    lambdaF = 1e9*2*np.pi/kF # make it nm
    vF = hbar * kF / (m)


    ##########################################################################################################################
    # Pauli matrix in Nambu spinor
    ##########################################################################################################################

    taux = np.array([[0, 1], [1, 0]])
    tauy = np.array([[0, -1j], [1j, 0]])
    tauz = np.array([[1, 0], [0, -1]])

    I = np.identity(2)
    s_0 = np.identity(2)
    s_z = np.array([[1, 0], [0, -1]])
    s_x = np.array([[0, 1], [1, 0]])
    s_y = np.array([[0, -1j], [1j, 0]])
    s_p = np.array([[0, 2], [0, 0]])
    s_m = np.array([[0, 0], [2, 0]])

    # s_0_t_p = tinyarray.array(np.kron(s_p, s_0))
    # s_0_t_m = tinyarray.array(np.kron(s_m, s_0))
    # s_0_t_z = tinyarray.array(np.kron(s_z, s_0))
    # t_z_s_0 = tinyarray.array(np.kron(s_0, s_z))
    # s_y_t_z = tinyarray.array(np.kron(s_z, s_y))
    # s_x_t_z = tinyarray.array(np.kron(s_z, s_x))
    # s_0_s_x = tinyarray.array(np.kron(s_x, s_0))
    # s_0_s_y = tinyarray.array(np.kron(s_y, s_0))
    # s_y_s_x = tinyarray.array(np.kron(s_x, s_y))
    # s_x_s_y = tinyarray.array(np.kron(s_y, s_x))
    # t_z_s_y = tinyarray.array(np.kron(s_y, s_z))
    # t_z_s_x = tinyarray.array(np.kron(s_x, s_z))
    ##########################################################################################################################
    # GridFactor: tune the discretized grid size
    # DavidPot: use David method or Nextnano potential
    # W_r, L_r, D_2DEG: width, length and depth of 2DEG in nm
    # L_s: the length of the region that have superconductor contact
    # d: thickness of the Nb in nm
    # T: temperature
    # BField: list of all magnetic field to sweep
    # V_A: applied gate voltage
    # TStrength: list of tunnel barrier height in the unit of eV
    # TunnelLength: the gird of tunnel barrier
    # Tev: list of tunnel term in eV
    # E_excited: list of excitation energy in eV
    # phi: phase difference between two superconductor
    # Vbias: list of DC bias between two leads
    # PeriBC: list of whether or not to have periodic boundary condition (0/1)
    # SNjunc: list of whether to have two superconducting leads or one normal one superconducting ('SN'/'SNS')
    # ProximityOn: list of turn on off proximity effect(0/1)
    # constantDelta: when have proximity on, this flags a constant induced delta in the 2DEG having the value of parent gap
    # delta: superconductor pairing, order parameter, superconductor gap
    # NextNanoName: the folder path of the Nextnano result
    # ShowDensity: whether to plot density plot or not
    # Digits: round the parameter to what digits
    # SaveNameNote: the end of the file to save, as the note to the result
    # Swave: whether Swave or other, Dwave Pwave
    # CombineMu: Make muS and muN the same, without sweep permutatively, if true, the chemical potential list will be mu_SC
    # mu_N: chemical potential of normal material in eV
    # mu_SC: chemical potential of the superconductor in eV
    # VGate_shift: the potential shift from the gate bias
    # a: the actual length between every two grid in nm
    # DefectAmp: the amplotude of the random Gaussian defect in the unit of t
    # SwpID: what parameter to sweep and form figures: Vg/Vbias/E/B
    # AddOrbitEffect: whether to include orbital effect in the Hamiltonian
    # PlotbeforeFigures: how many figures between two displayed plots
    # showBands: whether to plot the bands of the Hamiltonian
    # NumBands: number of points to how bands, it equals the nunber of modes in the bands
    # ACFix and AC: fix the AC signal as the E excitation is in the unit of t and t is in unit of eV, so fix AC means fix product of E and t
    # CloseSystem: whether to perform a extra calculation on a closed system of the scattering region, get k_Num lowest eigenvector and output the mode_Num waverfucntion density
    def __init__(self, DavidPot=False,alpha = 2.25e-3 ,beta = 2.25e-3,gn = -3.4,
                 Temp=0.1,delta=0.125, delta_real = 0.58e-3,muN=0.25, muSC=0.25,muLead=0.25, VGate_shift=-0.1, DefectAmp=0.5, DefectNumPer = 10,SeriesR=0,
                 W_g=300, S_g=400, Nb_d=100, D_2DEG=120, W_r=1400, L_r=5000, L_s=4000, WSC=200, a=30, GridFactor=1,
                 BField=[0], V_A=np.arange(0, -1.49, -0.01), Vbias_List=[0],Tev=[1e-3], Tev_Tunnel=[2e-3],
                 E_excited=[5e-3],TStrength=[0], TunnelLength=2, Phase=[np.pi / 4],  PeriBC=[0],SNjunc=['SNS'], ProOn=[1],
                 DateT = '',TimeT = '',SwpID="Vg", deltaPairingMatrix = "sigma_y",deltaPairingMatrix_sign = "+",
                 NextNanoName=None, ReferenceData=None, SaveNameNote=None,Masterfilepath = None,FieldDependentGap = True,Surface2DEG = False,
                 ShowDensity=False, ShowCurrent = False, GetLDOS = False, GetConductance = True, Swave=False, TeV_Normal=True,
                 CombineTev=True, CombineMu=False, ACFix = False, AC = 0, Mapping = False,constantDelta = False,
                 MasterMultiRun=False, BlockWarnings=True, showBands=False,NumBands = 1,CloseSystem = False,mode_Num = 5, k_Num = 10,
                 AddOrbitEffect=True, AddZeemanField = True, AddRashbaSOI = True, AddDresselhausSOI = True, LockFieldAngle = False,
                 PlotbeforeFigures=5,PlotbeforeFigures_Ana = 20):
        self.DavidPot = DavidPot
        self.alpha = alpha
        self.beta = beta
        self.delta_raw = delta
        self.deltaNormalitionFactor = delta / delta_real
        # self.gn = gn  # g-factor
        self.gn_muB = 0.2 / (self.deltaNormalitionFactor)  # this is normalized to EZ = 0.1 with B = 1T
        self.Surface2DEG = Surface2DEG
        self.Temp = Temp
        self.deltaPairingMatrix=deltaPairingMatrix
        self.deltaPairingMatrix_sign=deltaPairingMatrix_sign
        self.CloseSystem = CloseSystem
        self.mode_Num =mode_Num
        self.k_Num =k_Num
        if ACFix:
            if not AC == 0:
                E_excited = AC/Tev
            else:
                print('AC not fixed')

        self.Zeeman = AddZeemanField
        self.RashbaSOI = AddRashbaSOI
        self.DresselhausSOI = AddDresselhausSOI
        self.GetLDOS = GetLDOS
        self.GetConductance = GetConductance
        self.showBands = showBands
        self.FieldDependentGap = FieldDependentGap
        self.NumBands = NumBands

        self.Mapping = Mapping
        self.constantDelta = constantDelta
        self.BlockWarnings = BlockWarnings
        self.ReferenceData = ReferenceData
        if self.ReferenceData != None:
            self.GetReferenceData(self.ReferenceData)
        self.SeriesR = SeriesR
        self.a = a
        self.Nb_d = Nb_d*1e-9
        t_test = 1000*(self.hbar ** 2 / (2 * self.m * (20*1e-9) ** 2))/self.e
        self.Orbit = AddOrbitEffect

        self.DefectNumPer = DefectNumPer

        self.ShowCurrent = ShowCurrent
        self.TeV_Normal = TeV_Normal
        self.CombineMu = CombineMu
        self.CombineTev = CombineTev
        if self.CombineTev == 1:
            Tev_Tunnel = Tev
        if self.CombineMu == 1:
            muN = muSC

        self.PlotbeforeFigures = PlotbeforeFigures
        self.PlotbeforeFigures_Ana = PlotbeforeFigures_Ana

        V_A = np.round(V_A, 5)

        self.SaveNameNote = SaveNameNote
        self.DefectAmp = DefectAmp

        self.fileEnd = '-Kwt'
        self.SwpID = SwpID

        if self.SwpID == 'Vg':
            self.SwpUnit = ' (V)'
        elif self.SwpID == 'Vbias':
            self.SwpUnit = ' (V)'
        elif self.SwpID == 'E':
            if self.Mapping:
                if len(BField)>len(Phase):
                    # Initialize the index of the changing element
                    changing_element_index = None

                    # Iterate through the list to detect the changing element

                    if LockFieldAngle:
                        if self.BlockWarnings:
                            warnings.filterwarnings("ignore")

                        ThetaAngleList = np.round(np.arccos([t[2]/(np.sqrt(t[0]**2+t[1]**2+t[2]**2)) for t in BField])/np.pi,5)
                        PhiAngleList = np.round(np.arcsin([t[1] / (np.sin(a)*np.sqrt(t[0] ** 2 + t[1] ** 2 + t[2] ** 2)) for t,a in zip(BField,ThetaAngleList)])/np.pi,5)
                        if self.BlockWarnings:
                            warnings.filterwarnings("always")
                        PhiAngleList[np.isnan(PhiAngleList)] = 0
                        ThetaAngleList[np.isnan(ThetaAngleList)] = 0

                        for i in range(len(ThetaAngleList)-1):
                            if ThetaAngleList[i] != ThetaAngleList[i+1]:
                                self.VarMaptxt = 'Theta_B (pi)'
                                self.VarMap = ThetaAngleList
                                break
                            elif PhiAngleList[i] != PhiAngleList[i+1]:
                                self.VarMaptxt = 'Phi_B (pi)'
                                self.VarMap = PhiAngleList
                                break

                    else:
                        for i in range(len(BField[0])):
                            if BField[0][i] != BField[1][i]:
                                changing_element_index = i
                                break
                        self.VarMap = [t[changing_element_index] for t in BField]
                        self.VarMaptxt = 'B (T)'
                elif len(Phase)>len(BField):
                    self.VarMap = Phase
                    self.VarMaptxt = 'Phase (rad)'
            self.SwpUnit = ' (meV)'
        elif self.SwpID == 'B':
            self.SwpUnit = ' (T)'
        elif self.SwpID == 'Phase':
            self.SwpUnit = ' (rad)'

        self.ShowDensity = ShowDensity
        self.Swave = Swave
        self.VGate_shift = VGate_shift  # in the unit of number of t(normalised to t), the gate voltage shift when perform David method
        self.GlobalStartTime = time.time()
        self.GlobalRunCount = 0
        self.GlobalVswpCount = 0
        self.u_sl_ref = 0  # (eV/t) initialised reference external potential for the Hamiltonian
        self.TunnelLength = TunnelLength  # (int) the length of the Tunnel barrier in the y direction in the unit of grid point
        self.GridFactor = GridFactor
        # Use David method or import potential from Nextnano calculation

        if self.DavidPot:
            self.GateSplit = int(GridFactor * S_g / self.a)
            self.GateWidth = int(GridFactor * W_g / self.a)
            self.Depth2DEG = int(GridFactor * D_2DEG / self.a)
            self.NextNanoName = 'DavidMethod'
        else:
            if not NextNanoName == None:
                self.NextNanoName = NextNanoName
                self.Dict, self.VgList = SearchFolder(self.NextNanoName, 'bandedges_2d_2DEG_NoBoundary.fld', 'Gamma',Vg_target=V_A, ylim=(-1600, 2400))

        self.OriginFilePath = self.NextNanoName + self.fileEnd + '/OriginFile/'
        if not MasterMultiRun:
            if not os.path.exists(self.OriginFilePath):
                os.makedirs(self.OriginFilePath)
        else:
            if not os.path.exists(self.OriginFilePath+DateT+'-'+TimeT+'/'):
                os.makedirs(self.OriginFilePath+DateT+'-'+TimeT+'/')

        self.W_reduced_r = 180  # (nm) the width that the 2DEG reduced to get NN interface
        self.W_r = W_r
        self.L_r = L_r
        self.WSC_r = WSC
        self.W = int(W_r * GridFactor / self.a)  # (int) width of the junction (y direction)
        self.L = int(L_r * GridFactor / self.a)  # (int) length of the junction (x direction)
        self.L_extract_half = int((L_r - L_s) * GridFactor / (2*self.a))
        self.WSC = int(WSC * GridFactor / self.a)  # (nm)
        now = datetime.now()
        self.Date = now.strftime("%YY%mM%dD")
        self.Time = now.strftime("%Hh%Mm%Ss")
        current_file_path = __file__

        if not MasterMultiRun:
            self.SaveTime = self.Date + '-' + self.Time
            new_file_path = self.NextNanoName + self.fileEnd + '/' + self.Date + '-' + self.Time + '/'
        else:
            self.SaveTime = DateT+'-'+TimeT+'/'+self.Date+'-'+self.Time
            new_file_path = self.NextNanoName + self.fileEnd + '/' + DateT + '-' + TimeT + '/'

        if not os.path.exists(self.OriginFilePath +self.SaveTime+'-LDOS'):
            os.makedirs(self.OriginFilePath +self.SaveTime+'-LDOS')
        if not os.path.exists(new_file_path):
            os.makedirs(new_file_path)

        new_file = new_file_path + 'Kwant_Class.py'
        os.system(f'cp {current_file_path} {new_file}')

        new_file = new_file_path + 'main.py'
        os.system(f'cp {Masterfilepath} {new_file}')

        self.XX = np.arange(0, self.L)
        self.YY = np.arange(0, self.W)



        self.Combine_Change(SNjunc,PeriBC,ProOn,Tev,Tev_Tunnel)


        if self.SwpID == "Vbias":
            self.Combine_Still(muSC,muN,muLead,E_excited,V_A,TStrength,BField,Phase)
            self.VarSwp = Vbias_List
            self.TotalNrun = len(self.comb_change) * len(self.comb_still) * len(Vbias_List)

        elif self.SwpID == "Vg":
            self.Combine_Still(muSC, muN, muLead,E_excited, TStrength, BField,Vbias_List, Phase)
            self.VarSwp = V_A
            self.TotalNrun = len(self.comb_change) * len(self.comb_still) * len(V_A)

        elif self.SwpID == "E":
            self.Combine_Still(muSC, muN, muLead,V_A, TStrength, BField, Vbias_List, Phase)
            self.VarSwp = E_excited
            self.TotalNrun = len(self.comb_change) * len(self.comb_still) * len( E_excited)

        elif self.SwpID == "B":
            self.Combine_Still(muSC, muN,muLead, E_excited, V_A,TStrength, Vbias_List, Phase,[0],[0])
            self.VarSwp = BField
            self.TotalNrun = len(self.comb_change) * len(self.comb_still) * len(BField)

        elif self.SwpID == "Phase":
            self.Combine_Still(muSC, muN,muLead, E_excited,V_A, TStrength, Vbias_List, BField)
            self.VarSwp = Phase
            self.TotalNrun = len(self.comb_change) * len(self.comb_still) * len(Phase)
        if self.CombineMu or not self.SwpID == 'B':
            self.comb_still = list(map(self.Upzip, self.comb_still))
        self.TempDefineHc()
        # self.Run_sweep()
    def Combine_Change(self,SNjunc,PeriBC,ProOn,Tev,Tev_Tunnel):
        if self.CombineTev:
            self.comb_change = list(
                itertools.product(list(SNjunc), list(PeriBC), list(ProOn), zip(list(Tev),list(Tev_Tunnel))))
            self.comb_change = list(map(self.Upzip, self.comb_change))
        else:
            self.comb_change = list(
                itertools.product(list(SNjunc), list(PeriBC), list(ProOn), list(Tev),list(Tev_Tunnel)))

    def Combine_Still(self,V1,V2,V3,V4,V5,V6,V7,V8,V9 = [],V10 = []):
        if self.CombineMu:
            if self.SwpID == 'B':
                self.comb_still = list(
                    itertools.product(zip(V1, V2), V3, V4,
                                      V5, V6, V7, V8,V9,V10))
            else:
                self.comb_still = list(
                    itertools.product(zip(V1, V2), V3, V4,
                                      V5, V6, V7, V8))
        else:
            if self.SwpID == 'B':
                self.comb_still = list(
                    itertools.product(V1, V2, V3,V4,
                                      V5, V6, V7,V8,V9,V10))
            else:
                self.comb_still = list(
                    itertools.product(V1, V2, V3, V4,
                                      V5, V6, V7, V8))

    def GetReferenceData(self, Path):
        self.referdata = pd.read_excel(Path)

    def DiracDelta(self,En, mu, sigma):
        En = np.array(En)
        ans = np.zeros(En.size, dtype=complex)
        lim = 500


        ans[np.real(En - mu) ** 2 / (2 * sigma ** 2) > lim] = 0
        ans[np.real(En - mu) ** 2 / (2 * sigma ** 2) < -lim] = 0
        ans[(np.real(En - mu) ** 2 / (2 * sigma ** 2) <= lim) & (
                np.real(En - mu) ** 2 / (2 * sigma ** 2) >= -lim)] = (1 / np.sqrt(2 * np.pi * sigma ** 2)) * np.exp(
            -np.real(En[(np.real(En - mu) ** 2 / (2 * sigma ** 2) <= lim) & (
                    np.real(En - mu) ** 2 / (2 * sigma ** 2) >= -lim)] - mu) ** 2 / (2 * sigma ** 2))
        return ans

    def Fermi(self, En, mu):

        En = np.array(En)
        ans = np.zeros(En.size, dtype=complex)

        lim = 500
        if self.Temp == 0:
            ans[np.real(En - mu) > 0] = 0
            ans[np.real(En - mu) < 0] = 1
            ans[np.real(En - mu) ==0] = 0.5

        else:
            ans[np.real(En - mu) / (self.kB * self.Temp) > lim] = 0
            ans[np.real(En - mu) / (self.kB * self.Temp) < -lim] = 1
            ans[(np.real(En - mu) / (self.kB * self.Temp) <= lim) & (np.real(En - mu) / (self.kB * self.Temp) >= -lim)] = (
                    1 / (np.exp(np.real(En[(np.real(En - mu) / (self.kB * self.Temp) <= lim) & (
                        np.real(En - mu) / (self.kB * self.Temp) >= -lim)] - mu) / (self.kB * self.Temp)) + 1))
            # ans = 1 / (np.exp((En - mu) / (kB * T)) + 1)
        return ans



    def Upzip(self, a):
        c = []
        e = []

        for b in a:

            try:
                if type(b) == str:
                    e.append(b)
                else:
                    for d in b:
                        e.append(d)
            except:
                e.append(b)
        c = tuple(e)

        return c

    def DavidPotential(self):
        x = np.arange(0, self.L + 1)
        y = np.arange(0, self.W)

        def G(u, v):
            return np.arctan2(u * v, (self.Depth2DEG * np.sqrt(u ** 2 + v ** 2 + self.Depth2DEG ** 2))) / (2 * np.pi)

        X, Y = np.meshgrid(x - max(x) / 2, y - max(y) / 2)
        PHIS = self.V_Applied * (
                    (np.arctan2(self.GateWidth / 2 + Y, self.Depth2DEG) + np.arctan2(self.GateWidth / 2 - Y,
                                                                                     self.Depth2DEG)) / np.pi -
                    G(self.GateSplit / 2 + X, self.GateWidth / 2 + Y) - G(self.GateSplit / 2 + X,
                                                                          self.GateWidth / 2 - Y) -
                    G(self.GateSplit / 2 - X, self.GateWidth / 2 + Y) - G(self.GateSplit / 2 - X,
                                                                          self.GateWidth / 2 - Y))

        # Fv = interp2d(x, y, PHIS, kind='cubic', copy=True, bounds_error=False, fill_value=None)
        self.u_sl = -PHIS.T * self.t  # in eV

    def make_system(self):

        def central_region(site):
            x, y = site.pos
            if self.SN == 'SN':
                # return (0 <= x <= self.L and 0 <= y < self.W) or (self.L_extract_half <= x <= self.L-self.L_extract_half and -self.WSC <= y < 0) or (
                #         self.L_extract_half <= x <= self.L-self.L_extract_half and self.W <= y <=(self.W - int(self.W_reduced_r / self.a)) )
                return 0 <= x < self.L and -self.WSC <= y < (self.W - int(self.W_reduced_r / self.a))
            else:
                # return (0 <= x <= self.L and 0 <= y < self.W) or (self.L_extract_half <= x <= self.L - self.L_extract_half and -self.WSC <= y < 0) or (
                 #             self.L_extract_half <= x <= self.L - self.L_extract_half and self.W <= y <= (self.W + self.WSC))

                return 0 <= x < self.L and -self.WSC <= y < (self.W + self.WSC)

        lat = kwant.lattice.square(norbs=4)
        # if self.SN == 'SN':
        #     lat = kwant.lattice.square(norbs = 4)
        # else:
        #     lat = kwant.lattice.square(norbs = 4)
        # as t is in eV, only the term alpha is not normalised, need to be provide in nature unit
        # other parameter mu V etc, should be in eV unit
        if self.Swave:
            PHMatrix = "sigma_z"






            PHMatrix_sign = "+"
        else:
            PHMatrix = self.deltaPairingMatrix
            PHMatrix_sign = self.deltaPairingMatrix_sign

        if self.SN == "NNN":
            self.Ham = """
                                k_x**2+k_y**2 - (mu(x,y)-VG(x,y))/t
            """
            self.Ham_l_up_S = """
                                        k_x**2+k_y**2 - mu_N/t
                    """
            self.Ham_l_dn_S = """
                                        k_x**2+k_y**2  -mu_N/t
                    """
            self.Ham_l_dn_N = """
                                        k_x**2+k_y**2- mu_N/t
                    """
        else:
            # Hamiltonian in the ubit of actual energy eV/t[eV] [overall unitless], so it start with 4 as onsite energy
            # t is in the unit of eV. t = hbar^2/(2ma^2)
            # mu(V, Vg, TB, Delta) is in eV, so mu/t is the one used in H
            # EZ is in J, so just /(e*t)
            # alpha^2*m is in J need to /(e*t)
            # # The excitation energy is given in the unit of
            # Ham is the one in the scattering region
            TeV_N_Txt = ""
            TeV_NN_Txt = ""
            if self.TeV_Normal:
                TeV_N_Txt = "/t(x,y)"
            else:
                TeV_NN_Txt = "*t(x,y)"

            if self.Orbit:
                OrbitalHam = """
                                   +(((Y_rl(x,y)**2)*Orbital1)*kron(sigma_z, sigma_0)  -
                                   (Y_rl(x,y)*Orbital2)*kron(sigma_0, sigma_0) -
                                   (Y_rl(x,y)*Orbital3)*kron(sigma_0, sigma_y))
                             """ + TeV_N_Txt

            else:
                OrbitalHam = ""

            if self.Zeeman:
                ZeemanHam ="""+ (EZx(x,y)*kron(sigma_0,sigma_x)+EZy(x,y)*kron(sigma_0,sigma_y)+EZz(x,y)*kron(sigma_0,sigma_z))""" + TeV_N_Txt
                ZeemanHam_Fix ="""+ (EZ_x_fix*kron(sigma_0,sigma_x)+EZ_y_fix*kron(sigma_0,sigma_y)+EZ_z_fix*kron(sigma_0,sigma_z))""" + TeV_N_Txt


                # ZeemanHam = """+ 2*(sin_theta*cos_phi*kron(sigma_0,sigma_x)+sin_theta*sin_phi*kron(sigma_0,sigma_y)+cos_theta*kron(sigma_0,sigma_z))"""

            else:
                ZeemanHam = ""

            if self.RashbaSOI:
                # HamPreRashba = """+ (m*alpha**2/(2*e*hbar**2))""" + TeV_N_Txt
                HamPreRashba = """+ ((m*e**2)*(alpha_dis(x,y)**2)/(hbar**2))""" + TeV_NN_Txt
                RashbaHam = """+ alpha_dis(x,y)*(k_x*kron(sigma_0, sigma_y) - k_y*kron(sigma_0, sigma_x))*kron(sigma_z, sigma_0)""" + TeV_N_Txt
                RashbaHam_Fix = """+ alpha_fix*(k_x*kron(sigma_0, sigma_y) - k_y*kron(sigma_0, sigma_x))*kron(sigma_z, sigma_0)""" + TeV_N_Txt

                # RashbaHam = """+ 0.05*1j*(k_x*kron(sigma_0, sigma_y) - k_y*kron(sigma_0, sigma_x))*kron(sigma_z, sigma_0)"""

            else:
                HamPreRashba = ""
                RashbaHam = ""

            if self.DresselhausSOI:
                # HamPreDresselhaus = """+ (m*beta**2/(2*e*hbar**2))""" + TeV_N_Txt
                HamPreDresselhaus = """+ ((m*e**2)*(beta_dis(x,y)**2)/(hbar**2))"""+ TeV_NN_Txt
                DresselhausHam = """+beta_dis(x,y)*(k_x*kron(sigma_0, sigma_x) - k_y*kron(sigma_0, sigma_y))*kron(sigma_z, sigma_0)""" + TeV_N_Txt
                DresselhausHam_Fix = """+beta_fix*(k_x*kron(sigma_0, sigma_x) - k_y*kron(sigma_0, sigma_y))*kron(sigma_z, sigma_0)""" + TeV_N_Txt

            else:
                HamPreDresselhaus = ""
                DresselhausHam = ""

            DeltaHam = """+(Delta_0(x,y)*kron(sigma_x+1j*sigma_y,""" + PHMatrix + """) """ + PHMatrix_sign + """ Delta_0_prime(x,y)*kron(sigma_x-1j*sigma_y,""" + PHMatrix + """))""" + TeV_N_Txt

            # self.Ham = """ ((k_x**2+k_y**2) - 0.25)*kron(sigma_z, sigma_0) """ + ZeemanHam + RashbaHam  + DeltaHam
            # self.Ham = """ ((k_x**2+k_y**2)""" + TeV_NN_Txt + """ - (mu(x,y)+V(x,y)-VG(x,y)-TB(x,y))""" + TeV_N_Txt + HamPreRashba + HamPreDresselhaus + """)*kron(sigma_z, sigma_0) """ + ZeemanHam +RashbaHam+DresselhausHam+DeltaHam+OrbitalHam
            self.Ham = """ ((k_x**2+k_y**2)""" + TeV_NN_Txt + """ - (mu(x,y)+V(x,y)-VG(x,y)-TB(x,y))""" + TeV_N_Txt  + """)*kron(sigma_z, sigma_0) """ + ZeemanHam + RashbaHam + DresselhausHam + DeltaHam + OrbitalHam

            #
            self.Ham_l_dn_S = """
                                           ((k_x**2+k_y**2)""" + TeV_NN_Txt + """ - mu_S"""+ TeV_N_Txt+""")*kron(sigma_z, sigma_0) +
                                           (Delta_SC_dn*kron(sigma_x+1j*sigma_y,""" + PHMatrix + """) """ + PHMatrix_sign + """ Delta_SC_dn_prime*kron(sigma_x-1j*sigma_y,""" + PHMatrix + """))""" + TeV_N_Txt + """
                                       """
            # self.Ham_l_dn_S = """
            #                  ((k_x**2+k_y**2) - 0.25 )*kron(sigma_z, sigma_0) +
            #                   (Delta_SC_dn*kron(sigma_x+1j*sigma_y,""" + PHMatrix + """) + Delta_SC_dn_prime*kron(sigma_x-1j*sigma_y,""" + PHMatrix + """))""" + TeV_N_Txt  + ZeemanHam + RashbaHam +"""
            #                  """


            #self.Ham_l_Se = """ ((k_x**2+k_y**2)""" + TeV_NN_Txt + """ - mu_N""" + TeV_N_Txt + HamPreRashba + HamPreDresselhaus + """)*kron(sigma_z, sigma_0) """ + ZeemanHam +RashbaHam+DresselhausHam+OrbitalHam
            self.Ham_l_Se = """ ((k_x**2+k_y**2)""" + TeV_NN_Txt + """ - mu_N""" + TeV_N_Txt  + """)*kron(sigma_z, sigma_0) """ + ZeemanHam_Fix + RashbaHam_Fix + DresselhausHam_Fix + OrbitalHam
            # self.Ham_l_Se = """ ((k_x**2+k_y**2) - 0.25)*kron(sigma_z, sigma_0) """ + ZeemanHam + RashbaHam

            # self.Ham_l_dn_N = """
            #                                ((k_x**2+k_y**2)""" + TeV_NN_Txt + """ - (mu(x,y)+V(x,y)-VG(x,y)-TB(x,y))""" + TeV_N_Txt + """+ (m*alpha**2/(2*e*hbar**2))""" + TeV_N_Txt + """+ (m*beta**2/(2*e*hbar**2))""" + TeV_N_Txt + """)*kron(sigma_z, sigma_0) +
            #                                (EZ(x,y)/e)*(sin_theta*cos_phi*kron(sigma_0,sigma_x)+sin_theta*sin_phi*kron(sigma_0,sigma_y)+cos_theta*kron(sigma_0,sigma_z))""" + TeV_N_Txt + """ +
            #                                (alpha/e)*(k_x*kron(sigma_0, sigma_y) - k_y*kron(sigma_0, sigma_x))*kron(sigma_z, sigma_0)""" + TeV_N_Txt + """ +
            #                                (beta/e)*(k_x*kron(sigma_0, sigma_x) - k_y*kron(sigma_0, sigma_y))*kron(sigma_z, sigma_0)""" + TeV_N_Txt + """
            #                              """
            self.Ham_l_N_metal = """
                                           ((k_x**2+k_y**2)""" + TeV_NN_Txt + """ - mu_Lead""" + TeV_N_Txt + """)*kron(sigma_z, sigma_0)
                                 """
            # self.Ham_l_N_metal = """
            #                 ((k_x**2+k_y**2) - 0.25)*kron(sigma_z, sigma_0)
            #                       """
            # self.Ham_l_N_metal = """
            #                                           ((k_x**2+k_y**2)""" + TeV_NN_Txt + """ - mu_S""" + TeV_N_Txt + """ + (m*alpha**2/(2*e*hbar**2))""" + TeV_N_Txt + """)*kron(sigma_z, sigma_0)
            #                                 """
            # if self.Orbit == False:  # add magntic field effect or not
            #     self.Ham = """
            #                                ((k_x**2+k_y**2) - (mu(x,y)+V(x,y)-VG(x,y)-TB(x,y))/t(x,y) + m*alpha**2/(2*e*t(x,y)))*kron(sigma_z, sigma_0) +
            #                                EZ(x,y)*kron(sigma_0, sigma_x)/(e*t(x,y)) +
            #                                alpha*(k_x*kron(sigma_0, sigma_y) - k_y*kron(sigma_0, sigma_x))*kron(sigma_z, sigma_0)/(e*t(x,y)) +
            #                                (Delta_0(x,y)*kron(sigma_x+1j*sigma_y,""" + PHMatrix + """) + Delta_0_prime(x,y)*kron(sigma_x-1j*sigma_y,""" + PHMatrix + """))/t(x,y)
            #                           """
            # else:
            #     # self.Ham = """
            #     #                                     ((k_x**2+k_y**2) - (mu(x,y)+V(x,y)-VG(x,y)-TB(x,y))/t )*kron(sigma_z, sigma_0) +
            #     #                                     EZ(x,y)*kron(sigma_0, sigma_x)/(e*t) +
            #     #                                     alpha*(k_x*kron(sigma_0, sigma_y) - k_y*kron(sigma_0, sigma_x))*kron(sigma_z, sigma_0)/(a*e*t) +
            #     #                                     (Delta_0(x,y)*kron(sigma_x+1j*sigma_y,""" + PHMatrix + """) + Delta_0_prime(x,y)*kron(sigma_x-1j*sigma_y,""" + PHMatrix + """))/t +
            #     #                                     ((e * (B**2) * (Y_rl(x,y)**2) * (c**2)/(2*m))*kron(sigma_z, sigma_0) -
            #     #                                     (hbar*B*Y_rl(x,y)*k_x/(m*c))*kron(sigma_0, sigma_0) -
            #     #                                     (alpha*Y_rl(x,y)*B/(hbar*c))*kron(sigma_0, sigma_y))/(t)
            #     #                                 """
            #     # make sure it is in eV /t
            #     self.Ham = """
            #                                ((k_x**2+k_y**2) - (mu(x,y)+V(x,y)-VG(x,y)-TB(x,y))/t(x,y) + m*alpha**2/(2*e*t(x,y)*hbar**2))*kron(sigma_z, sigma_0) +
            #                                EZ(x,y)*kron(sigma_0, sigma_x)/(e*t(x,y)) +
            #                                alpha*(k_x*kron(sigma_0, sigma_y) - k_y*kron(sigma_0, sigma_x))*kron(sigma_z, sigma_0)/(e*t(x,y)) +
            #                                (Delta_0(x,y)*kron(sigma_x+1j*sigma_y,""" + PHMatrix + """) + Delta_0_prime(x,y)*kron(sigma_x-1j*sigma_y,""" + PHMatrix + """))/t(x,y)+
            #                                ((e * (B**2) * (Y_rl(x,y)**2) /(2*m* (c**2)))*kron(sigma_z, sigma_0) -
            #                                (hbar*B*Y_rl(x,y)*k_x/(m*c))*kron(sigma_0, sigma_0) -
            #                                (alpha*Y_rl(x,y)*B/(hbar*c))*kron(sigma_0, sigma_y))/t(x,y)
            #                            """
            #
            # self.Ham_l_up_S = """
            #                                ((k_x**2+k_y**2) - (mu_S + V_bias)/t(x,y) + m*alpha**2/(2*e*t(x,y)*hbar**2))*kron(sigma_z, sigma_0) +
            #                                alpha*(k_x*kron(sigma_0, sigma_y) - k_y*kron(sigma_0, sigma_x))*kron(sigma_z, sigma_0)/(e*t(x,y)) +
            #                                (Delta_SC_up*kron(sigma_x+1j*sigma_y,""" + PHMatrix + """) + Delta_SC_up_prime*kron(sigma_x-1j*sigma_y,""" + PHMatrix + """))/t(x,y)
            #                            """
            # self.Ham_l_dn_S = """
            #                                ((k_x**2+k_y**2) - mu_S/t(x,y) + m*alpha**2/(2*e*t(x,y)*hbar**2))*kron(sigma_z, sigma_0) +
            #                                alpha*(k_x*kron(sigma_0, sigma_y) - k_y*kron(sigma_0, sigma_x))*kron(sigma_z, sigma_0)/(e*t(x,y))  +
            #                                (Delta_SC_dn*kron(sigma_x+1j*sigma_y,""" + PHMatrix + """) + Delta_SC_dn_prime*kron(sigma_x-1j*sigma_y,""" + PHMatrix + """))/t(x,y)
            #                            """
            # self.Ham_l_dn_N = """
            #                                ((k_x**2+k_y**2) - (mu_N - V_ref)/t(x,y) + m*alpha**2/(2*e*t(x,y)*hbar**2))*kron(sigma_z, sigma_0) +
            #                                EZ_fix*kron(sigma_0, sigma_x)/(e*t(x,y)) +
            #                                alpha*(k_x*kron(sigma_0, sigma_y) - k_y*kron(sigma_0, sigma_x))*kron(sigma_z, sigma_0)/(e*t(x,y))
            #                            """
            #

        def make_lead(a=1, t=1.0, W=1):
            # Start with an empty lead with a single square lattice
            lat = kwant.lattice.square(a)

            sym_lead = kwant.TranslationalSymmetry((-a, 0))
            lead = kwant.Builder(sym_lead)

            # build up one unit cell of the lead, and add the hoppings
            # to the next unit cell

            return lead




        template = kwant.continuum.discretize(self.Ham)
        # print(template)
        # template_l_up_S = kwant.continuum.discretize(self.Ham_l_up_S)
        # template_l_dn_S = kwant.continuum.discretize(self.Ham_l_dn_S)

        template_l_up_S = kwant.continuum.discretize(self.Ham_l_N_metal)
        template_l_dn_S = kwant.continuum.discretize(self.Ham_l_N_metal)

        # template_l_dn_N = kwant.continuum.discretize(self.Ham_l_dn_N)
        # print(template)
        template_l_dn_N = kwant.continuum.discretize(self.Ham_l_N_metal)


        sys = kwant.Builder()
        sys.fill(template, central_region, (0, 0));
        if self.CloseSystem:
            sys_close = kwant.Builder()
            sys_close.fill(template, central_region, (0, 0));
        ########################################################################################

        ########################################################################################

        ########################################################################################

        ########################################################################################
        sym1 = kwant.TranslationalSymmetry((0, -1))
        sym2 = kwant.TranslationalSymmetry((0, 1))

        def lead_shape(site):
            (x, y) = site.pos
            return (self.L_extract_half < x < self.L-self.L_extract_half-1)

        lead_up = kwant.Builder(sym1)
        if self.PB == 1 and not self.CloseSystem:
            def lead_shape_PB(site):
                (x, y) = site.pos
                if self.SN == 'SN':
                    return (-self.WSC < y < (self.W - int(self.W_reduced_r / self.a)))
                else:
                    return (-self.WSC < y < (self.W + self.WSC-1))
            def lead_shape_PB_2(site):
                (x, y) = site.pos

                return (0 < x < self.L_extract_half)

            def lead_shape_PB_3(site):
                (x, y) = site.pos

                return ((self.L - self.L_extract_half) < x < self.L-1)

            sym3 = kwant.TranslationalSymmetry((-1, 0))
            sym4 = kwant.TranslationalSymmetry((1, 0))
            template_l_Se = kwant.continuum.discretize(self.Ham_l_Se)
            lead_left = kwant.Builder(sym3)
            lead_right = kwant.Builder(sym4)
            if self.L_extract_half > 0:
                lead_up_PB1 = kwant.Builder(sym1)
                lead_dn_PB1= kwant.Builder(sym2)
                lead_up_PB2 = kwant.Builder(sym1)
                lead_dn_PB2 = kwant.Builder(sym2)
                lead_up_PB1.fill(template_l_Se, lead_shape_PB_2, (0, -self.WSC))
                lead_up_PB2.fill(template_l_Se, lead_shape_PB_3, ((self.L - self.L_extract_half+1), -self.WSC))
                if self.SN == 'SN':
                    lead_dn_PB1.fill(template_l_Se, lead_shape_PB_2, (0, self.W - int(self.W_reduced_r / self.a)))
                    lead_dn_PB2.fill(template_l_Se, lead_shape_PB_3,
                                     ((self.L - self.L_extract_half+1), self.W - int(self.W_reduced_r / self.a)))
                else:
                    lead_dn_PB1.fill(template_l_Se, lead_shape_PB_2, (0, self.W + self.WSC))
                    lead_dn_PB2.fill(template_l_Se, lead_shape_PB_3,
                                     ((self.L - self.L_extract_half+1), self.W + self.WSC))

            lead_left.fill(template_l_Se, lead_shape_PB, (0, int(self.W/2)))
            lead_right.fill(template_l_Se, lead_shape_PB, (int(self.L), int(self.W/2)))



        if self.showBands:
            def lead_shape_test(site):
                (x, y) = site.pos
                return (1 <= x < self.NumBands+1)
            Test_l_dn_S = kwant.continuum.discretize(self.Ham_l_dn_S)
            Test_Ham = kwant.continuum.discretize(self.Ham)
            Test_Metal = kwant.continuum.discretize(self.Ham_l_N_metal)
            self.lead_test = kwant.Builder(sym2)
            self.lead_test_Ham = kwant.Builder(sym2)
            self.lead_test_Metal = kwant.Builder(sym2)

            self.lead_test.fill(Test_l_dn_S, lead_shape_test, (1,1))
            self.lead_test_Ham.fill(Test_Ham, lead_shape_test, (1, 1))
            self.lead_test_Metal.fill(Test_Metal, lead_shape_test, (1, 1))

        lead_up.fill(template_l_up_S, lead_shape, (int(self.L / 2), -self.WSC)) # this is attached at y = negative value
        if self.SN == 'SN':

            lead_dn = kwant.Builder(sym2, conservation_law=-np.kron(self.s_z, self.s_0),
                                    particle_hole=np.kron(self.s_x, self.s_0))
            # lead_dn = kwant.Builder(sym2, conservation_law=-self.s_z, particle_hole=self.s_y)
            # lead_dn = kwant.Builder(sym2)
            lead_dn.fill(template_l_dn_N, lead_shape, (int(self.L / 2), self.W - int(self.W_reduced_r / self.a)))

        else:
            lead_dn = kwant.Builder(sym2)
            lead_dn.fill(template_l_dn_S, lead_shape, (int(self.L / 2), self.W + self.WSC))

        # self.LeadTest = lead_dn


        sys.attach_lead(lead_up)
        sys.attach_lead(lead_dn)
        if self.PB == 1 and not self.CloseSystem:
            sys.attach_lead(lead_left)
            sys.attach_lead(lead_right)
            if  self.L_extract_half>0:
                sys.attach_lead(lead_up_PB1)
                sys.attach_lead(lead_dn_PB1)
                sys.attach_lead(lead_up_PB2)
                sys.attach_lead(lead_dn_PB2)
        syst = sys.finalized()

        if self.CloseSystem:
            syst_close = sys_close.finalized()
        else:
            syst_close = []
        # # kwant.plotter.plot(syst,site_color = 'k',fig_size = (20,10))

        return syst, syst_close
        #

    def density(self, sys, params, lead_nr):

        wf = kwant.wave_function(sys, self.E, params=params, check_hermiticity=True)
        A = wf(lead_nr) # it is 2D array the first axis is the mode number, so need to sum all the mode
        # maxtri = wf.rhs[0]
        # plt.spy(maxtri)
        # plt.show()
        A2 = wf(1 - lead_nr)
        e_up = np.arange(0, len(A[0, :]), 4)
        e_down = np.arange(1, len(A[0, :]), 4)
        h_down = np.arange(2, len(A[0, :]), 4)
        h_up = np.arange(3, len(A[0, :]), 4)
        # Ans = ((A[:, e_up] * np.conj(A[:, h_down]) + A[:, e_down] * np.conj(A[:, h_up])) * np.tanh(
        #     energy * t * e / (2 * kB * T))).sum(axis=0)
        Ans = ((A[:, e_up] * np.conj(A[:, h_down]) + A[:, e_down] * np.conj(A[:, h_up])) * np.tanh(
            self.E * self.t * self.e / (2 * self.kB * self.Temp))).sum(axis=0) + \
              ((A2[:, e_up] * np.conj(A2[:, h_down]) + A2[:, e_down] * np.conj(A2[:, h_up])) * np.tanh(
                  self.E * self.t * self.e / (2 * self.kB * self.Temp))).sum(axis=0)
        self.d_raw = (abs(wf(lead_nr)) ** 2).sum(axis=0)
        if self.ShowCurrent:
            self.Current = kwant.operator.Current(sys, sum=False).bind(params=params)
            for i in range(wf(lead_nr).shape[0]):
                if i == 0:
                    self.CurrentOp = self.Current(wf(lead_nr)[0])
                else:
                    self.CurrentOp = self.CurrentOp + self.Current(wf(lead_nr)[i])

        self.Deltamap = Ans

    def Gen_SaveFileName(self):
        self.PBtxt = 'PB' if self.PB == 1 else 'nPB'
        self.Proximitytxt = 'On' if self.ProximityOn == 1 else 'Off'

        self.SAVEFILENAME_origin = str(self.GlobalVswpCount + 1) + ':' + self.SaveTime
        self.SAVENOTETitle = ["DATE(Y/M/D)", "TIME(h/m/s)", "Ee(meV)", "Bx(T)","By(T)","Bz(T)", "Vg(V)", "VB(V)", "Phase(pi_rad)",
                              "SN-SNS", "PB?", "Proxy?", "muN(meV)",
                              "muS(meV)","muLead(meV)", "t(meV)", "t_tunnelcouple(meV)", "Tl_B(t)", "Defect(t)", "Delta(ueV)", "Note"]

        if self.SwpID == "Vg":
            self.SAVEFILENAME = 'VgSwp'

            self.SAVENOTE_buff = [1000*self.E*self.t/self.deltaNormalitionFactor, self.Bx,self.By,self.Bz, "X", self.Vbias,self.phi / np.pi]
            SAVENOTE_rounded = [np.round(1000*self.E*self.t/self.deltaNormalitionFactor,5),
                                np.round(self.Bx,5),np.round(self.By,5),np.round(self.Bz,5),
                                "X", np.round(self.Vbias,5), np.round(self.phi / np.pi,5)]


        elif self.SwpID == "E":
            self.SAVEFILENAME = 'Eswp'

            self.SAVENOTE_buff = ["X", self.Bx,self.By,self.Bz, self.V_Applied, self.Vbias, self.phi / np.pi]
            SAVENOTE_rounded = ["X", np.round(self.Bx,5), np.round(self.By,5), np.round(self.Bz,5),
                                np.round(self.V_Applied,5), np.round(self.Vbias,5), np.round(self.phi / np.pi, 5)]


        elif self.SwpID == "B":
            self.SAVEFILENAME = 'BSwp'
            self.SAVENOTE_buff = [1000*self.E*self.t/self.deltaNormalitionFactor, "X","X","X",
                                  self.V_Applied, self.Vbias, self.phi / np.pi]
            SAVENOTE_rounded = [np.round(1000 * self.E * self.t / self.deltaNormalitionFactor, 5), "X", "X", "X",
                         np.round(self.V_Applied,5), np.round(self.Vbias,5), np.round(self.phi / np.pi, 5)]


        elif self.SwpID == "Vbias":
            self.SAVEFILENAME = 'VbSwp'
            self.SAVENOTE_buff = [1000*self.E*self.t/self.deltaNormalitionFactor,
                                  self.Bx,self.By,self.Bz, self.V_Applied, "X", self.phi / np.pi]

            SAVENOTE_rounded = [np.round(1000 * self.E * self.t / self.deltaNormalitionFactor, 5),
                             np.round(self.Bx,5), np.round(self.By,5),np.round(self.Bz,5),
                             np.round(self.V_Applied,5), "X", np.round(self.phi / np.pi, 5)]


        elif self.SwpID == "Phase":
            self.SAVEFILENAME = 'PhaseSwp'
            self.SAVENOTE_buff = [1000*self.E*self.t/self.deltaNormalitionFactor, self.Bx,self.By,self.Bz, self.V_Applied, self.Vbias, "X"]
            SAVENOTE_rounded = [np.round(1000 * self.E * self.t / self.deltaNormalitionFactor, 5), np.round(self.Bx,5), np.round(self.By,5),
                             np.round(self.Bz,5), np.round(self.V_Applied,5), np.round(self.Vbias,5), "X"]

        self.SAVEFILENAME = self.NextNanoName + self.fileEnd + '/' + self.SaveTime + '/' + self.SN + '-' + self.PBtxt + '-' + \
                            self.Proximitytxt + '-muN' + str(np.round(self.mu_N * 1e3, 3)) + 'meV-muS' + \
                            str(np.round(self.mu_SC * 1e3, 3)) + 'meV-t' + str(
            np.round(self.t * 1e3, 3)) + 'meV-Tl' + str(self.TunnelStrength) + 't-DF' + str(
            self.DefectAmp) + 't' + self.SAVEFILENAME + '/'

        if self.GlobalVswpCount == 0:
            self.SAVENOTE = np.vstack((self.SAVENOTETitle,
                                       [self.Date, self.Time] + self.SAVENOTE_buff + [self.SN, self.PBtxt,
                                                                                      self.Proximitytxt,
                                                                                      np.round(self.mu_N * 1e3/self.deltaNormalitionFactor, 3), \
                                                                                      np.round(self.mu_SC * 1e3/self.deltaNormalitionFactor, 3),
                                                                                      np.round(self.mu_Lead * 1e3 / self.deltaNormalitionFactor,3),
                                                                                      np.round(self.t * 1e3/self.deltaNormalitionFactor, 3),
                                                                                      np.round(self.t_Tunnel * 1e3/self.deltaNormalitionFactor, 3),
                                                                                      self.TunnelStrength,
                                                                                      self.DefectAmp,
                                                                                      np.round(self.delta_raw * 1e6/self.deltaNormalitionFactor, 3), \
                                                                                      self.SaveNameNote]))
        else:
            self.SAVENOTE = np.vstack((self.SAVENOTE,
                                       [self.Date, self.Time] + self.SAVENOTE_buff + [self.SN, self.PBtxt,
                                                                                      self.Proximitytxt,
                                                                                      np.round(self.mu_N * 1e3/self.deltaNormalitionFactor, 3), \
                                                                                      np.round(self.mu_SC * 1e3/self.deltaNormalitionFactor, 3),
                                                                                      np.round(self.mu_Lead * 1e3 / self.deltaNormalitionFactor,3),
                                                                                      np.round(self.t * 1e3/self.deltaNormalitionFactor, 3),
                                                                                      np.round(self.t_Tunnel * 1e3/self.deltaNormalitionFactor, 3),
                                                                                      self.TunnelStrength,
                                                                                      self.DefectAmp,
                                                                                      np.round(self.delta_raw * 1e6/self.deltaNormalitionFactor, 3), \
                                                                                      self.SaveNameNote]))


        table2 = list(map(str,SAVENOTE_rounded + [self.SN, self.PBtxt, self.Proximitytxt,
                                                     np.round(self.mu_N * 1e3/self.deltaNormalitionFactor, 3), \
                                                     np.round(self.mu_SC * 1e3/self.deltaNormalitionFactor, 3),
                                                     np.round(self.mu_Lead * 1e3 / self.deltaNormalitionFactor, 3),
                                                     np.round(self.t * 1e3/self.deltaNormalitionFactor, 3),
                                                     np.round(self.t_Tunnel * 1e3/self.deltaNormalitionFactor, 3),
                                                     self.TunnelStrength, self.DefectAmp, np.round(self.delta_raw * 1e6/self.deltaNormalitionFactor, 3)]))
        table = [["     Ee(meV)     ", "  Bx(T)  ", "  By(T)  ", "  Bz(T)  ", "  Vg(V)  ", "  VB(V)  ", "  Phi(pi)  ", "SN-SNS", "PB?", "Proxy?", "  muN(meV)  ",
                  "  muS  ","  muL  ", "  t(meV)  ", "  t_tc(meV)  ", "  Tl(t)  ", "  DF(t)  ", "  Delta(ueV)  "], table2]
        Mese = PrettyTable(table[0])
        Mese.add_rows(table[1:])
        result = re.search('\n(.*)\n(.*)\n(.*)', str(Mese))
        self.MeseTitle = result.group(1)
        self.MeseValue = result.group(3)
        # self.MesaTitle =

        if not os.path.exists(self.SAVEFILENAME):
            os.makedirs(self.SAVEFILENAME)


        if self.SaveNameNote != None:
            self.SAVEFILENAME_origin = self.SAVEFILENAME_origin + '-' + self.SaveNameNote
            # self.SAVEFILENAME = self.SAVEFILENAME + self.SaveNameNote + '_'

    def Gen_Site_Plot(self, sys, params):


        local_dos = kwant.ldos(sys, params=params, energy=self.E)  # Calculate local density of state
        pick_electron_up = np.arange(0, len(local_dos), 4)
        self.local_dos = local_dos[pick_electron_up] + local_dos[pick_electron_up+1]  # pickout the electron LDOS
        sites = kwant.plotter.sys_leads_sites(sys, 0)[0]  # Get the site and coordinate to plot
        coords = kwant.plotter.sys_leads_pos(sys, sites)
        self.img, Amin, Amax = kwant.plotter.mask_interpolate(coords, self.d)  # Make colormap
        self.img_LDOS, Amin2, Amax2 = kwant.plotter.mask_interpolate(coords, self.local_dos)  # Make colormap

        if self.BlockWarnings:
            warnings.filterwarnings("ignore")
        self.fig = plt.figure(figsize=(40, 20))
        Ax0 = plt.subplot(3, 4, 1)
        set_size(6, 2, Ax0)
        kwant.plotter.plot(sys, ax=Ax0)
        plt.axis('off')

        Ax1 = plt.subplot(3, 4, 2)
        kwant.plotter.map(sys, self.local_dos, ax=Ax1)
        plt.title('LDOS')
        plt.axis('off')

        Ax2 = plt.subplot(3, 4, 3)
        # kwant.plotter.map(sys, np.abs(self.Deltamap), ax=Ax2)
        pcolor = Ax2.imshow(np.abs(self.SpatialDeltaMap).T)
        plt.title('Order Parameter')
        plt.axis('off')

        Ax3 = plt.subplot(3, 4, 4)
        pcolor = Ax3.imshow(self.Defect_Map.T)
        # cbar = fig0.colorbar(pcolor)
        plt.title('Defect')
        plt.axis('off')

        Ax4 = plt.subplot(3, 4, 5)
        pcolor = Ax4.imshow(self.Delta_abs_Map.T)
        # cbar = fig0.colorbar(pcolor)
        plt.title('Delta_abs')
        plt.axis('off')

        Ax5 = plt.subplot(3, 4, 6)
        pcolor = Ax5.imshow(self.Delta_phase_Map.T)
        # cbar = fig0.colorbar(pcolor)
        plt.title('Delta_phase')
        plt.axis('off')

        Ax6 = plt.subplot(3, 4, 7)
        # kwant.plotter.current(sys, self.CurrentOp, ax=Ax6)
        pcolor = Ax6.imshow(self.gn_Map.T)
        # cbar = fig0.colorbar(pcolor)
        plt.title('gn')
        plt.axis('off')

        Ax7 = plt.subplot(3,4, 8)
        pcolor = Ax7.imshow(self.Potential_Map.T)
        # cbar = fig0.colorbar(pcolor)
        plt.title('VGate')
        plt.axis('off')

        Ax8 = plt.subplot(3,4, 9)
        pcolor = Ax8.imshow(self.Tunnel_Map.T)
        # cbar = fig0.colorbar(pcolor)
        plt.title('Tunnel barrier')
        plt.axis('off')

        Ax8 = plt.subplot(3, 4, 10)
        pcolor = Ax8.imshow(self.alpha_Map.T)
        # cbar = fig0.colorbar(pcolor)
        plt.title('Rasha SOI')
        plt.axis('off')

        Ax8 = plt.subplot(3, 4, 11)
        pcolor = Ax8.imshow(self.beta_Map.T)
        # cbar = fig0.colorbar(pcolor)
        plt.title('Dresselhaus SOI')
        plt.axis('off')


        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=0.9,
                            top=0.9,
                            wspace=0.4,
                            hspace=0.4)

        if self.BlockWarnings:
            warnings.filterwarnings("always")

    def Gen_Ana_Plot(self):

        self.fig = plt.figure(figsize=(14, 11))

        self.fig.suptitle(self.MeseTitle+"\n"+self.MeseValue)
        # self.fig.suptitle("\r{0}".format(self.MeseTitle) + "\n" + "\r{0}".format(self.MeseValue))

        cmap = kwant._common.lazy_import('_plotter')._colormaps.kwant_red
        ax0 = plt.subplot(3, 3, 1)
        base = plt.gca().transData
        rot = Affine2D().rotate_deg(-90)
        ax0.plot(self.img.T[:, int(np.shape(self.img.T)[1] / 4)], transform=rot + base)
        plt.title('Wf up down cut(Gate)')

        ax1 = plt.subplot(3, 3, 2)
        pcolor = ax1.imshow(self.img.T, cmap=cmap)
        ax1.axvline(x=int(np.shape(self.img.T)[1] / 4),linestyle='--')
        ax1.axvline(x=int(np.shape(self.img.T)[1] / 2), linestyle='--')
        ax1.axhline(y=int(np.shape(self.img.T)[0] / 2), linestyle='--')

        cbar = self.fig.colorbar(pcolor)

        ax2 = plt.subplot(3, 3, 3)
        base = plt.gca().transData
        rot = Affine2D().rotate_deg(-90)
        ax2.plot(self.img.T[:, int(np.shape(self.img.T)[1] / 2)], transform=rot + base)
        plt.title('Wf up down cut(noGate)')

        # ax2, aux_ax2 = setup_axes(fig, 222)
        # aux_ax2.plot(img.T[:,int(np.shape(img.T)[1]/2)])

        ax4 = plt.subplot(3, 3, 4)
        # pcolor = ax4.pcolormesh(Potential_Map.T, shading='auto')
        # pcolor = ax5.pcolormesh(Delta_Map.T, shading='auto')
        pcolor = ax4.imshow((2e3*self.Delta_abs_Map/self.deltaNormalitionFactor).T)
        cbar = self.fig.colorbar(pcolor)
        plt.title('Delta(meV)')

        ax5 = plt.subplot(3, 3, 5)
        ax5_2 = ax5.twinx()
        ax5.plot(self.img.T[int(np.shape(self.img.T)[0] / 2), :],color = '#0000FF')
        ax5_2.plot(self.img_LDOS.T[int(np.shape(self.img_LDOS.T)[0] / 2), :],color = '#FF0000')
        ax5.tick_params(axis="y", labelcolor='#0000FF')
        ax5_2.tick_params(axis="y", labelcolor='#FF0000')

        ax5.axvline(x=3*self.L_extract_half, linestyle='--')
        ax5.axvline(x=3*(self.L-self.L_extract_half), linestyle='--')
        plt.title('Wf/LDOS left right cut(Gate)')

        ax6 = plt.subplot(3, 3, 6)
        # pcolor = ax4.pcolormesh(Potential_Map.T, shading='auto')
        pcolor = ax6.imshow(self.Potential_Map.T)
        ax6.axvline(x=int(np.shape(self.Potential_Map.T)[1] / 4), linestyle='--')
        ax6.axvline(x=int(np.shape(self.Potential_Map.T)[1] / 2), linestyle='--')
        cbar = self.fig.colorbar(pcolor)
        plt.title('Potential[eV/t]')

        ax7 = plt.subplot(3, 3, 7)
        ax7.plot((2e3*self.Delta_abs_Map/self.deltaNormalitionFactor).T[:, int(np.shape(self.Delta_abs_Map.T)[1] / 2)])
        ax7.axhline(y=self.Delta_induced, color='r')
        ax7.text(x=0, y=2e6*self.Delta_induced/self.deltaNormalitionFactor, s=str(np.round(2e6 * self.Delta_induced/self.deltaNormalitionFactor, 3)))
        plt.title('Delta up down cut')

        ax8 = plt.subplot(3, 3, 8)
        ax8.plot(self.Potential_Map.T[:, int(np.shape(self.Potential_Map.T)[1] / 4)])
        ax8.plot(self.Potential_Map.T[:, int(3 * np.shape(self.Potential_Map.T)[1] / 4)])
        plt.title('Potential up down cut(Gate)')

        ax9 = plt.subplot(3, 3, 9)
        ax9.plot(self.Potential_Map.T[:, int(np.shape(self.Potential_Map.T)[1] / 2)])
        plt.title('Potential up down cut(no Gate)')

        # self.fig.tight_layout()

    def Gen_Conduct_Plot(self, x, y, Xlabel,Ylabel,y2 = [], subloc=[],Y2label='',initfig = True,figsize = (14,11)):
        if self.BlockWarnings:
            warnings.filterwarnings("ignore")
        if initfig:
            self.fig = plt.figure(figsize=figsize)

        self.fig.suptitle(self.MeseTitle + "\n" + self.MeseValue)
        # self.fig.suptitle("\r{0}".format(self.MeseTitle) + "\n" + "\r{0}".format(self.MeseValue))
        if self.SeriesR == 0:
            if len(subloc) == 0:
                subloc = [1, 1, 1]
        else:
            if len(subloc) == 0:
                subloc = [1, 2, 1]

            ax1 = plt.subplot(subloc[0], subloc[1], subloc[2]+1)
            ax1.plot(x, (1 / (self.SeriesR + 1 / (7.74809173e-5 * np.array(y)))) / 7.74809173e-5,
                     label=str(self.SeriesR) + " Ohm")
            ax1.legend()
            if self.ReferenceData != None:
                ax1.plot(self.referdata.Vg1, self.referdata.G1, self.referdata.Vg2, self.referdata.G2)
            plt.xlabel(Xlabel)
            plt.ylabel(Ylabel)

        ax0 = plt.subplot(subloc[0],subloc[1],subloc[2])
        ax0.plot(x, y, label=Ylabel, color = 'b')
        if not len(y2) == 0:
            ax0.tick_params(axis="y", labelcolor='b')
        if self.SwpID == 'E':
            # ax0.axvline(x=self.delta, color='r')
            # ax0.axvline(x=-self.delta, color='r')
            ax0.axvline(x=self.delta * 2000,  ymin = 0, ymax = 0.1, color='b')
            ax0.axvline(x=-self.delta * 2000, ymin = 0, ymax = 0.1,color='b')
            ax0.text(x=self.Delta_induced * 2/self.t, y=0, s=str(np.round(self.Delta_induced * 1e6, 3)))
            # ax0.axvline(x=self.Delta_induced, color='r')
            # ax0.axvline(x=-self.Delta_induced, color='r')
            ax0.axvline(x=self.Delta_induced * 2000, ymin = 0, ymax = 0.1,color='r')
            ax0.axvline(x=-self.Delta_induced * 2000,ymin = 0, ymax = 0.1, color='r')
        if self.ReferenceData != None:
            ax0.plot(self.referdata.Vg1, self.referdata.G1, self.referdata.Vg2, self.referdata.G2)
        ax0.legend()
        ax0.set_xlim(np.min(x),np.max(x))
        plt.xlabel(Xlabel)
        plt.ylabel(Ylabel)
        if not len(y2) == 0:
            ax0_2 = ax0.twinx()
            ax0_2.plot(x, y2, label=Y2label,color = 'r')
            ax0_2.tick_params(axis="y", labelcolor='r')
            ax0_2.legend()
            plt.xlabel(Xlabel)
            plt.ylabel(Y2label)


        # ax2 = plt.subplot(2, 3, 3)
        # ax2.plot(x, (1 / (1000 + 1 / (7.74809173e-5 * np.array(y)))) / 7.74809173e-5,
        #          label=" 1000 Ohm")
        # if self.ReferenceData != None:
        #     ax2.plot(self.referdata.Vg1, self.referdata.G1, self.referdata.Vg2, self.referdata.G2)
        # ax2.legend()
        # plt.xlabel(Xlabel)
        # plt.ylabel("G/G0[/(2e^2/h)]")
        # ax3 = plt.subplot(2, 3, 4)
        # ax3.plot(x, (1 / (3000 + 1 / (7.74809173e-5 * np.array(y)))) / 7.74809173e-5,
        #          label=" 3000 Ohm")
        # if self.ReferenceData != None:
        #     ax3.plot(self.referdata.Vg1, self.referdata.G1, self.referdata.Vg2, self.referdata.G2)
        # ax3.legend()
        # plt.xlabel(Xlabel)
        # plt.ylabel("G/G0[/(2e^2/h)]")
        # ax4 = plt.subplot(2, 3, 5)
        # ax4.plot(x, (1 / (5000 + 1 / (7.74809173e-5 * np.array(y)))) / 7.74809173e-5,
        #          label=" 5000 Ohm")
        # if self.ReferenceData != None:
        #     ax4.plot(self.referdata.Vg1, self.referdata.G1, self.referdata.Vg2, self.referdata.G2)
        # ax4.legend()
        # plt.xlabel(Xlabel)
        # plt.ylabel("G/G0[/(2e^2/h)]")
        # ax5 = plt.subplot(2, 3, 6)
        # ax5.plot(x, (1 / (8000 + 1 / (7.74809173e-5 * np.array(y)))) / 7.74809173e-5,
        #          label=" 8000 Ohm")
        # if self.ReferenceData != None:
        #     ax5.plot(self.referdata.Vg1, self.referdata.G1, self.referdata.Vg2, self.referdata.G2)
        # ax5.legend()
        # plt.xlabel(Xlabel)
        # plt.ylabel("G/G0[/(2e^2/h)]")
        if self.BlockWarnings:
            warnings.filterwarnings("always")

    def MapCutPlot(self, Map, Title=None, subTitle=None):
        self.fig = plt.figure(figsize=(11, 11))
        self.fig.suptitle(Title)
        cmap = kwant._common.lazy_import('_plotter')._colormaps.kwant_red
        ax0 = plt.subplot(2, 3, 1)
        base = plt.gca().transData
        rot = Affine2D().rotate_deg(-90)
        ax0.plot(Map.T[:, int(np.shape(Map.T)[1] / 4)], transform=rot + base)
        if subTitle != None:
            plt.title(subTitle + ' up down cut(Gate)')

        ax1 = plt.subplot(2, 3, 2)
        pcolor = ax1.imshow(Map.T, cmap=cmap)
        ax1.axhline(y=int(np.shape(Map.T)[0] / 4), color='b', linestyle='--')
        ax1.axhline(y=int(np.shape(Map.T)[0] / 2), color='b', linestyle='--')
        ax1.axvline(x=int(np.shape(Map.T)[1] / 4), color='b', linestyle='--')
        ax1.axvline(x=int(np.shape(Map.T)[1] / 2), color='b', linestyle='--')
        cbar = self.fig.colorbar(pcolor)

        ax2 = plt.subplot(2, 3, 3)
        base = plt.gca().transData
        rot = Affine2D().rotate_deg(-90)
        ax2.plot(Map.T[:, int(np.shape(Map.T)[1] / 2)], transform=rot + base)
        if Title != None:
            plt.title(Title + ' up down cut(noGate)')

        ax5 = plt.subplot(2, 3, 5)
        ax5.plot(Map.T[int(np.shape(Map.T)[0] / 2), :])
        if Title != None:
            plt.title(Title + ' left right cut(Gate)')
        ax6 = plt.subplot(2, 3, 6)
        ax6.plot(Map.T[int(np.shape(Map.T)[0] / 4), :])
        if Title != None:
            plt.title(Title + ' left right cut(noGate)')

    def PrintCheckPHsymmetry(self, SMatrix):
        s_ee = SMatrix.submatrix((1, 0), (1, 0))
        # Hole to hole block
        s_hh = SMatrix.submatrix((1, 1), (1, 1))
        print('s_ee: \n', np.round(s_ee, 3))
        print('s_hh: \n', np.round(s_hh[::-1, ::-1], 3))
        print('s_ee - s_hh^*: \n',
              np.round(s_ee - s_hh[::-1, ::-1].conj(), 3), '\n')
        # Electron to hole block
        s_he = SMatrix.submatrix((1, 1), (1, 0))
        # Hole to electron block
        s_eh = SMatrix.submatrix((1, 0), (1, 1))
        print('s_he: \n', np.round(s_he, 3))
        print('s_eh: \n', np.round(s_eh[::-1, ::-1], 3))
        print('s_he + s_eh^*: \n',
              np.round(s_he + s_eh[::-1, ::-1].conj(), 3))

    def MakeClosePB(self,Ham,Dirc):
        yMatrixLenUnit = 4 * (self.W + 2 * self.WSC)
        xMatrixLenUnit = 4
        xMax = self.L
        Ham_Buff = Ham.toarray()
        def extract_subarray(arr, start_row, end_row, start_col, end_col):
            # Extract a subarray based on the given row and column range
            subarray = []
            for i in range(start_row, end_row + 1):
                subarray_row = arr[i][start_col:end_col + 1]
                subarray.append(subarray_row)
            return subarray

        def replace_subarray(arr, start_row, end_row, start_col, end_col, new_subarray):

            # Get the dimensions of the new subarray7
            if not new_subarray == 0:
                new_rows = len(new_subarray)
                new_cols = len(new_subarray[0])

                # Check if the dimensions of the new subarray match the specified range
                if (end_row - start_row + 1 != new_rows) or (end_col - start_col + 1 != new_cols):
                    raise ValueError("Dimensions of new subarray do not match specified range")

            # Replace the subarray with the new subarray
            for i in range(start_row, end_row + 1):
                for j in range(start_col, end_col + 1):
                    if new_subarray == 0:
                        arr[i][j] = 0
                    else:
                        arr[i][j] = new_subarray[i - start_row][j - start_col]
        if Dirc == 'x':
            A = extract_subarray(Ham_Buff,0,yMatrixLenUnit-1,yMatrixLenUnit,2*yMatrixLenUnit-1)
            B = extract_subarray(Ham_Buff, yMatrixLenUnit, 2 * yMatrixLenUnit - 1,0,yMatrixLenUnit-1 )

            replace_subarray(Ham_Buff,0,yMatrixLenUnit-1,(xMax-1)*yMatrixLenUnit,xMax*yMatrixLenUnit-1,A)
            replace_subarray(Ham_Buff, (xMax-1)*yMatrixLenUnit,xMax*yMatrixLenUnit-1,0,yMatrixLenUnit-1 , B)
        elif Dirc == 'y':
            for i in range(xMax):
                A = extract_subarray(Ham_Buff, i*yMatrixLenUnit, i*yMatrixLenUnit+xMatrixLenUnit - 1, i*yMatrixLenUnit+xMatrixLenUnit, i*yMatrixLenUnit+2 * xMatrixLenUnit - 1)
                B = extract_subarray(Ham_Buff, i*yMatrixLenUnit+xMatrixLenUnit, i*yMatrixLenUnit+2 * xMatrixLenUnit - 1, i*yMatrixLenUnit, i*yMatrixLenUnit+xMatrixLenUnit - 1)

                replace_subarray(Ham_Buff, i*yMatrixLenUnit, i*yMatrixLenUnit+xMatrixLenUnit - 1, (i+1)*yMatrixLenUnit - xMatrixLenUnit, (i+1)*yMatrixLenUnit - 1, A)
                replace_subarray(Ham_Buff, (i+1)*yMatrixLenUnit - xMatrixLenUnit, (i+1)*yMatrixLenUnit - 1, i*yMatrixLenUnit, i*yMatrixLenUnit+xMatrixLenUnit - 1, B)
        elif Dirc == 'xy':
            A = extract_subarray(Ham_Buff, 0, yMatrixLenUnit - 1, yMatrixLenUnit, 2 * yMatrixLenUnit - 1)
            B = extract_subarray(Ham_Buff, yMatrixLenUnit, 2 * yMatrixLenUnit - 1, 0, yMatrixLenUnit - 1)

            replace_subarray(Ham_Buff, 0, yMatrixLenUnit - 1, (xMax - 1) * yMatrixLenUnit, xMax * yMatrixLenUnit - 1, A)
            replace_subarray(Ham_Buff, (xMax - 1) * yMatrixLenUnit, xMax * yMatrixLenUnit - 1, 0, yMatrixLenUnit - 1, B)
            for i in range(xMax):
                A = extract_subarray(Ham_Buff, i*yMatrixLenUnit, i*yMatrixLenUnit+xMatrixLenUnit - 1, i*yMatrixLenUnit+xMatrixLenUnit, i*yMatrixLenUnit+2 * xMatrixLenUnit - 1)
                B = extract_subarray(Ham_Buff, i*yMatrixLenUnit+xMatrixLenUnit, i*yMatrixLenUnit+2 * xMatrixLenUnit - 1, i*yMatrixLenUnit, i*yMatrixLenUnit+xMatrixLenUnit - 1)

                replace_subarray(Ham_Buff, i*yMatrixLenUnit, i*yMatrixLenUnit+xMatrixLenUnit - 1, (i+1)*yMatrixLenUnit - xMatrixLenUnit, (i+1)*yMatrixLenUnit - 1, A)
                replace_subarray(Ham_Buff, (i+1)*yMatrixLenUnit - xMatrixLenUnit, (i+1)*yMatrixLenUnit - 1, i*yMatrixLenUnit, i*yMatrixLenUnit+xMatrixLenUnit - 1, B)

        # plt.spy(Ham_Buff, precision=0.1, markersize=5)
        # plt.show()
        Ham = None
        A = None
        B = None
        return Ham_Buff
    def GaussianDefect(self, FWHM):


        def get_random_numbers(num_numbers,start,end):
            numbers = []

            for _ in range(num_numbers):
                number = np.random.randint(start, end)  # Generate a random number between 10 and 30
                numbers.append(number)

            return numbers
        def makeGaussian(Amp, center, Array=self.Defect_Map, fwhm=FWHM):

            x = np.arange(0, Array.shape[0], 1, float)
            y = np.arange(0, Array.shape[1], 1, float)
            X, Y = np.meshgrid(x, y, indexing='ij')
            x0 = center[0]
            y0 = center[1]

            return Amp * np.exp(-4 * np.log(2) * ((X - x0) ** 2 + (Y - y0) ** 2) / fwhm ** 2)

        array_buf = np.array(self.Defect_Map)
        # DefectPer = 1  # percentage of the sites that contain the defects
        SiteNum = array_buf.shape[0] * array_buf.shape[1]
        DefectNum = int(self.DefectNumPer * SiteNum / 100)
        RandX = get_random_numbers(DefectNum, 0, array_buf.shape[0])
        RandY = get_random_numbers(DefectNum, 0, array_buf.shape[1])
        RandXY =  zip(RandX, RandY)




        array_buf_def = np.zeros((array_buf.shape[0], array_buf.shape[1]))
        array_buf = abs(self.DefectAmp * self.t) * (2 * np.random.rand(array_buf.shape[0], array_buf.shape[1]) - 1)

        for i,j in RandXY:
            array_buf_def = array_buf_def + makeGaussian(Array=array_buf, fwhm=FWHM, center=[i, j],
                                                             Amp=array_buf[i, j])

        # A = array_buf_def/self.t
        # self.fig = plt.figure(figsize=(14, 11))
        # ax4 = plt.subplot(1, 2, 1)
        # # pcolor = ax4.pcolormesh(Potential_Map.T, shading='auto')
        # # pcolor = ax5.pcolormesh(Delta_Map.T, shading='auto')
        # pcolor = ax4.imshow(A.T)
        # cbar = self.fig.colorbar(pcolor)
        # plt.title('Test')
        #
        # ax5 = plt.subplot(1, 2, 2)
        # ax5.plot(A.T[:,int(np.shape(A.T)[1] / 2)])
        # plt.title('Cut View')
        # self.fig.show()
        return array_buf_def

    def DefOutputMap(self):
        self.SpatialDeltaMap = np.zeros((len(self.XX), len(self.YY)), dtype=complex)
        if self.SN == 'SN':
            self.Defect_Map = np.zeros((self.L, self.WSC + self.W - int(self.W_reduced_r / self.a) ))
            self.Potential_Map = np.zeros((self.L, self.WSC + self.W - int(self.W_reduced_r / self.a)))
            self.Delta_abs_Map = np.zeros((self.L, self.WSC + self.W - int(self.W_reduced_r / self.a)))
            self.Delta_phase_Map = np.zeros((self.L, self.WSC + self.W - int(self.W_reduced_r / self.a) ))
            self.Onsite_Map = np.zeros((self.L , self.WSC + self.W - int(self.W_reduced_r / self.a)))
            self.gn_Map = np.zeros((self.L , self.WSC + self.W - int(self.W_reduced_r / self.a)))
            self.alpha_Map = np.zeros((self.L, self.WSC + self.W - int(self.W_reduced_r / self.a)))
            self.beta_Map = np.zeros((self.L, self.WSC + self.W - int(self.W_reduced_r / self.a)))
            self.Vbias_Map = np.zeros((self.L, self.WSC + self.W - int(self.W_reduced_r / self.a)))
            self.Tunnel_Map = np.zeros((self.L , self.WSC + self.W - int(self.W_reduced_r / self.a)))
        else:
            self.Defect_Map = np.zeros((self.L , 2 * self.WSC + self.W ))
            self.Potential_Map = np.zeros((self.L , 2 * self.WSC + self.W))
            self.Delta_abs_Map = np.zeros((self.L , 2 * self.WSC + self.W))
            self.Delta_phase_Map = np.zeros((self.L , 2 * self.WSC + self.W))
            self.Onsite_Map = np.zeros((self.L, 2 * self.WSC + self.W))
            self.gn_Map = np.zeros((self.L, 2 * self.WSC + self.W))
            self.alpha_Map = np.zeros((self.L, 2 * self.WSC + self.W))
            self.beta_Map = np.zeros((self.L, 2 * self.WSC + self.W))
            self.Vbias_Map = np.zeros((self.L, 2 * self.WSC + self.W))
            self.Tunnel_Map = np.zeros((self.L, 2 * self.WSC + self.W))

    def orderDelta(self, X, Y, Bz, lambdaIn, leadN, PHI0, Bx=0, alphaangle=np.pi):
        # Theory based on <Controlled finite momentum pairing and spatially
        # varying order parameter in proximitized HgTe
        # quantum wells>
        # Theory based on <Controlled finite momentum pairing and spatially
        # varying order parameter in proximitized HgTe
        # quantum wells>
        X1 = (self.L_r / 10000) * (np.linspace(-10000, 10000, 40001))
        X_m = (X - self.L / 2) * self.a / self.GridFactor
        Y_m = Y * self.a / self.GridFactor
        PHIJ = PHI0 + ((-1) ** leadN * X1 *0 * (self.W_r) * 1e-18) / (
                2 * self.hbar / (2 * self.e))
        lambda_sp = np.abs(lambdaIn) * ExpRounded(PHIJ)
        Dk = self.e * self.gn_muB * Bx / (self.hbar * self.vF)

        gamma = Dk * np.sqrt((X_m - X1) ** 2 + Y_m ** 2) * 1e-9
        # gamma = Dk * (np.round(np.sin(alphaangle), 15)  * Y_m * 1e-9 + np.round(np.cos(alphaangle), 15) * (X_m - X1) * 1e-9)

        F = (self.m / (8 * np.pi ** 2 * self.hbar)) * (np.round(np.cos(gamma), 15)) / (((X_m - X1) ** 2 + (Y_m) ** 2))
        ORDER = np.trapz(lambda_sp * F, X1)/ (self.Factor)
        A = abs(ORDER)
        X1 = None
        gamma = None
        F = None
        PHIJ = None
        lambda_sp = None

        return ORDER
    def TempDefineHc(self):
        def equation(x, Nb_d, lambda_london):
            return (4 * (x ** 2 - 1) * np.cosh(x * Nb_d / (2 * lambda_london))) / (
                        1 - (lambda_london / (x * Nb_d)) * np.sinh(x * Nb_d / lambda_london)) - (2 - x ** 2) / (
                        1 - (2 * lambda_london / (x * Nb_d)) * np.sinh(x * Nb_d / (2 * lambda_london)))

        solutions = fsolve(equation, [1], args=(self.Nb_d, self.lambda_london))
        A = solutions[0]
        self.Hc = np.sqrt(self.Hcb ** 2 * A ** 2 * (2 - A ** 2) / (
                1 - (2 * self.lambda_london / (A * self.Nb_d)) * np.tanh(A * self.Nb_d / self.lambda_london)))


    def TempDefineGap(self):
        # model in <Magnetic Field Dependence
        # of the Superconducting Energy Gap
        # in Ginzburg-Landau Theory with Application to Al>
        B = np.sqrt( self.Bx**2 + self.By**2 +self.Bz**2 )
        def equation2(A0, Nb_d, lambda_london):
            return B  ** 2 - self.Hcb ** 2 * 4 * A0 ** 2 * (A0 ** 2 - 1) * np.cosh(A0 * Nb_d / (2 * lambda_london)) ** 2 / (
                        1 - (lambda_london / (A0 * Nb_d)) * np.sinh(A0 * Nb_d / lambda_london))

        # initial_guess = np.array([1])
        if self.FieldDependentGap:
            if B <= self.Hc:
                solutions = fsolve(equation2, [1], args=(self.Nb_d, self.lambda_london))
                A2 = solutions[0]
                self.delta = A2 * self.delta_raw
            else:
                self.delta = 0
        else:
            self.delta = self.delta_raw
        # print(1)


    def Run_sweep(self):

        def Delta_0_dis(x, y):

            Square = np.heaviside(y, 1) - np.heaviside(y - self.W, 1)
            Delta_Spatial = self.delta * (1 - np.heaviside(y, 1)) + \
                            self.delta * np.heaviside(y - self.W, 1)

            Phase_Spatial = -self.phi * (1 - np.heaviside(y, 1)) + self.phi * np.heaviside(
                y - self.W, 1)
            DELTA = Delta_Spatial * ExpRounded(Phase_Spatial / 2)
            if self.ProximityOn == 1:

                if (0 <= x < self.L) and (0 <= y < self.W):
                    if self.constantDelta:

                        DELTA = DELTA + self.SpatialDeltaMap[int(x), int(y)] * ExpRounded((-self.phi * 2 * (0.5 - np.heaviside(y- self.W/2, 0))) / 2)* Square
                    else:
                        DELTA = DELTA + self.SpatialDeltaMap[int(x), int(y)] * Square

            if (x < self.L_extract_half or x > self.L-self.L_extract_half):
                DELTA = 0
            self.Delta_abs_Map[int(x), int(y) + self.WSC] = np.abs(DELTA)
            # self.Delta_abs_Map[int(x), int(y) + self.WSC] = np.angle(DELTA)

            self.Delta_phase_Map[int(x), int(y) + self.WSC] = np.angle(DELTA)
            # self.Delta_induced = np.min(self.Delta_abs_Map)
            return DELTA

        def Delta_0_prime_dis(x, y):
            Square = np.heaviside(y, 1) - np.heaviside(y - self.W, 1)
            Delta_Spatial = self.delta * (1 - np.heaviside(y, 1)) + \
                            self.delta * np.heaviside(y - self.W, 1)

            Phase_Spatial = -self.phi * (1 - np.heaviside(y, 1)) + self.phi * np.heaviside(
                y - self.W, 1)



            DELTA = Delta_Spatial * ExpRounded(Phase_Spatial / 2)
            if self.ProximityOn == 1:
                if (0 <= x < self.L) and (0 <= y < self.W):

                    if self.constantDelta:

                        DELTA = DELTA + self.SpatialDeltaMap[int(x), int(y)]* ExpRounded((-self.phi * 2 * (0.5 - np.heaviside(y- self.W/2, 0))) / 2)* Square

                    else:
                        DELTA = DELTA + self.SpatialDeltaMap[int(x), int(y)] *Square
            if (x < self.L_extract_half or x > self.L-self.L_extract_half):
                DELTA = 0

            return np.conjugate(DELTA)

        def EZ_x_dis(x, y):
            if self.Surface2DEG:
                Square = 1
            else:
                Square = np.heaviside(y, 1) - np.heaviside(y - self.W, 1)
            #g = Square*self.gn
            g_muB = Square*self.gn_muB
            self.gn_Map[int(x), int(y) + self.WSC] = np.real(g_muB)
            return np.round(self.deltaNormalitionFactor * g_muB * self.Bx / 2,15)
        def EZ_y_dis(x, y):
            if self.Surface2DEG:
                Square = 1
            else:
                Square = np.heaviside(y, 1) - np.heaviside(y - self.W, 1)
            #g = Square*self.gn
            g_muB = Square*self.gn_muB
            return np.round(self.deltaNormalitionFactor * g_muB * self.By / 2,15)
        def EZ_z_dis(x, y):
            if self.Surface2DEG:
                Square = 1
            else:
                Square = np.heaviside(y, 1) - np.heaviside(y - self.W, 1)
            #g = Square*self.gn
            g_muB = Square*self.gn_muB
            return np.round(self.deltaNormalitionFactor * g_muB * self.Bz / 2,15)
        def alpha_dis(x, y):
            if self.Surface2DEG:
                Square = 1
            else:
                Square = np.heaviside(y, 1) - np.heaviside(y - self.W, 1)
            alpha =Square* self.alpha
            # alpha = self.alpha
            self.alpha_Map[int(x), int(y) + self.WSC] = abs(alpha)
            return alpha

        def beta_dis(x, y):
            if self.Surface2DEG:
                Square = 1
            else:
                Square = np.heaviside(y, 1) - np.heaviside(y - self.W, 1)
            beta =  Square*self.beta
            # beta =  self.beta
            self.beta_Map[int(x), int(y) + self.WSC] = beta
            return beta
        def mu_dis(x, y):
            if self.Surface2DEG:
                Square = 1
            else:
                Square = np.heaviside(y, 1) - np.heaviside(y - self.W, 1)

            AntiSquare = 1 - Square
            MU = Square * (self.mu_N + self.Defect_Map[int(x), int(y)]) + AntiSquare * self.mu_SC
            return MU

        def VGate_dis(x, y):
            Square = np.heaviside(y, 1) - np.heaviside(y - self.W, 1)
            if self.DavidPot:
                try:
                    VGate = self.u_sl[int(x), int(y)]
                    if (abs(x - self.L / 2) > self.GateSplit / 2 and abs(y - self.W / 2) < self.GateWidth / 2):
                        VGate = VGate + self.VGate_shift
                except:
                    VGate = 0
            else:
                VGate = (self.u_sl(x * self.L_r / self.L,
                                   y * self.W_r / self.W) - self.u_sl_ref)
            if np.isnan(VGate):
                VGate = 0
            else:
                VGate = VGate * Square
            VGate = VGate * self.deltaNormalitionFactor
            self.Potential_Map[int(x), int(y) + self.WSC] = VGate

            return VGate

        def V_dis(x, y):

            V = (1 - np.heaviside(y, 1)) * self.Vbias
            self.Vbias_Map[int(x), int(y) + self.WSC] = np.real(V)
            return V

        def TunnelBarrier_dis(x, y):
            TunnelBarrier = 0
            if -np.ceil(self.TunnelLength / 2) <= (y) < np.floor(self.TunnelLength/2) or -np.floor(self.TunnelLength / 2) <=(y - self.W) < np.ceil(self.TunnelLength/2):
                TunnelBarrier = self.GammaTunnel
            self.Tunnel_Map[int(x), int(y) + self.WSC] = np.real(TunnelBarrier)
            return TunnelBarrier

        def t_dis(x, y):
            t = self.t
            if -np.ceil(self.TunnelLength / 2) <= (y) < np.floor(self.TunnelLength/2) or -np.floor(self.TunnelLength / 2) <=(y - self.W)  < np.ceil(self.TunnelLength/2):
                t = self.t_Tunnel
            return t

        def Y_rl_dis(x, y):
            if self.Surface2DEG:
                Square = 1
            else:
                Square = np.heaviside(y, 1) - np.heaviside(y - self.W, 1)

            result = Square * 1e-9 * (y - self.W / 2) * self.a / self.GridFactor

            return result  # in actual nm

        elapsed_tol = 0



        if self.DavidPot:
            self.u_sl_ref_2DEG = 0
        else:

            Index0 = self.VgList.index(0.0)
            u_sl_0 = self.Dict[Index0]
            diffmidLen = (np.max(u_sl_0.points[:, 0]) - np.min(u_sl_0.points[:, 0])) / 2 - self.L_r / 2
            u_sl_0.points[:, 0] = u_sl_0.points[:, 0] - diffmidLen
            self.u_sl_ref_2DEG = u_sl_0(self.L_r / 2, self.W_r - self.W_reduced_r)
            # self.u_sl_ref = u_sl_0(self.L_r / 2, self.W_reduced_r)
            self.u_sl_ref = u_sl_0(self.L_r / 2, 2)
            # TestMap = np.zeros((self.L_r+1,self.W_r+1))
            # for x in range(self.L_r+1):
            #     for y in range(self.W_r+1):
            #         TestMap[x,y] = u_sl_0(x, y)
            # self.fig = plt.figure(figsize=(14, 11))
            # ax4 = plt.subplot(1, 2, 1)
            # # pcolor = ax4.pcolormesh(Potential_Map.T, shading='auto')
            # # pcolor = ax5.pcolormesh(Delta_Map.T, shading='auto')
            # pcolor = ax4.imshow(TestMap.T)
            # cbar = self.fig.colorbar(pcolor)
            # plt.title('Test')
            #
            # ax5 = plt.subplot(1, 2, 2)
            # ax5.plot(TestMap.T[:,int(np.shape(TestMap.T)[1] / 2)])
            # plt.title('Cut View')
            # self.fig.show()

            # print(1)

        # syst.stdout.write(
        #     "\r{0}".format('--------------------------- Start Sweep -----------------------------------'))
        # syst.stdout.flush()
        print("--------------------------- Start Sweep -----------------------------------", end='\r')

        for self.SN, self.PB, self.ProximityOn,self.t, self.t_Tunnel in self.comb_change:




            sys,sys_close = self.make_system()



            self.DefOutputMap()
            self.Defect_Map = self.GaussianDefect(FWHM=1)
            if self.SN == 'SN':
                V_ref_dis = self.u_sl_ref_2DEG
            else:
                V_ref_dis = 0


            VarMap = []
            for Var1, Var2, Var3, Var4, Var5, Var6, Var7, Var8, Var9, Var10 in self.comb_still:
                # self.delta = self.delta_raw
                if self.SwpID == "Vbias":
                    self.mu_SC, self.mu_N, self.mu_Lead,self.E, self.V_Applied, self.TunnelStrength, self.Bx,self.By,self.Bz, self.phi = Var1, Var2, Var3, Var4, Var5, Var6, Var7, Var8, Var9, Var10

                elif self.SwpID == "Vg":
                    self.mu_SC, self.mu_N, self.mu_Lead,self.E, self.TunnelStrength, self.Bx,self.By,self.Bz,  self.Vbias, self.phi = Var1, Var2, Var3, Var4, Var5, Var6, Var7, Var8, Var9, Var10

                elif self.SwpID == "E":
                    self.mu_SC, self.mu_N, self.mu_Lead,self.V_Applied, self.TunnelStrength, self.Bx,self.By,self.Bz,  self.Vbias, self.phi = Var1, Var2, Var3, Var4, Var5, Var6, Var7, Var8, Var9, Var10

                elif self.SwpID == "B":
                    self.mu_SC, self.mu_N, self.mu_Lead,self.E, self.V_Applied, self.TunnelStrength, self.Vbias, self.phi = Var1, Var2, Var3, Var4, Var5, Var6, Var7, Var8

                elif self.SwpID == "Phase":
                    self.mu_SC, self.mu_N, self.mu_Lead,self.E, self.V_Applied, self.TunnelStrength, self.Vbias, self.Bx,self.By,self.Bz = Var1, Var2, Var3, Var4, Var5, Var6, Var7, Var8, Var9, Var10

                if not self.SwpID == "E":

                    if not self.TeV_Normal:
                        self.E = self.E/ self.deltaNormalitionFactor
                    VarSwp_buff2 = [self.E]

                self.GammaTunnel = self.TunnelStrength

                # self.alpha_TB = self.alpha / (2 * self.a)  # eV

                self.conductances = []
                self.conductances2 = []
                if self.GetLDOS:
                    self.LDOS_edge_e_Dn = []
                    self.LDOS_bulk_e_Dn = []
                    self.LDOS_edge_e_Up = []
                    self.LDOS_bulk_e_Up = []
                    self.LDOS_edge_h_Dn = []
                    self.LDOS_bulk_h_Dn = []
                    self.LDOS_edge_h_Up = []
                    self.LDOS_bulk_h_Up = []


                RunCount = 0

                self.Gen_SaveFileName()

                if self.SwpID != 'B':
                    self.TempDefineGap()

                if self.SwpID != 'Vg':
                    if self.DavidPot:
                        self.DavidPotential()
                    else:
                        Index = self.VgList.index(self.V_Applied)
                        self.u_sl = self.Dict[Index]
                        diffmidLen = (np.max(self.u_sl.points[:, 0]) - np.min(
                            self.u_sl.points[:, 0])) / 2 - self.L_r / 2
                        self.u_sl.points[:, 0] = self.u_sl.points[:, 0] - diffmidLen
                        self.u_sl_ref_2DEG = self.u_sl(self.L_r / 2, self.W_r - self.W_reduced_r)
                        # self.u_sl_ref = u_sl_0(self.L_r / 2, self.W_reduced_r)
                        self.u_sl_ref = self.u_sl(self.L_r / 2, 2)

                now = datetime.now()
                DateLocal = now.strftime("%YY%mM%dD")
                TimeLocal = now.strftime("%Hh%Mm%Ss")
                self.LocalSave = DateLocal + '-' + TimeLocal
                VarSwp_buff = self.VarSwp

                if self.SwpID == "E":
                    VarSwp_buff = [0]
                    if not self.TeV_Normal:
                        self.VarSwp = self.VarSwp / self.deltaNormalitionFactor
                    VarSwp_buff2 = self.VarSwp



                for VSwp in VarSwp_buff:
                    if not self.SwpID == "E":
                        TimeBeforeEverySwp = time.time()
                    if self.SwpID == "Vbias":
                        self.Vbias = VSwp

                    elif self.SwpID == "Vg":

                        self.V_Applied = VSwp
                        if self.DavidPot:
                            self.DavidPotential()
                        else:
                            Index = self.VgList.index(self.V_Applied)
                            self.u_sl = self.Dict[Index]
                            diffmidLen = (np.max(self.u_sl.points[:,0]) - np.min(self.u_sl.points[:,0]))/2 - self.L_r / 2
                            self.u_sl.points[:,0] = self.u_sl.points[:,0] - diffmidLen
                            # if self.GlobalRunCount == 0:
                            self.u_sl_ref_2DEG = self.u_sl(self.L_r / 2, self.W_r - self.W_reduced_r)
                            # self.u_sl_ref = u_sl_0(self.L_r / 2, self.W_reduced_r)
                            self.u_sl_ref = self.u_sl(self.L_r / 2, 2)
                    elif self.SwpID == "B":

                        self.Bx, self.By, self.Bz = VSwp

                        self.TempDefineGap()
                    elif self.SwpID == "Phase":

                        self.phi = VSwp

                    self.VStr = str(round(self.V_Applied, 5))



                    # Get the initial reference gate potential and proximity effect factor depend on B
                    if RunCount == 0 or self.SwpID == 'B' or self.SwpID == 'Phase':


                        if self.BlockWarnings:
                            warnings.filterwarnings("ignore")

                        if self.ProximityOn and not self.delta == 0:
                            if self.constantDelta:
                                self.SpatialDeltaMap[:,:] = self.delta
                            else:
                                self.Factor = 1  # correct the proximity effect of order parameter
                                if self.SN == 'SN':
                                    self.Factor = np.abs(self.orderDelta((self.L) / 2, -1,
                                                                         self.Bz,
                                                                         self.delta, 0,
                                                                         -self.phi,
                                                                         Bx=self.Bx) / self.delta)
                                else:
                                    self.Factor = np.abs(self.orderDelta((self.L) / 2, -1,
                                                                         self.Bz,
                                                                         self.delta, 0,
                                                                         -self.phi / 2,
                                                                         Bx=self.Bx) + \
                                                         self.orderDelta((self.L) / 2, self.W+1,
                                                                         self.Bz,
                                                                         self.delta, 1,
                                                                         self.phi / 2,
                                                                         Bx=self.Bx)) / self.delta
                                for i in range(len(self.XX)):
                                    for j in range(len(self.YY)):
                                        if self.SN == 'SN':

                                            self.SpatialDeltaMap[i, j] = self.orderDelta(self.XX[i], self.YY[j]+1,
                                                                                         self.Bz,
                                                                                         self.delta, 0,
                                                                                         self.phi,
                                                                                         Bx=self.Bx)

                                        else:

                                            self.SpatialDeltaMap[i, j] = self.orderDelta(self.XX[i], self.YY[j]+1,
                                                                                         self.Bz,
                                                                                         self.delta, 0,
                                                                                         -self.phi / 2,
                                                                                         Bx=self.Bx) + \
                                                                         self.orderDelta(self.XX[i],
                                                                                         self.W - self.YY[j],
                                                                                         self.Bz,
                                                                                         self.delta, 1,
                                                                                         self.phi / 2,
                                                                                         Bx=self.Bx)
                                # A = np.abs(self.SpatialDeltaMap).T

                        if self.BlockWarnings:
                            warnings.filterwarnings("always")







                    params = dict(a=1e-9, e=self.e, m=self.m, mu_S=self.mu_SC, mu_N=self.mu_N,mu_Lead = self.mu_Lead,hbar=self.hbar,
                                  mu=mu_dis,
                                  EZx=EZ_x_dis,                         EZ_x_fix=np.round(self.deltaNormalitionFactor*self.gn_muB * self.Bx /2,15),
                                  EZy=EZ_y_dis,                         EZ_y_fix=np.round(self.deltaNormalitionFactor * self.gn_muB * self.By / 2, 15),
                                  EZz=EZ_z_dis,                         EZ_z_fix=np.round(self.deltaNormalitionFactor * self.gn_muB * self.Bz / 2, 15),

                                  alpha_dis=alpha_dis,                  alpha_fix = self.alpha,
                                  beta_dis=beta_dis,                    beta_fix = self.beta,
                                  V=V_dis,                              V_bias=self.Vbias,
                                  VG=VGate_dis,                         V_ref=V_ref_dis,
                                  Delta_0=Delta_0_dis,                  Delta_0_prime=Delta_0_prime_dis,
                                  Y_rl=Y_rl_dis,
                                  TB=TunnelBarrier_dis,
                                  t=t_dis,

                                  Delta_SC_up=self.delta * ExpRounded(- self.phi / 2),
                                  Delta_SC_dn=self.delta * ExpRounded(self.phi / 2),
                                  Delta_SC_up_prime=self.delta * ExpRounded(self.phi / 2),
                                  Delta_SC_dn_prime=self.delta * ExpRounded(- self.phi / 2),

                                  Orbital1 = (self.deltaNormalitionFactor*self.e * (self.Bz**2)/(2*self.m)),
                                  Orbital2 = ((self.deltaNormalitionFactor*self.Bz)/ (self.m *np.sqrt(1/(2*self.m*self.e/self.deltaNormalitionFactor)))),
                                  Orbital3 =  ((self.alpha*np.sqrt(self.e/(2*self.m/self.deltaNormalitionFactor))) *self.Bz),
                                  )
# ======================================================================================================================
# ======================================================================================================================

                    monitorX = int(self.L / 2)
                    monitorY_Semi = int(self.W / 2)
                    # monitorY_Semi = 0
                    # monitorY_Semi = self.W
                    monitorY_SC_dn = int(-self.WSC / 2)
                    lat = kwant.lattice.square(norbs=4)
                    self.SemiSiteID = sys.id_by_site[lat(monitorX, monitorY_Semi)]
                    self.SemiSiteXID = sys.id_by_site[lat(monitorX + 1, monitorY_Semi)]
                    self.SemiSiteYID = sys.id_by_site[lat(monitorX, monitorY_Semi + 1)]
                    self.SCdnSiteID = sys.id_by_site[lat(monitorX, monitorY_SC_dn)]
                    self.SCdnSiteXID = sys.id_by_site[lat(monitorX + 1, monitorY_SC_dn)]
                    self.SCdnSiteYID = sys.id_by_site[lat(monitorX, monitorY_SC_dn + 1)]
                    TestHam_Semi = np.array(sys.hamiltonian(self.SemiSiteID, self.SemiSiteID, params=params))
                    TestHopx_Semi = np.array(sys.hamiltonian(self.SemiSiteID, self.SemiSiteXID, params=params))
                    TestHopy_Semi = np.array(sys.hamiltonian(self.SemiSiteID, self.SemiSiteYID, params=params))
                    TestHam_SC_dn = np.array(sys.hamiltonian(self.SCdnSiteID, self.SCdnSiteID, params=params))
                    TestHopx_SC_dn = np.array(sys.hamiltonian(self.SCdnSiteID, self.SCdnSiteXID, params=params))
                    TestHopy_SC_dn = np.array(sys.hamiltonian(self.SCdnSiteID, self.SCdnSiteYID, params=params))
                    result = 1e-9 * (0 - self.W / 2) * self.a / self.GridFactor
                    AA = np.sqrt(self.hbar**2/(2*self.m*self.e/self.deltaNormalitionFactor))
                    ParaDict = {
                        'EZx' : self.deltaNormalitionFactor*self.gn_muB * self.Bx / (2*self.t),
                        'EZy': self.deltaNormalitionFactor * self.gn_muB * self.By / (2 * self.t),
                        'EZz': self.deltaNormalitionFactor * self.gn_muB * self.Bz / (2 * self.t),
                        'alpha' : self.alpha /self.t,
                        'alphaHamTerm': self.m**2*self.alpha**2/(self.hbar**2),
                        'mu_N': self.mu_N/self.t,
                        'mu_Sc': self.mu_SC/self.t,
                        'Delta': self.delta/self.t,
                        'dletaMatrix1': np.kron(self.taux+1j*self.tauy,self.taux)+np.kron(self.taux-1j*self.tauy,self.taux),
                        'dletaMatrix2': np.kron(self.taux + 1j * self.tauy, self.tauy)+np.kron(self.taux - 1j * self.tauy, self.tauy),
                        'dletaMatrix3': np.kron(self.taux + 1j * self.tauy, self.tauz)+np.kron(self.taux - 1j * self.tauy, self.tauz),
                        'dletaMatrix4': np.kron(self.taux + 1j * self.tauy, 1j*self.taux)-np.kron(self.taux - 1j * self.tauy, 1j*self.taux),
                        'dletaMatrix5': np.kron(self.taux + 1j * self.tauy, 1j * self.tauy)-np.kron(self.taux - 1j * self.tauy, 1j * self.tauy),
                        'Orbital1': (result**2) * (self.deltaNormalitionFactor*self.e * (self.Bz**2)/(2*self.m))*np.kron(self.tauz,self.I),
                        'Orbital2': result * ((self.deltaNormalitionFactor*self.Bz)/ (self.m *np.sqrt(1/(2*self.m*self.e/self.deltaNormalitionFactor)))) *np.kron(self.I,self.I),
                        'Orbital3': result * ((self.alpha*np.sqrt(self.e/(2*self.m/self.deltaNormalitionFactor))) * (self.Bz)) *np.kron(self.I,self.tauy),
                    }
# ======================================================================================================================
# ======================================================================================================================
                    if self.CloseSystem:
                        TimeBeforeEverySwp = time.time()
                        ham_mat = sys_close.hamiltonian_submatrix(sparse=True, params=params)

                        if self.PB:
                            sites = kwant.plotter.sys_leads_sites(sys, 0)[0]  # Get the site and coordinate to plot
                            # coords = kwant.plotter.sys_leads_pos(sys, sites)

                            ham_mat = self.MakeClosePB(ham_mat,'y')
                            evals, evecs = eigh(ham_mat)
                        else:
                            evals, evecs = eigh(ham_mat.toarray())
                        # round the eigenvector
                        ham_mat = None
                        # evecs = np.round(evecs.real,4)+np.round(evecs.imag,4)*1j
                        LDOS = np.abs(evecs)**2


                        if self.SwpID == 'E':
                            pick_electron = np.vstack((np.arange(0, len(LDOS), 4), np.arange(1, len(LDOS), 4))).reshape((-1,), order='F')
                            LDOS_Electron =LDOS[pick_electron,:]
                            LDOS_Hole =LDOS[pick_electron+2,:]
                            LDOS_e_Up_list = []
                            LDOS_e_Dn_list = []
                            LDOS_h_Dn_list = []
                            LDOS_h_Up_list = []
                            for E_ID in range(len(VarSwp_buff2)):
                                E = VarSwp_buff2[E_ID]
                                pick_electron_up = np.arange(0, LDOS_Electron.shape[0], 2)

                                GaussianSelection = np.real(self.DiracDelta(evals, E, 0.02 * 2 * self.delta))
                                #GaussianSelectionRev = np.real(self.DiracDelta(-evals, E,0.02*2*self.delta))

                                LDOS_buff = np.sum(LDOS_Electron * GaussianSelection, axis=1)

                                LDOS_e_Up_list.append(list(LDOS_buff[pick_electron_up]))
                                LDOS_e_Dn_list.append(list(LDOS_buff[pick_electron_up + 1]))

                                LDOS_buff = np.sum(LDOS_Hole * GaussianSelection, axis=1)

                                LDOS_h_Dn_list.append(list(LDOS_buff[pick_electron_up]))
                                LDOS_h_Up_list.append(list(LDOS_buff[pick_electron_up + 1]))
                            pick_electron = None
                            pick_electron_up = None
                            LDOS_Electron = None
                            LDOS_Hole = None
                            GaussianSelection = None
                            LDOS_buff = None
                            LDOS = None
                            evecs = None
                            evals = None


                            # TargetEigenValueIndices = np.arange(len(evals) - next(x for x, val in enumerate(reversed(evals)) if val <= np.min(self.VarSwp)) -1,next(x for x, val in enumerate(evals)if val > np.max(self.VarSwp))+1)
                            # self.TotalNrun = len(self.comb_change) * len(self.comb_still) * len(TargetEigenValueIndices)
                            # self.evals = evals[TargetEigenValueIndices]
                            # VarSwp_buff2 = evals[TargetEigenValueIndices]
                            # evecs = evecs[:,TargetEigenValueIndices]

                            # if len(evals) == 0:
                            #     raise('Energy range not valid!')
                    A = self.Tunnel_Map
                    ax1 = plt.subplot(1,2,1)
                    ax1.imshow(self.Tunnel_Map.T)
                    ax2 = plt.subplot(1, 2, 2)
                    ax2.imshow(self.Tunnel_Map[int(np.shape(self.Tunnel_Map)[0]/2),:])
                    plt.show()

                    for E_ID in range(len(VarSwp_buff2)):
                        if self.SwpID == 'E' and not self.CloseSystem:
                            TimeBeforeEverySwp = time.time()
                        RunCount = RunCount + 1
                        self.GlobalRunCount = self.GlobalRunCount + 1

                        self.E = VarSwp_buff2[E_ID]

                        if self.GetConductance:
                            SMatrix = kwant.solvers.default.smatrix(sys, self.E, params=params, out_leads=[0, 1],
                                                                    in_leads=[0, 1])
                        if self.GetLDOS:

                            if self.CloseSystem:


                                if self.SwpID == "E":

                                    LDOS_e_Up = np.array(LDOS_e_Up_list[E_ID])
                                    LDOS_e_Dn = np.array(LDOS_e_Dn_list[E_ID])

                                    LDOS_h_Dn = np.array(LDOS_h_Dn_list[E_ID])
                                    LDOS_h_Up = np.array(LDOS_h_Up_list[E_ID])

                                else:
                                    LDOS = np.abs(evecs[:, self.mode_Num]) ** 2
                                    pick_electron_up = np.arange(0, len(LDOS), 4)
                                    LDOS_e_Up = LDOS[pick_electron_up]
                                    LDOS_e_Dn = LDOS[pick_electron_up + 1]
                                    LDOS_h_Dn = LDOS[pick_electron_up + 2]
                                    LDOS_h_Up = LDOS[pick_electron_up + 3]
                            else:

                                LDOS =  kwant.ldos(sys, params=params, energy=self.E)
                                pick_electron_up = np.arange(0, len(LDOS), 4)
                                LDOS_e_Up = LDOS[pick_electron_up]
                                LDOS_e_Dn = LDOS[pick_electron_up+1]
                                LDOS_h_Dn = LDOS[pick_electron_up+2]
                                LDOS_h_Up = LDOS[pick_electron_up+3]

                            sites = kwant.plotter.sys_leads_sites(sys, 0)[0]  # Get the site and coordinate to plot
                            coords = kwant.plotter.sys_leads_pos(sys, sites)
                            # LDOS, Amin, Amax = kwant.plotter.mask_interpolate(coords, LDOS)
                            def find_coordinates_in_range(arr, x_range, y_range):
                                # Convert the N by 2 array to a NumPy array
                                np_arr = np.array(arr)

                                # Find the indices of coordinates within the specified ranges
                                x_indices = np.where((np_arr[:, 0] >= x_range[0]) & (np_arr[:, 0] <= x_range[1]))
                                y_indices = np.where((np_arr[:, 1] >= y_range[0]) & (np_arr[:, 1] <= y_range[1]))

                                # Find the common indices that satisfy both x and y conditions
                                common_row_indices = np.intersect1d(x_indices, y_indices)

                                return common_row_indices


                            target_X_edge = [self.L_extract_half, self.L_extract_half + 9]
                            target_Y_edge = [0, self.W - 1]
                            target_X_bulk = [int(self.L / 2)-5, int(self.L / 2)+4]
                            target_Y_bulk = [0, self.W - 1]
                            # target_Y_edge = [-self.WSC, self.W + self.WSC]
                            # target_Y_bulk = [-self.WSC, self.W + self.WSC]
                            # target_Y_edge = [int(self.W/4), int(3*self.W/4)]
                            # target_Y_bulk = [int(self.W/4), int(3*self.W/4)]

                            found_row_edge = find_coordinates_in_range(coords, target_X_edge,target_Y_edge)
                            found_row_bulk = find_coordinates_in_range(coords, target_X_bulk,target_Y_bulk)

                            C_edge_e_Up = np.mean(LDOS_e_Up[found_row_edge])
                            C_bulk_e_Up = np.mean(LDOS_e_Up[found_row_bulk])
                            C_edge_e_Dn = np.mean(LDOS_e_Dn[found_row_edge])
                            C_bulk_e_Dn = np.mean(LDOS_e_Dn[found_row_bulk])
                            C_edge_h_Dn = np.mean(LDOS_h_Dn[found_row_edge])
                            C_bulk_h_Dn = np.mean(LDOS_h_Dn[found_row_bulk])
                            C_edge_h_Up = np.mean(LDOS_h_Up[found_row_edge])
                            C_bulk_h_Up = np.mean(LDOS_h_Up[found_row_bulk])

                            self.LDOS_edge_e_Up.append(C_edge_e_Up)
                            self.LDOS_bulk_e_Up.append(C_bulk_e_Up)
                            self.LDOS_edge_e_Dn.append(C_edge_e_Dn)
                            self.LDOS_bulk_e_Dn.append(C_bulk_e_Dn)
                            self.LDOS_edge_h_Dn.append(C_edge_h_Dn)
                            self.LDOS_bulk_h_Dn.append(C_bulk_h_Dn)
                            self.LDOS_edge_h_Up.append(C_edge_h_Up)
                            self.LDOS_bulk_h_Up.append(C_bulk_h_Up)

                        if self.BlockWarnings:
                            warnings.filterwarnings("ignore")
                        self.Delta_induced = np.min(self.Delta_abs_Map.T[:, int(np.shape(self.Delta_abs_Map.T)[1] / 2)])

                        # if RunCount%self.PlotbeforeFigures_Ana == 0 and not self.CloseSystem:
                        if RunCount % self.PlotbeforeFigures_Ana == 0:
                            # try:
                            if self.showBands:
                                fig = kwant.plotter.bands(self.lead_test.finalized(), show=False, params=params)
                                fig.xlabel("momentum [(lattice constant)^-1]")
                                fig.ylabel("energy [t]")
                                fig.title('Superconductor')
                                fig.show()
                                fig = kwant.plotter.bands(self.lead_test_Ham.finalized(), show=False, params=params)
                                fig.xlabel("momentum [(lattice constant)^-1]")
                                fig.ylabel("energy [t]")
                                fig.title('Semiconductor')
                                fig.show()
                                fig = kwant.plotter.bands(self.lead_test_Metal.finalized(), show=False, params=params)
                                fig.xlabel("momentum [(lattice constant)^-1]")
                                fig.ylabel("energy [t]")
                                fig.title('Metal')
                                fig.show()

                            # if self.CloseSystem:
                            #     if self.SwpID == "E":
                            #         WF = np.abs(evecs[:,E_ID]) ** 2
                            #     else:
                            #         WF = np.abs(evecs[:, self.mode_Num]) ** 2
                            #
                            #     self.d = WF[pick_electron_up]+WF[pick_electron_up+1]
                            # else:
                            self.density(sys, params, 1)  # Calculate density
                            pick_electron_up = np.arange(0, len(self.d_raw), 4)  # pickout the electron density part
                            # pickout the electron density part
                            self.d = self.d_raw[pick_electron_up] + self.d_raw[pick_electron_up + 1]


                            self.Gen_Site_Plot(sys, params)
                            self.fig.savefig(self.SAVEFILENAME +self.LocalSave+ '_' +self.SwpID+ str(np.round(VSwp,5)) + "_Sites.png")
                            if self.ShowDensity == 1:
                                self.fig.show()
                            self.Gen_Ana_Plot()
                            self.fig.savefig(self.SAVEFILENAME +self.LocalSave+ '_' +self.SwpID+  str(np.round(VSwp,5)) + "_Ana.png")
                            if self.ShowDensity == 1:
                                self.fig.show()
                            if self.ShowCurrent:
                                self.fig = plt.figure(figsize=(14, 11))
                                Ax0 = plt.subplot(1, 1, 1)
                                kwant.plotter.current(sys, self.CurrentOp,colorbar=False,fig_size = (10, 10),ax = Ax0)
                                # pcolor = Ax6.imshow(self.gn_Map.T)
                                # cbar = fig0.colorbar(pcolor)
                                plt.title('Current')
                                plt.axis('off')
                                if self.ShowDensity == 1:
                                    self.fig.show()
                                self.fig.savefig(self.SAVEFILENAME + self.LocalSave + '_' +self.SwpID+  str(np.round(VSwp,5)) + "_Current.png")
                            # except:
                            #     syst.stdout.write("Site plot not generated")
                            #     syst.stdout.flush()

                        if self.BlockWarnings:
                            warnings.filterwarnings("always")
                        if self.GetConductance:
                            if self.SN == 'SN':
                                # A = SMatrix.submatrix((1, 0), (1, 0))
                                C1 = SMatrix.transmission(1, 0) / 2  # /2 to became 2e^2/h
                                C2 = (SMatrix.submatrix((1, 0), (1, 0)).shape[0] - SMatrix.transmission((1, 0), (
                                    1, 0)) + SMatrix.transmission((1, 1), (1, 0))) / 2

                                self.conductances2.append(C2)
                            else:
                                # A = SMatrix.submatrix((1, 0), (1, 0))
                                C1 = SMatrix.transmission(1, 0) / 2
                                self.conductances2.append('nan')
                            self.conductances.append(C1)


                        elapsed = np.round(time.time() - TimeBeforeEverySwp, 2)
                        elapsed_tol = elapsed_tol + elapsed
                        # Elapsed = TimeFormat(elapsed)
                        LeftRuns = np.round(self.TotalNrun - self.GlobalRunCount, 0)
                        TimeSpend = np.round(time.time() - self.GlobalStartTime, 2)
                        TimeTXT = 'total:' + TimeFormat(TimeSpend) + '/left:' + TimeFormat(
                            LeftRuns * elapsed_tol / self.GlobalRunCount) + ' ' + TimeFormat(
                            elapsed_tol / self.GlobalRunCount) + '/point'
                        if self.CloseSystem:
                            TimeBeforeEverySwp = time.time()
                        if RunCount == 1:
                            if self.GlobalVswpCount == 0:
                                syst.stdout.write("\r{0}".format(self.MeseTitle))
                                syst.stdout.write("\n")
                                syst.stdout.write("\r{0}".format(self.MeseValue))
                                syst.stdout.write("\n")
                            else:

                                syst.stdout.write("\r{0}".format(self.MeseValue))
                                syst.stdout.write("\n")
                        self.ProgressBar(TimeTXT)



                self.GlobalVswpCount = self.GlobalVswpCount + 1
                # print('\n',end ="")
                # syst.stdout.write("\r{0}".format('\n'))
                # syst.stdout.flush()
                # xData = self.VarSwp
                if self.SwpID == "Vbias":
                    TitleTxt1 = ["Vb", "V", "Bias_Voltage"]
                elif self.SwpID == "Vg":
                    TitleTxt1 = ["Vg", "V", "Bias"]
                elif self.SwpID == "E":
                    TitleTxt1 = ["E", "meV", "Excitation_Energy"]
                elif self.SwpID == "B":
                    TitleTxt1 = ["B", "T", "Magnetic_Field"]
                elif self.SwpID == "Phase":
                    TitleTxt1 = ["Theta", "rad", "Phase"]



                if self.GlobalVswpCount%self.PlotbeforeFigures == 0:
                    self.SaveDatatoOrigin(TitleTxt1, Plot=1)
                else:
                    self.SaveDatatoOrigin(TitleTxt1)
            if self.Mapping and self.SwpID == "E" and self.GetLDOS:
                if self.GetConductance:
                    self.LoadDatatoPlot([self.OriginFilePath + self.SaveTime + '.txt'],self.VarSwp,self.VarMap,'G',
                                        ['Conductance'],'V (meV)',self.VarMaptxt,self.SAVEFILENAME + self.LocalSave + "-GMap")

                MapPlotList = [self.OriginFilePath + self.SaveTime + '-LDOS/' + 'LDOS_B_e_Up.txt',
                               self.OriginFilePath + self.SaveTime + '-LDOS/' + 'LDOS_E_e_Up.txt',
                               self.OriginFilePath + self.SaveTime + '-LDOS/' + 'LDOS_B_e_Dn.txt',
                               self.OriginFilePath + self.SaveTime + '-LDOS/' + 'LDOS_E_e_Dn.txt',
                               self.OriginFilePath + self.SaveTime + '-LDOS/' + 'LDOS_B_h_Dn.txt',
                               self.OriginFilePath + self.SaveTime + '-LDOS/' + 'LDOS_E_h_Dn.txt',
                               self.OriginFilePath + self.SaveTime + '-LDOS/' + 'LDOS_B_h_Up.txt',
                               self.OriginFilePath + self.SaveTime + '-LDOS/' + 'LDOS_E_h_Up.txt',
                               self.OriginFilePath + self.SaveTime + '-LDOS/' + 'LDOS_B_h.txt',
                               self.OriginFilePath + self.SaveTime + '-LDOS/' + 'LDOS_E_h.txt',
                               self.OriginFilePath + self.SaveTime + '-LDOS/' + 'LDOS_B_e.txt',
                               self.OriginFilePath + self.SaveTime + '-LDOS/' + 'LDOS_E_e.txt',
                               self.OriginFilePath + self.SaveTime + '-LDOS/' + 'LDOS_B.txt',
                               self.OriginFilePath + self.SaveTime + '-LDOS/' + 'LDOS_E.txt']
                MapPlotTitleList = ['Bulk LDOS e Up','Edge LDOS e Up','Bulk LDOS e Dn','Edge LDOS e Dn',
                                    'Bulk LDOS h Dn','Edge LDOS h Dn','Bulk LDOS h Up','Edge LDOS h Up',
                                    'Bulk LDOS h','Edge LDOS h','Bulk LDOS e','Edge LDOS e','Bulk LDOS','Edge LDOS']

                self.LoadDatatoPlot(MapPlotList,
                                    self.VarSwp,self.VarMap,'LDOS',MapPlotTitleList
                                    ,'V (meV)', self.VarMaptxt,savefilename=
                                    self.SAVEFILENAME + self.LocalSave + "-LDOS_Map",
                           )




        # print('---------------------- All Finished (Total Time:'+TimeFormat(
        #                 TimeSpend)+') ----------------------')
        syst.stdout.write("\r{0}".format('------------------------- All Finished (Total:' + TimeFormat(
            TimeSpend) + ';   Ave:' + TimeFormat(elapsed_tol / self.GlobalRunCount) + '/run) -------------------------'))
        syst.stdout.flush()
    def LoadDatatoPlot(self,datapathlist,xdata,ydata,zlabel,title,xlabel,ylabel,savefilename,):

        # if self.SwpID == 'E':
        #     xdata = 1000 * xdata * self.t

        data_frame = pd.read_csv(datapathlist[0], delimiter='\t')
        zdata_columns = [column for column in data_frame.columns if column.startswith(zlabel)]
        zdata = data_frame[zdata_columns]
        zdata = zdata[2:].astype(float)
        zdata = zdata.to_numpy()

        if len(datapathlist) > 1:
            plotlen = len(datapathlist)
        else:
            plotlen = int(np.size(zdata,1)/len(ydata))
        x0_Index = np.argmin(np.abs(xdata))
        subplotarrayY = int(np.ceil(np.sqrt(plotlen)))
        subplotarrayX = int(np.floor(np.sqrt(plotlen)))
        if subplotarrayY*subplotarrayX<len(datapathlist):
            subplotarrayX = int(subplotarrayX+1)

        if len(datapathlist) == 1:
                # fig = plt.figure(figsize=(12*subplotarrayY, 11*subplotarrayX))
                fig, ax = plt.subplots(subplotarrayY, subplotarrayX,figsize=(6*subplotarrayY, 6*subplotarrayX))
        for i in range(int(np.size(zdata,1)/len(ydata))):
            if len(datapathlist) > 1:
                # fig = plt.figure(figsize=(12*subplotarrayY, 11*subplotarrayX))
                fig, ax = plt.subplots(subplotarrayY, subplotarrayX,figsize=(6*subplotarrayY, 6*subplotarrayX))
                # fig2 = plt.figure(figsize=(24, 11))
                fig2,ax2 = plt.subplots(1,2,figsize=(12, 6))

            for j in range(len(datapathlist)):
                data_frame = pd.read_csv(datapathlist[j], delimiter='\t')
                zdata_columns = [column for column in data_frame.columns if column.startswith(zlabel)]
                zdata = data_frame[zdata_columns]
                zdata = zdata[2:].astype(float)
                zdata = zdata.to_numpy()
                # if len(datapathlist) > 1:
                #     ax = plt.subplot(subplotarrayX, subplotarrayY, j + 1)
                # else:
                #     ax = plt.subplot(subplotarrayX, subplotarrayY, i + 1)
                if len(datapathlist) > 1:

                    ax[int(np.floor(j/ax.shape[1])),int(j%ax.shape[1])].pcolormesh(ydata, xdata, zdata[:, i * len(ydata):(i + 1) * len(ydata)])
                    ax[int(np.floor(j/ax.shape[1])),int(j%ax.shape[1])].title.set_text(title[j] + ' ' + str(i + 1))

                    ax[int(np.floor(j/ax.shape[1])),int(j%ax.shape[1])].set_xlabel(ylabel, fontsize=15)
                    ax[int(np.floor(j/ax.shape[1])),int(j%ax.shape[1])].set_ylabel(xlabel, fontsize=15)
                    ax[int(np.floor(j/ax.shape[1])),int(j%ax.shape[1])].tick_params(axis='x', labelsize=12)
                    ax[int(np.floor(j/ax.shape[1])),int(j%ax.shape[1])].tick_params(axis='y', labelsize=12)
                else:
                    ax[int(np.floor(i/ax.shape[1])),int(i%ax.shape[1])].pcolormesh(ydata, xdata, zdata[:, i * len(ydata):(i + 1) * len(ydata)])
                    ax[int(np.floor(i/ax.shape[1])),int(i%ax.shape[1])].title.set_text(title[j] + ' ' + str(i + 1))

                    ax[int(np.floor(i/ax.shape[1])),int(i%ax.shape[1])].set_xlabel(ylabel, fontsize=15)
                    ax[int(np.floor(i/ax.shape[1])),int(i%ax.shape[1])].set_ylabel(xlabel, fontsize=15)
                    ax[int(np.floor(i/ax.shape[1])),int(i%ax.shape[1])].tick_params(axis='x', labelsize=12)
                    ax[int(np.floor(i/ax.shape[1])),int(i%ax.shape[1])].tick_params(axis='y', labelsize=12)

                if datapathlist[j].split("/")[-1] == 'LDOS_B.txt' or datapathlist[j].split("/")[-1] == 'LDOS_E.txt':

                    if datapathlist[j].split("/")[-1] == 'LDOS_B.txt':
                        ax2[0] = plt.subplot(1, 2, 1)
                        ax2[0].plot(ydata, zdata[x0_Index, i * len(ydata):(i + 1) * len(ydata)])
                        ax2[0].set_xlabel(ylabel, fontsize=15)
                        ax2[0].set_ylabel('LDOS_B', fontsize=15)
                    if datapathlist[j].split("/")[-1] == 'LDOS_E.txt':
                        ax2[1] = plt.subplot(1, 2, 2)
                        ax2[1].plot(ydata, zdata[x0_Index, i * len(ydata):(i + 1) * len(ydata)])
                        ax2[1].set_xlabel(ylabel, fontsize=15)
                        ax2[1].set_ylabel('LDOS_E', fontsize=15)
            if len(datapathlist) > 1:
                fig.subplots_adjust(left=0.05,
                                    bottom=0.05,
                                    right=0.95,
                                    top=0.95,
                                    wspace=0.3,
                                    hspace=0.2)
                fig.savefig(savefilename+str(i)+'.png')
                fig.show()



            fig2.savefig(savefilename + str(i) +'_cut.png')
            fig2.show()


        if len(datapathlist) == 1:
            fig.subplots_adjust(left=0.05,
                    bottom=0.05,
                    right=0.95,
                    top=0.95,
                    wspace=0.3,
                    hspace=0.2)
            fig.savefig(savefilename + str(i) + '.png')
            fig.show()


    def SaveDatatoOrigin(self, TitleTxtX, Plot=0):


        DataState = pd.DataFrame(self.SAVENOTE)
        DataState.to_excel(
            self.OriginFilePath + self.SaveTime + '-SwpDetail.xlsx',
            index=False, header=False)

        Xdata = self.VarSwp
        # if self.SwpID == 'E':
        #     Xdata = 1000 * self.VarSwp * self.t
        # else:
        #     Xdata = self.VarSwp
        if self.GetConductance:
            TitleTxtY1 = ["G", "2e^2/h", self.SAVEFILENAME_origin + '_Conductance']
            Data = [list(a) for a in zip(TitleTxtX + list(Xdata), TitleTxtY1 + list(self.conductances))]
        if self.GetLDOS:
            TitleTxtY_LDOS_e = ["LDOS", " ", self.SAVEFILENAME_origin + '_Edge']
            TitleTxtY_LDOS_b = ["LDOS", " ", self.SAVEFILENAME_origin + '_Bulk']



            Data_LDOS_edge_e_Up = [list(a) for a in zip(TitleTxtX + list(Xdata), TitleTxtY_LDOS_e + list(self.LDOS_edge_e_Up))]
            Data_LDOS_bulk_e_Up = [list(a) for a in zip(TitleTxtX + list(Xdata), TitleTxtY_LDOS_b + list(self.LDOS_bulk_e_Up))]
            Data_LDOS_edge_e_Dn = [list(a) for a in zip(TitleTxtX + list(Xdata), TitleTxtY_LDOS_e + list(self.LDOS_edge_e_Dn))]
            Data_LDOS_bulk_e_Dn = [list(a) for a in zip(TitleTxtX + list(Xdata), TitleTxtY_LDOS_b + list(self.LDOS_bulk_e_Dn))]
            Data_LDOS_edge_h_Dn = [list(a) for a in zip(TitleTxtX + list(Xdata), TitleTxtY_LDOS_e + list(self.LDOS_edge_h_Dn))]
            Data_LDOS_bulk_h_Dn = [list(a) for a in zip(TitleTxtX + list(Xdata), TitleTxtY_LDOS_b + list(self.LDOS_bulk_h_Dn))]
            Data_LDOS_edge_h_Up = [list(a) for a in zip(TitleTxtX + list(Xdata), TitleTxtY_LDOS_e + list(self.LDOS_edge_h_Up))]
            Data_LDOS_bulk_h_Up = [list(a) for a in zip(TitleTxtX + list(Xdata), TitleTxtY_LDOS_b + list(self.LDOS_bulk_h_Up))]
            Data_LDOS_edge_e = [list(a) for a in zip(TitleTxtX + list(Xdata),
                                                 TitleTxtY_LDOS_e + list(np.array(self.LDOS_edge_e_Up) + np.array(self.LDOS_edge_e_Dn)))]
            Data_LDOS_bulk_e = [list(a) for a in zip(TitleTxtX + list(Xdata),
                                                 TitleTxtY_LDOS_e + list(np.array(self.LDOS_bulk_e_Up) + np.array(self.LDOS_bulk_e_Dn)))]
            Data_LDOS_edge_h = [list(a) for a in zip(TitleTxtX + list(Xdata),
                                                 TitleTxtY_LDOS_e + list(np.array(self.LDOS_edge_h_Up) + np.array(self.LDOS_edge_h_Dn)))]
            Data_LDOS_bulk_h = [list(a) for a in zip(TitleTxtX + list(Xdata),
                                                 TitleTxtY_LDOS_e + list(np.array(self.LDOS_bulk_h_Up) + np.array(self.LDOS_bulk_h_Dn)))]


            Data_LDOS_edge = [list(a) for a in zip(TitleTxtX + list(Xdata),
                                                     TitleTxtY_LDOS_e + list(np.array(self.LDOS_edge_h_Up) + np.array(
                                                         self.LDOS_edge_h_Dn)+np.array(self.LDOS_edge_e_Up) + np.array(
                                                         self.LDOS_edge_e_Dn)))]
            Data_LDOS_bulk = [list(a) for a in zip(TitleTxtX + list(Xdata),
                                                     TitleTxtY_LDOS_e + list(np.array(self.LDOS_bulk_h_Up) + np.array(
                                                         self.LDOS_bulk_h_Dn)+np.array(self.LDOS_bulk_e_Up) + np.array(
                                                         self.LDOS_bulk_e_Dn)))]

        if self.SeriesR != 0 and self.GetConductance:
            if self.BlockWarnings:
                warnings.filterwarnings("ignore")
            D_R = (1 / (self.SeriesR + 1 / (7.74809173e-5 * np.array(self.conductances)))) / 7.74809173e-5
            if self.BlockWarnings:
                warnings.filterwarnings("always")
            Data_R = [list(a) for a in zip(TitleTxtX + list(Xdata), TitleTxtY1 + list(D_R))]


        if self.GlobalVswpCount == 1:
            if self.GetConductance:
                savedata(self.OriginFilePath + self.SaveTime + '.txt',
                     init=True, initdata=Data)
            if self.GetLDOS:


                savedata(self.OriginFilePath + self.SaveTime+'-LDOS/' + 'LDOS_E_e_Up.txt',
                         init=True, initdata=Data_LDOS_edge_e_Up)
                savedata(self.OriginFilePath + self.SaveTime +'-LDOS/' + 'LDOS_B_e_Up.txt',
                         init=True, initdata=Data_LDOS_bulk_e_Up)
                savedata(self.OriginFilePath + self.SaveTime + '-LDOS/' + 'LDOS_E_e_Dn.txt',
                         init=True, initdata=Data_LDOS_edge_e_Dn)
                savedata(self.OriginFilePath + self.SaveTime + '-LDOS/' + 'LDOS_B_e_Dn.txt',
                         init=True, initdata=Data_LDOS_bulk_e_Dn)
                savedata(self.OriginFilePath + self.SaveTime + '-LDOS/' + 'LDOS_E_h_Dn.txt',
                         init=True, initdata=Data_LDOS_edge_h_Dn)
                savedata(self.OriginFilePath + self.SaveTime + '-LDOS/' + 'LDOS_B_h_Dn.txt',
                         init=True, initdata=Data_LDOS_bulk_h_Dn)
                savedata(self.OriginFilePath + self.SaveTime + '-LDOS/' + 'LDOS_E_h_Up.txt',
                         init=True, initdata=Data_LDOS_edge_h_Up)
                savedata(self.OriginFilePath + self.SaveTime + '-LDOS/' + 'LDOS_B_h_Up.txt',

                         init=True, initdata=Data_LDOS_bulk_h_Up)
                savedata(self.OriginFilePath + self.SaveTime + '-LDOS/' + 'LDOS_E_e.txt',
                         init=True, initdata=Data_LDOS_edge_e)
                savedata(self.OriginFilePath + self.SaveTime + '-LDOS/' + 'LDOS_E_h.txt',
                         init=True, initdata=Data_LDOS_edge_h)
                savedata(self.OriginFilePath + self.SaveTime + '-LDOS/' + 'LDOS_B_e.txt',
                         init=True, initdata=Data_LDOS_bulk_e)
                savedata(self.OriginFilePath + self.SaveTime + '-LDOS/' + 'LDOS_B_h.txt',
                         init=True, initdata=Data_LDOS_bulk_h)

                savedata(self.OriginFilePath + self.SaveTime + '-LDOS/' + 'LDOS_E.txt',
                         init=True, initdata=Data_LDOS_edge)
                savedata(self.OriginFilePath + self.SaveTime + '-LDOS/' + 'LDOS_B.txt',
                         init=True, initdata=Data_LDOS_bulk)

            if self.SeriesR != 0 and self.GetConductance:
                savedata(self.OriginFilePath + self.SaveTime + '-' + str(
                    round(self.SeriesR, 3)) + '.txt', init=True, initdata=Data_R)
        else:
            if self.GetConductance:
                savedata(self.OriginFilePath + self.SaveTime + '.txt',
                     init=False, newcol=Data)
            if self.GetLDOS:
                savedata(self.OriginFilePath + self.SaveTime + '-LDOS/' + 'LDOS_E_e_Up.txt',
                         init=False, newcol=Data_LDOS_edge_e_Up)
                savedata(self.OriginFilePath + self.SaveTime + '-LDOS/' + 'LDOS_B_e_Up.txt',
                         init=False, newcol=Data_LDOS_bulk_e_Up)
                savedata(self.OriginFilePath + self.SaveTime + '-LDOS/' + 'LDOS_E_e_Dn.txt',
                         init=False, newcol=Data_LDOS_edge_e_Dn)
                savedata(self.OriginFilePath + self.SaveTime + '-LDOS/' + 'LDOS_B_e_Dn.txt',
                         init=False, newcol=Data_LDOS_bulk_e_Dn)
                savedata(self.OriginFilePath + self.SaveTime + '-LDOS/' + 'LDOS_E_h_Dn.txt',
                         init=False, newcol=Data_LDOS_edge_h_Dn)
                savedata(self.OriginFilePath + self.SaveTime + '-LDOS/' + 'LDOS_B_h_Dn.txt',
                         init=False, newcol=Data_LDOS_bulk_h_Dn)
                savedata(self.OriginFilePath + self.SaveTime + '-LDOS/' + 'LDOS_E_h_Up.txt',
                         init=False, newcol=Data_LDOS_edge_h_Up)
                savedata(self.OriginFilePath + self.SaveTime + '-LDOS/' + 'LDOS_B_h_Up.txt',
                         init=False, newcol=Data_LDOS_bulk_h_Up)
                savedata(self.OriginFilePath + self.SaveTime + '-LDOS/' + 'LDOS_E_e.txt',
                         init=False, newcol=Data_LDOS_edge_e)
                savedata(self.OriginFilePath + self.SaveTime + '-LDOS/' + 'LDOS_E_h.txt',
                         init=False, newcol=Data_LDOS_edge_h)
                savedata(self.OriginFilePath + self.SaveTime + '-LDOS/' + 'LDOS_B_e.txt',
                         init=False, newcol=Data_LDOS_bulk_e)
                savedata(self.OriginFilePath + self.SaveTime + '-LDOS/' + 'LDOS_B_h.txt',
                         init=False, newcol=Data_LDOS_bulk_h)

                savedata(self.OriginFilePath + self.SaveTime + '-LDOS/' + 'LDOS_E.txt',
                         init=False, newcol=Data_LDOS_edge)
                savedata(self.OriginFilePath + self.SaveTime + '-LDOS/' + 'LDOS_B.txt',
                         init=False, newcol=Data_LDOS_bulk)
            if self.SeriesR != 0 and self.GetConductance:
                savedata(self.OriginFilePath + self.SaveTime + '-' + str(
                    round(self.SeriesR, 3)) + '.txt', init=False, newcol=Data_R)
        if self.GetConductance:
            self.Gen_Conduct_Plot(Xdata, self.conductances, self.SwpID+self.SwpUnit,"G/G0[/(2e^2/h)]")
            self.fig.savefig(self.SAVEFILENAME +self.LocalSave+ "-Conductance.png")
            if Plot == 1:
                self.fig.show()
        if self.GetLDOS:
            self.Gen_Conduct_Plot(Xdata, self.LDOS_edge_e_Up, self.SwpID + self.SwpUnit,"LDOS_E_e_Up",
                                  y2 = self.LDOS_bulk_e_Up,Y2label="LDOS_B_e_Up",subloc=[2,4,1],figsize = (25,11))

            self.Gen_Conduct_Plot(Xdata, self.LDOS_edge_e_Dn, self.SwpID + self.SwpUnit, "LDOS_E_e_Dn",
                                  y2=self.LDOS_bulk_e_Dn, Y2label="LDOS_B_e_Dn", subloc=[2,4,2],initfig = False)

            self.Gen_Conduct_Plot(Xdata, self.LDOS_edge_h_Dn, self.SwpID + self.SwpUnit, "LDOS_E_h_Dn",
                                  y2=self.LDOS_bulk_h_Dn, Y2label="LDOS_B_h_Dn", subloc=[2,4,3],initfig = False)

            self.Gen_Conduct_Plot(Xdata, self.LDOS_edge_h_Up, self.SwpID + self.SwpUnit, "LDOS_E_h_Up",
                                  y2=self.LDOS_bulk_h_Up, Y2label="LDOS_B_h_Up", subloc=[2,4,4],initfig = False)

            self.Gen_Conduct_Plot(Xdata, np.array(self.LDOS_edge_e_Up) + np.array(self.LDOS_edge_e_Dn),
                                  self.SwpID + self.SwpUnit, "LDOS_E_e",
                                  y2=np.array(self.LDOS_bulk_e_Up) + np.array(self.LDOS_bulk_e_Dn), Y2label="LDOS_B_e",
                                  subloc=[2,4,5],initfig = False)

            self.Gen_Conduct_Plot(Xdata, np.array(self.LDOS_edge_h_Up) + np.array(self.LDOS_edge_h_Dn),
                                  self.SwpID + self.SwpUnit, "LDOS_E_h",
                                  y2=np.array(self.LDOS_bulk_h_Up) + np.array(self.LDOS_bulk_h_Dn), Y2label="LDOS_B_h",
                                  subloc=[2,4,6],initfig = False)

            self.Gen_Conduct_Plot(Xdata, np.array(self.LDOS_edge_h_Up) + np.array(self.LDOS_edge_h_Dn) + np.array(
                                      self.LDOS_edge_e_Up) + np.array(self.LDOS_edge_e_Dn),
                                  self.SwpID + self.SwpUnit, "LDOS_E",
                                  y2=np.array(self.LDOS_bulk_h_Up) + np.array(self.LDOS_bulk_h_Dn) + np.array(
                self.LDOS_bulk_e_Up) + np.array(self.LDOS_bulk_e_Dn), Y2label="LDOS_B",
                                  subloc=[2,4,7],initfig = False)


            self.fig.subplots_adjust(left=0.05,
                                bottom=0.05,
                                right=0.95,
                                top=0.9,
                                wspace=0.4,
                                hspace=0.2)
            self.fig.savefig(self.SAVEFILENAME + self.LocalSave + "-LDOS.png")


            if Plot == 1:
                self.fig.show()


        if self.SN == 'SN' and self.GetConductance:

            TitleTxtY2 = ["G", "2e^2/h", self.SAVEFILENAME_origin + '_N-Ree+Reh']

            Data_2 = [list(a) for a in zip(TitleTxtX + list(Xdata), TitleTxtY2 + list(self.conductances2))]

            if self.SeriesR != 0:
                if self.BlockWarnings:
                    warnings.filterwarnings("ignore")

                D_R2 = (1 / (self.SeriesR + 1 / (7.74809173e-5 * np.array(self.conductances2)))) / 7.74809173e-5
                if self.BlockWarnings:
                    warnings.filterwarnings("always")

                Data_R_2 = [list(a) for a in zip(TitleTxtX + list(Xdata), TitleTxtY2 + list(D_R2))]


            if self.GlobalVswpCount == 1:

                savedata(self.OriginFilePath + self.SaveTime + '_N-Ree+Reh.txt',
                         init=True, initdata=Data_2)
                if self.SeriesR != 0:
                    savedata(self.OriginFilePath + self.SaveTime + '-' + str(
                            round(self.SeriesR, 3)) + '_N-Ree+Reh.txt', init=True, initdata=Data_R_2)
            else:

                savedata(self.OriginFilePath + self.SaveTime + '_N-Ree+Reh.txt',
                         init=False, newcol=Data_2)
                if self.SeriesR != 0:
                    savedata(self.OriginFilePath + self.SaveTime + '-' + str(
                        round(self.SeriesR, 3)) + '_N-Ree+Reh.txt', init=False, newcol=Data_R_2)

            if self.SwpID == 'E':
                self.Gen_Conduct_Plot(1000*self.VarSwp*self.t, self.conductances2, self.SwpID+self.SwpUnit,"G/G0[/(2e^2/h)]")
            else:
                self.Gen_Conduct_Plot(self.VarSwp, self.conductances2, self.SwpID+self.SwpUnit,"G/G0[/(2e^2/h)]")
            self.fig.savefig(self.SAVEFILENAME+self.LocalSave + "-N-Ree+Reh.png")
            if Plot == 1:
                self.fig.show()




    def clear_line(self, n=1):
        LINE_UP = '\033[1A'
        LINE_CLEAR = '\x1b[2K'
        for i in range(n):
            syst.stdout.write(LINE_UP + LINE_CLEAR)

    def ProgressBar(self, TimeTXT, res=2 ):
        percentage = np.round((self.GlobalRunCount / self.TotalNrun) * 100, 5)
        percentage_rounded = int((self.GlobalRunCount / self.TotalNrun) * 100 / res)
        if self.GlobalRunCount != 1:
            syst.stdout.write('\r')
        syst.stdout.write("{:8.4f}".format(percentage))
        syst.stdout.write('% ')
        for i in range(int(100 / res) + 1):

            if i <= percentage_rounded:

                syst.stdout.write('#')
            else:

                syst.stdout.write('-')

        syst.stdout.write(' ' + TimeTXT)
        # syst.stdout.flush()

    def list_duplicates_of(self, seq, item):
        start_at = -1
        locs = []
        while True:
            try:
                loc = seq.index(item, start_at + 1)
            except ValueError:
                break
            else:
                locs.append(loc)
                start_at = loc
        return locs

