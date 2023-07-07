import os
import csv
from os import system, name
from scipy.optimize import fsolve
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
    mu_B = e * hbar / (2 * me) # magneto
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
    vF = hbar * kF / m


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
    def __init__(self, DavidPot=False,alpha = 34e-3 ,beta = 34e-3,gn = -3.4, GridFactor=1, W_g=300, S_g=400,Nb_d=100,
                 D_2DEG=120, W_r=1400, L_r=5000, WSC=200, a=30, T=0.1,
                 BField=[0],B_theta=[0],B_phi = [0], V_A=np.arange(0, -1.49, -0.01), Tev=[1e-3], Tev_Tunnel=[2e-3], E_excited=[5e-3],
                 TStrength=[0], TunnelLength=3, Phase=[np.pi / 4], Vbias_List=[0], PeriBC=[0],
                 SNjunc=['SNS'], ProOn=[1], delta=5.5e-3,DateT = '',TimeT = '',MasterMultiRun = False,
                 muN=0, muSC=10e-3, VGate_shift=-0.1, DefectAmp=0.5, DefectNumPer = 10,SeriesR=0,
                 NextNanoName=None, ReferenceData=None, SaveNameNote=None,Masterfilepath = None,
                 ShowDensity=False, Swave=False, TeV_Normal=True, CombineTev=True, CombineMu=False, AddOrbitEffect=True,
                 BlockWarnings=True,showBands = False,NumBands = 1,
                 SwpID="Vg", Digits=5, PlotbeforeFigures=5,PlotbeforeFigures_Ana = 20):

        self.alpha =alpha  # eVnm SOI
        self.beta = beta
        self.showBands = showBands
        self.NumBands = NumBands
        self.gn = gn  # g-factor
        # a = 30  # nm # grid point separation
        self.BlockWarnings = BlockWarnings
        self.ReferenceData = ReferenceData
        if self.ReferenceData != None:
            self.GetReferenceData(self.ReferenceData)
        self.SeriesR = SeriesR
        self.a = a
        self.Nb_d = Nb_d*1e-9
        t_test = 1000*(self.hbar ** 2 / (2 * self.m * (20*1e-9) ** 2))/self.e
        self.Orbit = AddOrbitEffect
        self.delta_raw = delta
        self.Data = []
        self.TXT = []
        self.DefectNumPer = DefectNumPer


        self.TeV_Normal = TeV_Normal
        self.CombineMu = CombineMu
        self.CombineTev = CombineTev
        if self.CombineTev == 1:

            Tev_Tunnel = Tev

        if self.CombineMu == 1:

            muN = muSC


        self.PlotbeforeFigures = PlotbeforeFigures
        self.PlotbeforeFigures_Ana = PlotbeforeFigures_Ana

        V_A = np.round(V_A, Digits)
        self.Digits = Digits
       # phase difference between two SC
        self.SaveNameNote = SaveNameNote
        # Tmev = [self.t_cal]
        self.DefectAmp = DefectAmp


        self.fileEnd = '-Kwt'
        self.SwpID = SwpID
        if self.SwpID == 'Vg':
            self.SwpUnit = ' (V)'
        elif self.SwpID == 'Vbias':
            self.SwpUnit = ' (V)'
        elif self.SwpID == 'E':
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
        self.DavidPot = DavidPot

        if self.DavidPot:
            self.GateSplit = int(GridFactor * S_g / self.a)
            self.GateWidth = int(GridFactor * W_g / self.a)
            self.Depth2DEG = int(GridFactor * D_2DEG / self.a)
            self.NextNanoName = 'DavidMethod'
        else:
            if not NextNanoName == None:
                self.NextNanoName = NextNanoName
                # syst.stdout.write("\r{0}".format('--------------------------- Loading Poisson Result -----------------------------------'))
                # syst.stdout.flush()
                # # print('--------------------------- Loading Poisson Result -----------------------------------')
                # self.Dict, self.VgList = SearchFolder(NextNanoName, 'bandedges_2d_2DEG_NoBoundary.fld', 'Gamma',
                #                                       ylim=(-1600, 2400))

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
        self.X1 = (self.L_r / 10000) * 1e-9 * (np.linspace(0, 10000, 10001) - 5000)
        self.W = int(W_r * GridFactor / self.a)  # (int) width of the junction (y direction)
        self.L = int(L_r * GridFactor / self.a)  # (int) length of the junction (x direction)
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



        if not os.path.exists(new_file_path):
            os.makedirs(new_file_path)
        new_file = new_file_path + 'Kwant_Class.py'
        os.system(f'cp {current_file_path} {new_file}')

        new_file = new_file_path + 'main.py'
        os.system(f'cp {Masterfilepath} {new_file}')

        self.XX = np.arange(0, self.L)
        self.YY = np.arange(0, self.W)

        self.Temp = T

        self.Combine_Change(SNjunc,PeriBC,ProOn,Tev,Tev_Tunnel,B_theta,B_phi)

        if self.SwpID == "Vbias":
            self.Combine_Still(muSC,muN,E_excited,V_A,TStrength,BField,Phase)
            self.VarSwp = Vbias_List
            self.TotalNrun = len(self.comb_change) * len(self.comb_still) * len(Vbias_List)

        elif self.SwpID == "Vg":
            self.Combine_Still(muSC, muN, E_excited, TStrength, BField,Vbias_List, Phase)
            self.VarSwp = V_A
            self.TotalNrun = len(self.comb_change) * len(self.comb_still) * len(V_A)

        elif self.SwpID == "E":
            self.Combine_Still(muSC, muN, V_A, TStrength, BField, Vbias_List, Phase)
            self.VarSwp = E_excited
            self.TotalNrun = len(self.comb_change) * len(self.comb_still) * len( E_excited)

        elif self.SwpID == "B":
            self.Combine_Still(muSC, muN, E_excited, V_A,TStrength, Vbias_List, Phase)
            self.VarSwp = BField
            self.TotalNrun = len(self.comb_change) * len(self.comb_still) * len(BField)

        elif self.SwpID == "Phase":
            self.Combine_Still(muSC, muN, E_excited,V_A, TStrength, Vbias_List, BField)
            self.VarSwp = Phase
            self.TotalNrun = len(self.comb_change) * len(self.comb_still) * len(Phase)
        if self.CombineMu:
            self.comb_still = list(map(self.Upzip, self.comb_still))
        self.TempDefineHc()
        self.Run_sweep()
    def Combine_Change(self,SNjunc,PeriBC,ProOn,Tev,Tev_Tunnel,B_theta,B_phi):
        if self.CombineTev:
            self.comb_change = list(
                itertools.product(list(SNjunc), list(PeriBC), list(ProOn),list(B_theta),list(B_phi), zip(list(Tev),list(Tev_Tunnel))))
            self.comb_change = list(map(self.Upzip, self.comb_change))
        else:
            self.comb_change = list(
                itertools.product(list(SNjunc), list(PeriBC), list(ProOn),list(B_theta),list(B_phi), list(Tev),list(Tev_Tunnel)))

    def Combine_Still(self,V1,V2,V3,V4,V5,V6,V7):
        if self.CombineMu:
            self.comb_still = list(
                itertools.product(zip(list(V1), list(V2)), list(V3), list(V4),
                                  list(V5), list(V6), list(V7)))
        else:
            self.comb_still = list(
                itertools.product(list(V1), list(V2), list(V3), list(V4),
                                  list(V5), list(V6), list(V7)))
    def GetReferenceData(self, Path):
        self.referdata = pd.read_excel(Path)

    def Fermi(self, En, mu):

        En = np.array(En)
        ans = np.zeros(En.size, dtype=complex)

        lim = 500
        if self.Temp == 0:
            if np.real(En - mu) > 0:
                ans = 0
            elif np.real(En - mu) < 0:
                ans = 1
            else:
                ans = 0.5
        else:
            if np.real(En - mu) / (self.kB * self.Temp) > lim:
                ans = 0
            elif np.real(En - mu) / (self.kB * self.Temp) < -lim:
                ans = 1
            else:
                ans = 1 / (np.exp(np.real(En[(np.real(En - mu) / (self.kB * self.Temp) <= lim) & (
                        np.real(En - mu) / (self.kB * self.Temp) >= -lim)] - mu) / (self.kB * self.Temp)) + 1)
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
                return 0 <= x <= self.L and -self.WSC <= y <= (self.W - int(self.W_reduced_r / self.a))
            else:
                return 0 <= x <= self.L and -self.WSC <= y <= (self.W + self.WSC)

        lat = kwant.lattice.square(norbs=4)
        # if self.SN == 'SN':
        #     lat = kwant.lattice.square(norbs = 4)
        # else:
        #     lat = kwant.lattice.square(norbs = 4)
        # as t is in eV, only the term alpha is not normalised, need to be provide in nature unit
        # other parameter mu V etc, should be in eV unit
        if self.Swave:
            PHMatrix = "sigma_z"
        else:
            PHMatrix = "sigma_0"

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
            if self.Orbit == False:  # add magntic field effect or not
                self.Ham = """ 
                                           ((k_x**2+k_y**2)""" + TeV_NN_Txt + """ - (mu(x,y)+V(x,y)-VG(x,y)-TB(x,y))""" + TeV_N_Txt + """+ (m*alpha**2/(2*e*hbar**2))""" + TeV_N_Txt + """+ (m*beta**2/(2*e*hbar**2))""" + TeV_N_Txt + """)*kron(sigma_z, sigma_0) +
                                           (EZ(x,y)/e)*(sin_theta*cos_phi*kron(sigma_0,sigma_x)+sin_theta*sin_phi*kron(sigma_0,sigma_y)+cos_theta*kron(sigma_0,sigma_z))""" + TeV_N_Txt + """ +
                                           (alpha/e)*(k_x*kron(sigma_0, sigma_y) - k_y*kron(sigma_0, sigma_x))*kron(sigma_z, sigma_0)""" + TeV_N_Txt + """ +
                                           (beta/e)*(k_x*kron(sigma_0, sigma_x) - k_y*kron(sigma_0, sigma_y))*kron(sigma_z, sigma_0)""" + TeV_N_Txt + """ +
                                           (Delta_0(x,y)*kron(sigma_x+1j*sigma_y,""" + PHMatrix + """) + Delta_0_prime(x,y)*kron(sigma_x-1j*sigma_y,""" + PHMatrix + """))""" + TeV_N_Txt + """+
                                           
                                      """
            else:
                # self.Ham = """
                #                                     ((k_x**2+k_y**2) - (mu(x,y)+V(x,y)-VG(x,y)-TB(x,y))/t )*kron(sigma_z, sigma_0) +
                #                                     EZ(x,y)*kron(sigma_0, sigma_x)/(e*t) +
                #                                     alpha*(k_x*kron(sigma_0, sigma_y) - k_y*kron(sigma_0, sigma_x))*kron(sigma_z, sigma_0)/(a*e*t) +
                #                                     (Delta_0(x,y)*kron(sigma_x+1j*sigma_y,""" + PHMatrix + """) + Delta_0_prime(x,y)*kron(sigma_x-1j*sigma_y,""" + PHMatrix + """))/t +
                #                                     ((e * (B**2) * (Y_rl(x,y)**2) * (c**2)/(2*m))*kron(sigma_z, sigma_0) -
                #                                     (hbar*B*Y_rl(x,y)*k_x/(m*c))*kron(sigma_0, sigma_0) -
                #                                     (alpha*Y_rl(x,y)*B/(hbar*c))*kron(sigma_0, sigma_y))/(t)
                #                                 """
                # make sure it is in eV /t
                # self.Ham = """
                #                                            ((k_x**2+k_y**2)""" + TeV_NN_Txt + """ - (mu(x,y)+V(x,y)-VG(x,y)-TB(x,y))""" + TeV_N_Txt + """ + (m*alpha**2/(2*e*hbar**2))""" + TeV_N_Txt + """)*kron(sigma_z, sigma_0) +
                #                                            (EZ(x,y)/e)*kron(sigma_0, sigma_x)""" + TeV_N_Txt + """ +
                #                                            (alpha/e)*(k_x*kron(sigma_0, sigma_y) - k_y*kron(sigma_0, sigma_x))*kron(sigma_z, sigma_0)""" + TeV_N_Txt + """ +
                #                                            (Delta_0(x,y)*kron(sigma_x+1j*sigma_y,""" + PHMatrix + """) + Delta_0_prime(x,y)*kron(sigma_x-1j*sigma_y,""" + PHMatrix + """))""" + TeV_N_Txt + """+
                #                                            ((e * (B**2) * (Y_rl(x,y)**2) /(2*m* (c**2)))*kron(sigma_z, sigma_0) -
                #                                            (hbar*B*Y_rl(x,y)*k_x/(m*c))*kron(sigma_0, sigma_0) -
                #                                            (alpha*Y_rl(x,y)*B/(hbar*c))*kron(sigma_0, sigma_y))""" + TeV_N_Txt + """
                #                                        """
                self.Ham = """
                                           ((k_x**2+k_y**2)""" + TeV_NN_Txt + """ - (mu(x,y)+V(x,y)-VG(x,y)-TB(x,y))""" + TeV_N_Txt + """+ (m*alpha**2/(2*e*hbar**2))""" + TeV_N_Txt + """+ (m*beta**2/(2*e*hbar**2))""" + TeV_N_Txt + """)*kron(sigma_z, sigma_0) +
                                           (EZ(x,y)/e)*(sin_theta*cos_phi*kron(sigma_0,sigma_x)+sin_theta*sin_phi*kron(sigma_0,sigma_y)+cos_theta*kron(sigma_0,sigma_z))""" + TeV_N_Txt + """ +
                                           (alpha/e)*(k_x*kron(sigma_0, sigma_y) - k_y*kron(sigma_0, sigma_x))*kron(sigma_z, sigma_0)""" + TeV_N_Txt + """ +
                                           (beta/e)*(k_x*kron(sigma_0, sigma_x) - k_y*kron(sigma_0, sigma_y))*kron(sigma_z, sigma_0)""" + TeV_N_Txt + """ +
                                           (Delta_0(x,y)*kron(sigma_x+1j*sigma_y,""" + PHMatrix + """) + Delta_0_prime(x,y)*kron(sigma_x-1j*sigma_y,""" + PHMatrix + """))""" + TeV_N_Txt + """+
                                           ((e * (B**2) * (Y_rl(x,y)**2) /(2*m* (c**2)))*kron(sigma_z, sigma_0) -
                                           (hbar*B*Y_rl(x,y)*k_x/(m*c))*kron(sigma_0, sigma_0) -
                                           (alpha*Y_rl(x,y)*B/(hbar*c))*kron(sigma_0, sigma_y))""" + TeV_N_Txt + """
                                       """

            # self.Ham_l_up_S = """
            #                                ((k_x**2+k_y**2)""" + TeV_NN_Txt + """ - (mu_S + V_bias)""" + TeV_N_Txt + """)*kron(sigma_z, sigma_0) +
            #                                (Delta_SC_up*kron(sigma_x+1j*sigma_y,""" + PHMatrix + """) + Delta_SC_up_prime*kron(sigma_x-1j*sigma_y,""" + PHMatrix + """))""" + TeV_N_Txt + """
            #                            """
            self.Ham_l_dn_S = """
                                           ((k_x**2+k_y**2)""" + TeV_NN_Txt + """ - mu_S"""+ TeV_N_Txt+""")*kron(sigma_z, sigma_0) +
                                           (Delta_SC_dn*kron(sigma_x+1j*sigma_y,""" + PHMatrix + """) + Delta_SC_dn_prime*kron(sigma_x-1j*sigma_y,""" + PHMatrix + """))""" + TeV_N_Txt + """
                                       """
            # self.Ham_l_dn_N = """
            #                                ((k_x**2+k_y**2)""" + TeV_NN_Txt + """ - (mu(x,y)+V(x,y)-VG(x,y)-TB(x,y))""" + TeV_N_Txt + """+ (m*alpha**2/(2*e*hbar**2))""" + TeV_N_Txt + """+ (m*beta**2/(2*e*hbar**2))""" + TeV_N_Txt + """)*kron(sigma_z, sigma_0) +
            #                                (EZ(x,y)/e)*(sin_theta*cos_phi*kron(sigma_0,sigma_x)+sin_theta*sin_phi*kron(sigma_0,sigma_y)+cos_theta*kron(sigma_0,sigma_z))""" + TeV_N_Txt + """ +
            #                                (alpha/e)*(k_x*kron(sigma_0, sigma_y) - k_y*kron(sigma_0, sigma_x))*kron(sigma_z, sigma_0)""" + TeV_N_Txt + """ +
            #                                (beta/e)*(k_x*kron(sigma_0, sigma_x) - k_y*kron(sigma_0, sigma_y))*kron(sigma_z, sigma_0)""" + TeV_N_Txt + """
            #                              """
            self.Ham_l_N_metal = """
                                           ((k_x**2+k_y**2)""" + TeV_NN_Txt + """ - mu_S""" + TeV_N_Txt + """)*kron(sigma_z, sigma_0)
                                 """
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
        # template_l_up_S = kwant.continuum.discretize(self.Ham_l_up_S)
        # template_l_dn_S = kwant.continuum.discretize(self.Ham_l_dn_S)

        template_l_up_S = kwant.continuum.discretize(self.Ham_l_N_metal)
        template_l_dn_S = kwant.continuum.discretize(self.Ham_l_N_metal)
        # template_l_dn_N = kwant.continuum.discretize(self.Ham_l_dn_N)
        # print(template)
        template_l_dn_N = kwant.continuum.discretize(self.Ham_l_N_metal)
        sys = kwant.Builder()
        sys.fill(template, central_region, (0, 0));
        ########################################################################################

        ########################################################################################

        ########################################################################################

        ########################################################################################
        sym1 = kwant.TranslationalSymmetry((0, -1))
        sym2 = kwant.TranslationalSymmetry((0, 1))

        def lead_shape(site):
            (x, y) = site.pos
            return (1 <= x < self.L)

        lead_up = kwant.Builder(sym1)


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

        lead_up.fill(template_l_up_S, lead_shape, (int(self.L / 2), -self.WSC))
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

        return sys.finalized()
        #

    def density(self, sys, params, lead_nr):
        wf = kwant.wave_function(sys, self.E, params=params, check_hermiticity=True)
        A = wf(lead_nr)
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
        self.d = (abs(wf(lead_nr)) ** 2).sum(axis=0)
        self.Deltamap = Ans

    def Gen_SaveFileName(self):
        self.PBtxt = 'PB' if self.PB == 1 else 'nPB'
        self.Proximitytxt = 'On' if self.ProximityOn == 1 else 'Off'

        self.SAVEFILENAME_origin = str(self.GlobalVswpCount + 1) + ':' + self.SaveTime
        self.SAVENOTETitle = ["DATE(Y/M/D)", "TIME(h/m/s)", "Ee(meV)", "B(T)", "Vg(V)", "VB(V)", "Phase(pi_rad)",
                              "SN-SNS", "PB?", "Proxy?", "muN(meV)",
                              "muS(meV)", "t(meV)", "t_tunnelcouple(meV)", "Tl_B(t)", "Defect(t)", "Delta(ueV)", "Note"]

        if self.SwpID == "Vg":
            self.SAVEFILENAME = 'VgSwp'

            self.SAVENOTE_buff = [1000*self.E*self.t, self.B, "X", self.Vbias, self.phi / np.pi]



        elif self.SwpID == "E":
            self.SAVEFILENAME = 'Eswp'

            self.SAVENOTE_buff = ["X", self.B, self.V_Applied, self.Vbias, self.phi / np.pi]



        elif self.SwpID == "B":
            self.SAVEFILENAME = 'BSwp'
            self.SAVENOTE_buff = [1000*self.E*self.t, "X", self.V_Applied, self.Vbias, self.phi / np.pi]



        elif self.SwpID == "Vbias":
            self.SAVEFILENAME = 'VbSwp'
            self.SAVENOTE_buff = [1000*self.E*self.t, self.B, self.V_Applied, "X", self.phi / np.pi]


        elif self.SwpID == "Phase":
            self.SAVEFILENAME = 'PhaseSwp'
            self.SAVENOTE_buff = [1000*self.E*self.t, self.B, self.V_Applied, self.Vbias, "X"]

        self.SAVEFILENAME = self.NextNanoName + self.fileEnd + '/' + self.SaveTime + '/' + self.SN + '-' + self.PBtxt + '-' + \
                            self.Proximitytxt + '-muN' + str(np.round(self.mu_N * 1e3, 3)) + 'meV-muS' + \
                            str(np.round(self.mu_SC * 1e3, 3)) + 'meV-t' + str(
            np.round(self.t * 1e3, 3)) + 'meV-Tl' + str(self.TunnelStrength) + 't-DF' + str(
            self.DefectAmp) + 't' + self.SAVEFILENAME + '/'

        if self.GlobalVswpCount == 0:
            self.SAVENOTE = np.vstack((self.SAVENOTETitle,
                                       [self.Date, self.Time] + self.SAVENOTE_buff + [self.SN, self.PBtxt,
                                                                                      self.Proximitytxt,
                                                                                      np.round(self.mu_N * 1e3, 3), \
                                                                                      np.round(self.mu_SC * 1e3, 3),
                                                                                      np.round(self.t * 1e3, 3),
                                                                                      np.round(self.t_Tunnel * 1e3, 3),
                                                                                      self.TunnelStrength,
                                                                                      self.DefectAmp,
                                                                                      np.round(self.delta_raw * 1e6, 3), \
                                                                                      self.SaveNameNote]))
        else:
            self.SAVENOTE = np.vstack((self.SAVENOTE,
                                       [self.Date, self.Time] + self.SAVENOTE_buff + [self.SN, self.PBtxt,
                                                                                      self.Proximitytxt,
                                                                                      np.round(self.mu_N * 1e3, 3), \
                                                                                      np.round(self.mu_SC * 1e3, 3),
                                                                                      np.round(self.t * 1e3, 3),
                                                                                      np.round(self.t_Tunnel * 1e3, 3),
                                                                                      self.TunnelStrength,
                                                                                      self.DefectAmp,
                                                                                      np.round(self.delta_raw * 1e6, 3), \
                                                                                      self.SaveNameNote]))
            # self.SAVEFILENAME_origin = self.SN + '_' + self.PBtxt + '_' + self.Proximitytxt + '_t' + str(
        #     self.tmev) + 'meV_E_excited'+str(self.E)+'t_Tunnel' + str(self.TunnelStrength) + 't_Field' + str(self.B) + 'T'

        table2 = list(map(str, self.SAVENOTE_buff + [self.SN, self.PBtxt, self.Proximitytxt,
                                                     np.round(self.mu_N * 1e3, 3), \
                                                     np.round(self.mu_SC * 1e3, 3),
                                                     np.round(self.t * 1e3, 3),
                                                     np.round(self.t_Tunnel * 1e3, 3),
                                                     self.TunnelStrength, self.DefectAmp, np.round(self.delta_raw * 1e6, 3)]))
        table = [["     Ee(meV)     ", "  B(T)  ", "  Vg(V)  ", "  VB(V)  ", "  Phi(pi)  ", "SN-SNS", "PB?", "Proxy?", "  muN(meV)  ",
                  "  muS  ", "  t(meV)  ", "  t_tc(meV)  ", "  Tl(t)  ", "  DF(t)  ", "  Delta(ueV)  "], table2]
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

        self.density(sys, params, 1)  # Calculate density
        pick_electron = np.arange(0, len(self.d), 4)  # pickout the electron density part
        self.d = self.d[pick_electron]
        local_dos = kwant.ldos(sys, params=params, energy=self.E)  # Calculate local density of state
        self.local_dos = local_dos[pick_electron]  # pickout the electron LDOS
        sites = kwant.plotter.sys_leads_sites(sys, 0)[0]  # Get the site and coordinate to plot
        coords = kwant.plotter.sys_leads_pos(sys, sites)
        self.img, Amin, Amax = kwant.plotter.mask_interpolate(coords, self.d)  # Make colormap

        if self.BlockWarnings:
            warnings.filterwarnings("ignore")
        self.fig = plt.figure(figsize=(14, 11))
        Ax0 = plt.subplot(3, 3, 1)
        set_size(6, 2, Ax0)
        kwant.plotter.plot(sys, ax=Ax0)
        plt.axis('off')
        Ax1 = plt.subplot(3, 3, 2)
        kwant.plotter.map(sys, self.local_dos, ax=Ax1)
        plt.title('LDOS')
        plt.axis('off')
        Ax2 = plt.subplot(3, 3, 3)

        # kwant.plotter.map(sys, np.abs(self.Deltamap), ax=Ax2)
        pcolor = Ax2.imshow(np.abs(self.SpatialDeltaMap).T)

        plt.title('Order Parameter')
        plt.axis('off')
        Ax3 = plt.subplot(3, 3, 4)
        pcolor = Ax3.imshow(self.Defect_Map.T)
        # cbar = fig0.colorbar(pcolor)
        plt.title('Defect')
        plt.axis('off')
        Ax4 = plt.subplot(3, 3, 5)
        pcolor = Ax4.imshow(self.Delta_abs_Map.T)
        # cbar = fig0.colorbar(pcolor)
        plt.title('Delta_abs')
        plt.axis('off')
        Ax5 = plt.subplot(3, 3, 6)
        pcolor = Ax5.imshow(self.Delta_phase_Map.T)
        # cbar = fig0.colorbar(pcolor)
        plt.title('Delta_phase')
        plt.axis('off')
        Ax6 = plt.subplot(3, 3, 7)
        pcolor = Ax6.imshow(self.gn_Map.T)
        # cbar = fig0.colorbar(pcolor)
        plt.title('gn')
        plt.axis('off')
        Ax7 = plt.subplot(3, 3, 8)
        pcolor = Ax7.imshow(self.Vbias_Map.T)
        # cbar = fig0.colorbar(pcolor)
        plt.title('Vbias')
        plt.axis('off')
        Ax8 = plt.subplot(3, 3, 9)
        pcolor = Ax8.imshow(self.Tunnel_Map.T)
        # cbar = fig0.colorbar(pcolor)
        plt.title('Tunnel barrier')
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
        # plt.subplots_adjust(left=0.125, bottom=0.125, right=0.15, top=0.5, wspace=0.2, hspace=0.2)

        # self.fig.suptitle(self.PBtxt + ';' + self.Proximitytxt + ';' + self.SN + ';TB=' + str(
        #     self.TunnelStrength) + "t;Vg=" + self.VStr + ";t=" + str(self.t) + ";E=" +
        #                   str(self.E) + ";muN=" + str(self.mu_N) + ";muSC=" + str(self.mu_SC) +
        #                   "eV;B=" + str(self.B) + 'T')
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
        pcolor = ax4.imshow(self.Delta_abs_Map.T)
        cbar = self.fig.colorbar(pcolor)
        plt.title('Delta')

        ax5 = plt.subplot(3, 3, 5)
        ax5.plot(self.img.T[int(np.shape(self.img.T)[0] / 2), :])
        plt.title('Wf left right cut(Gate)')

        ax6 = plt.subplot(3, 3, 6)
        # pcolor = ax4.pcolormesh(Potential_Map.T, shading='auto')
        pcolor = ax6.imshow(self.Potential_Map.T)
        ax6.axvline(x=int(np.shape(self.Potential_Map.T)[1] / 4), linestyle='--')
        ax6.axvline(x=int(np.shape(self.Potential_Map.T)[1] / 2), linestyle='--')
        cbar = self.fig.colorbar(pcolor)
        plt.title('Potential[eV/t]')

        ax7 = plt.subplot(3, 3, 7)
        ax7.plot(self.Delta_abs_Map.T[:, int(np.shape(self.Delta_abs_Map.T)[1] / 2)])
        ax7.axhline(y=self.Delta_induced, color='r')
        ax7.text(x=0, y=self.Delta_induced, s=str(np.round(1e6 * self.Delta_induced, 3)))
        plt.title('Delta up down cut')

        ax8 = plt.subplot(3, 3, 8)
        ax8.plot(self.Potential_Map.T[:, int(np.shape(self.Potential_Map.T)[1] / 4)])
        ax8.plot(self.Potential_Map.T[:, int(3 * np.shape(self.Potential_Map.T)[1] / 4)])
        plt.title('Potential up down cut(Gate)')

        ax9 = plt.subplot(3, 3, 9)
        ax9.plot(self.Potential_Map.T[:, int(np.shape(self.Potential_Map.T)[1] / 2)])
        plt.title('Potential up down cut(no Gate)')

        self.fig.tight_layout()

    def Gen_Conduct_Plot(self, x, y, Xlabel):
        if self.BlockWarnings:
            warnings.filterwarnings("ignore")
        self.fig = plt.figure(figsize=(14, 11))
        # self.fig.suptitle(self.PBtxt + ';' + self.Proximitytxt + ';' + self.SN + ';TB=' + str(
        #     self.TunnelStrength) + "t;Vg=" + self.VStr + ";t=" + str(self.t) + ";E=" +
        #                   str(self.E) + ";muN=" + str(self.mu_N) + ";muSC=" + str(self.mu_SC) +
        #                   "eV;B=" + str(self.B) + 'T')
        self.fig.suptitle(self.MeseTitle + "\n" + self.MeseValue)
        # self.fig.suptitle("\r{0}".format(self.MeseTitle) + "\n" + "\r{0}".format(self.MeseValue))

        ax0 = plt.subplot(1, 2, 1)
        ax0.plot(x, y, label=" 0 Ohm")
        if self.SwpID == 'E':
            # ax0.axvline(x=self.delta, color='r')
            # ax0.axvline(x=-self.delta, color='r')
            ax0.axvline(x=self.delta * 2000, color='b')
            ax0.axvline(x=-self.delta * 2000, color='b')
            ax0.text(x=self.Delta_induced * 2/self.t, y=0, s=str(np.round(self.Delta_induced * 1e6, 3)))
            # ax0.axvline(x=self.Delta_induced, color='r')
            # ax0.axvline(x=-self.Delta_induced, color='r')
            ax0.axvline(x=self.Delta_induced * 2000, color='r')
            ax0.axvline(x=-self.Delta_induced * 2000, color='r')
        if self.ReferenceData != None:
            ax0.plot(self.referdata.Vg1, self.referdata.G1, self.referdata.Vg2, self.referdata.G2)
        ax0.legend()
        plt.xlabel(Xlabel)
        plt.ylabel("G/G0[/(2e^2/h)]")
        ax1 = plt.subplot(1, 2, 2)
        ax1.plot(x, (1 / (self.SeriesR + 1 / (7.74809173e-5 * np.array(y)))) / 7.74809173e-5,
                 label=str(self.SeriesR)+" Ohm")
        ax1.legend()
        if self.ReferenceData != None:
            ax1.plot(self.referdata.Vg1, self.referdata.G1, self.referdata.Vg2, self.referdata.G2)
        plt.xlabel(Xlabel)
        plt.ylabel("G/G0[/(2e^2/h)]")
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
            self.Defect_Map = np.zeros((self.L + 1, self.WSC + self.W - int(self.W_reduced_r / self.a) + 1))
            self.Potential_Map = np.zeros((self.L + 1, self.WSC + self.W - int(self.W_reduced_r / self.a) + 1))
            self.Delta_abs_Map = np.zeros((self.L + 1, self.WSC + self.W - int(self.W_reduced_r / self.a) + 1))
            self.Delta_phase_Map = np.zeros((self.L + 1, self.WSC + self.W - int(self.W_reduced_r / self.a) + 1))
            self.Onsite_Map = np.zeros((self.L + 1, self.WSC + self.W - int(self.W_reduced_r / self.a) + 1))
            self.gn_Map = np.zeros((self.L + 1, self.WSC + self.W - int(self.W_reduced_r / self.a) + 1))
            self.Vbias_Map = np.zeros((self.L + 1, self.WSC + self.W - int(self.W_reduced_r / self.a) + 1))
            self.Tunnel_Map = np.zeros((self.L + 1, self.WSC + self.W - int(self.W_reduced_r / self.a) + 1))
        else:
            self.Defect_Map = np.zeros((self.L + 1, 2 * self.WSC + self.W + 1))
            self.Potential_Map = np.zeros((self.L + 1, 2 * self.WSC + self.W + 1))
            self.Delta_abs_Map = np.zeros((self.L + 1, 2 * self.WSC + self.W + 1))
            self.Delta_phase_Map = np.zeros((self.L + 1, 2 * self.WSC + self.W + 1))
            self.Onsite_Map = np.zeros((self.L + 1, 2 * self.WSC + self.W + 1))
            self.gn_Map = np.zeros((self.L + 1, 2 * self.WSC + self.W + 1))
            self.Vbias_Map = np.zeros((self.L + 1, 2 * self.WSC + self.W + 1))
            self.Tunnel_Map = np.zeros((self.L + 1, 2 * self.WSC + self.W + 1))

    def orderDelta(self, X, Y, Bz, lambdaIn, leadN, PHI0, Bx=0, alphaangle=0):
        # Theory based on <Controlled finite momentum pairing and spatially
        # varying order parameter in proximitized HgTe
        # quantum wells>

        X_m = 1e-9 * (X - (self.L - 1) / 2) * self.a / self.GridFactor
        Y_m = 1e-9 * Y * self.a / self.GridFactor
         # in m
        # W = 1e-9 * self.W_r / 4
        PHIJ = PHI0 / (2 * np.pi) + ((-1) ** leadN * self.X1 * Bz * (self.W_r + self.L_r) * 1e-9) / (
                4 * np.pi * self.hbar / (2 * self.e))
        lambda_sp = lambdaIn * np.exp(2 * np.pi * 1j * PHIJ)


        # kF = (2 * np.pi * 1e10) ** 0.5


        Dk = self.gn * self.mu_B * Bx / (self.hbar * self.vF)
        # if Dk != 0:
        #     A = 1/Dk
        #     print(1)
        # Dk = np.pi/(2*W)
        gamma = Dk * (np.sin(alphaangle) * Y_m + np.cos(alphaangle) * (X_m - self.X1))

        F = (np.exp(1j * gamma) + np.exp(-1j * gamma)) / (
                8 * (np.pi ** 2) * self.hbar * self.m * ((X_m - self.X1) ** 2 + Y_m ** 2))

        ORDER = np.trapz(lambda_sp * F, self.X1) / (self.Factor)

        # A = abs(ORDER)/lambdaIn

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

        def equation2(A0, Nb_d, lambda_london):
            return self.B ** 2 - self.Hcb ** 2 * 4 * A0 ** 2 * (A0 ** 2 - 1) * np.cosh(A0 * Nb_d / (2 * lambda_london)) ** 2 / (
                        1 - (lambda_london / (A0 * Nb_d)) * np.sinh(A0 * Nb_d / lambda_london))

        # initial_guess = np.array([1])
        if self.B <= self.Hc:
            solutions = fsolve(equation2, [1], args=(self.Nb_d, self.lambda_london))
            A2 = solutions[0]
            self.delta = A2 * self.delta_raw
        else:
            self.delta = 0

        # print(1)


    def Run_sweep(self):

        def Delta_0_dis(x, y):

            Square = np.heaviside(y, 0) - np.heaviside(y - self.W, 1)
            Delta_Spatial = self.delta * (1 - np.heaviside(y, 0)) + \
                            self.delta * np.heaviside(y - self.W, 1)

            Phase_Spatial = -self.phi * (1 - np.heaviside(y, 0)) + self.phi * np.heaviside(
                y - self.W, 1)
            DELTA = Delta_Spatial * np.exp(Phase_Spatial * 1j / 2)
            if self.ProximityOn == 1:
                if (0 <= x < self.L) and (0 <= y < self.W):
                    DELTA = DELTA + self.SpatialDeltaMap[int(x), int(y)] * Square
            self.Delta_abs_Map[int(x), int(y) + self.WSC] = np.abs(DELTA)
            # self.Delta_abs_Map[int(x), int(y) + self.WSC] = np.angle(DELTA)

            self.Delta_phase_Map[int(x), int(y) + self.WSC] = np.angle(DELTA)
            # self.Delta_induced = np.min(self.Delta_abs_Map)

            return DELTA

        def Delta_0_prime_dis(x, y):
            Square = np.heaviside(y, 0) - np.heaviside(y - self.W, 1)
            Delta_Spatial = self.delta * (1 - np.heaviside(y, 0)) + \
                            self.delta * np.heaviside(y - self.W, 1)

            Phase_Spatial = -self.phi * (1 - np.heaviside(y, 0)) + self.phi * np.heaviside(
                y - self.W, 1)
            DELTA = Delta_Spatial * np.exp(Phase_Spatial * 1j / 2)
            if self.ProximityOn == 1:
                if (0 <= x < self.L) and (0 <= y < self.W):
                    DELTA = DELTA + self.SpatialDeltaMap[int(x), int(y)] * Square
            return np.conjugate(DELTA)

        def EZ_dis(x, y):
            Square = np.heaviside(y, 0) - np.heaviside(y - self.W, 1)
            g = Square * self.gn
            self.gn_Map[int(x), int(y) + self.WSC] = np.real(g)
            return g * self.mu_B * self.B / 2

        def mu_dis(x, y):
            Square = np.heaviside(y, 0) - np.heaviside(y - self.W, 1)
            AntiSquare = 1 - Square
            MU = Square * (self.mu_N + self.Defect_Map[int(x), int(y)]) + AntiSquare * self.mu_SC
            return MU

        def VGate_dis(x, y):
            Square = np.heaviside(y, 0) - np.heaviside(y - self.W, 1)
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
            self.Potential_Map[int(x), int(y) + self.WSC] = VGate

            return VGate

        def V_dis(x, y):

            V = (1 - np.heaviside(y, 0)) * self.Vbias
            self.Vbias_Map[int(x), int(y) + self.WSC] = np.real(V)
            return V

        def TunnelBarrier_dis(x, y):
            TunnelBarrier = 0
            if abs(y) <= self.TunnelLength / 2 or abs(y - self.W) <= self.TunnelLength:
                TunnelBarrier = self.GammaTunnel
            self.Tunnel_Map[int(x), int(y) + self.WSC] = np.real(TunnelBarrier)
            return TunnelBarrier

        def t_dis(x, y):
            t = self.t
            if abs(y) <= self.TunnelLength / 2 or abs(y - self.W) <= self.TunnelLength:
                t = self.t_Tunnel
            return t

        def Y_rl_dis(x, y):
            result = 1e-9 * (y - self.W / 2) * self.a / self.GridFactor

            return result  # in actual nm

        elapsed_tol = 0



        if self.DavidPot:
            self.u_sl_ref_2DEG = 0
        else:

            Index0 = self.VgList.index(0.0)
            u_sl_0 = self.Dict[Index0]
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

        for self.SN, self.PB, self.ProximityOn, self.B_theta,self.B_phi,self.t, self.t_Tunnel in self.comb_change:




            sys = self.make_system()
            self.DefOutputMap()
            self.Defect_Map = self.GaussianDefect(FWHM=1)
            if self.SN == 'SN':
                V_ref_dis = self.u_sl_ref_2DEG
            else:
                V_ref_dis = 0



            for Var1, Var2, Var3, Var4, Var5, Var6, Var7 in self.comb_still:
                # self.delta = self.delta_raw
                if self.SwpID == "Vbias":
                    self.mu_SC, self.mu_N, self.E, self.V_Applied, self.TunnelStrength, self.B, self.phi = Var1, Var2, Var3, Var4, Var5, Var6, Var7

                elif self.SwpID == "Vg":
                    self.mu_SC, self.mu_N, self.E, self.TunnelStrength, self.B, self.Vbias, self.phi = Var1, Var2, Var3, Var4, Var5, Var6, Var7

                elif self.SwpID == "E":
                    self.mu_SC, self.mu_N, self.V_Applied, self.TunnelStrength, self.B, self.Vbias, self.phi = Var1, Var2, Var3, Var4, Var5, Var6, Var7

                elif self.SwpID == "B":
                    self.mu_SC, self.mu_N, self.E, self.V_Applied, self.TunnelStrength, self.Vbias, self.phi = Var1, Var2, Var3, Var4, Var5, Var6, Var7

                elif self.SwpID == "Phase":
                    self.mu_SC, self.mu_N, self.E, self.V_Applied, self.TunnelStrength, self.Vbias, self.B = Var1, Var2, Var3, Var4, Var5, Var6, Var7


                self.GammaTunnel = self.TunnelStrength

                # self.alpha_TB = self.alpha / (2 * self.a)  # eV

                self.conductances = []
                self.conductances2 = []
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

                        self.u_sl_ref_2DEG = self.u_sl(self.L_r / 2, self.W_r - self.W_reduced_r)
                        # self.u_sl_ref = u_sl_0(self.L_r / 2, self.W_reduced_r)
                        self.u_sl_ref = self.u_sl(self.L_r / 2, 2)
                now = datetime.now()
                DateLocal = now.strftime("%YY%mM%dD")
                TimeLocal = now.strftime("%Hh%Mm%Ss")
                self.LocalSave = DateLocal + '-' + TimeLocal
                for VSwp in self.VarSwp:
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

                            self.u_sl_ref_2DEG = self.u_sl(self.L_r / 2, self.W_r - self.W_reduced_r)
                            # self.u_sl_ref = u_sl_0(self.L_r / 2, self.W_reduced_r)
                            self.u_sl_ref = self.u_sl(self.L_r / 2, 2)
                    elif self.SwpID == "E":
                        self.E = VSwp
                    elif self.SwpID == "B":
                        self.B = VSwp
                        self.TempDefineGap()
                    elif self.SwpID == "Phase":
                        self.phi = VSwp
                    if not self.TeV_Normal:
                        self.E = self.E * self.t
                    self.VStr = str(round(self.V_Applied, self.Digits))
                    self.GlobalRunCount = self.GlobalRunCount + 1
                    RunCount = RunCount + 1

                    # Get the initial reference gate potential and proximity effect factor depend on B
                    if RunCount == 1 or self.SwpID == 'B' or self.SwpID == 'Phase':


                        if self.BlockWarnings:
                            warnings.filterwarnings("ignore")

                        if self.ProximityOn and not self.delta == 0:
                            self.Factor = 1  # correct the proximity effect of order parameter
                            if self.SN == 'SN':
                                self.Factor = np.abs(self.orderDelta((self.L - 1) / 2, 1,
                                                                     self.B * np.sin(self.B_theta) * np.sin(self.B_phi),
                                                                     self.delta, 0,
                                                                     -self.phi,
                                                                     Bx=self.B * np.sin(self.B_theta) * np.cos(
                                                                         self.B_phi),
                                                                     alphaangle=0) / self.delta)
                            else:
                                self.Factor = np.abs(self.orderDelta((self.L - 1) / 2, 1,
                                                                     self.B * np.sin(self.B_theta) * np.sin(self.B_phi),
                                                                     self.delta, 0,
                                                                     -self.phi / 2,
                                                                     Bx=self.B * np.sin(self.B_theta) * np.cos(
                                                                         self.B_phi),
                                                                     alphaangle=0) + \
                                                     self.orderDelta((self.L - 1) / 2, self.W - 1, self.B,
                                                                     self.delta, 1, self.phi / 2, Bx=0,
                                                                     alphaangle=0)) / self.delta
                            for i in range(len(self.XX)):
                                for j in range(len(self.YY)):
                                    if self.SN == 'SN':

                                        self.SpatialDeltaMap[i, j] = self.orderDelta(self.XX[i], self.YY[j] + 1, self.B*np.sin(self.B_theta)*np.sin(self.B_phi),
                                                                                     self.delta, 0,
                                                                                     self.phi, Bx=self.B*np.sin(self.B_theta)*np.cos(self.B_phi),
                                                                                     alphaangle=0)

                                    else:

                                        self.SpatialDeltaMap[i, j] = self.orderDelta(self.XX[i], self.YY[j] + 1, self.B*np.sin(self.B_theta)*np.sin(self.B_phi),
                                                                                     self.delta, 0,
                                                                                     -self.phi / 2, Bx=self.B*np.sin(self.B_theta)*np.cos(self.B_phi),
                                                                                     alphaangle=0) + \
                                                                     self.orderDelta(self.XX[i],
                                                                                     self.W - self.YY[j] + 1,
                                                                                     self.B*np.sin(self.B_theta)*np.sin(self.B_phi),
                                                                                     self.delta, 1, self.phi / 2, Bx=self.B*np.sin(self.B_theta)*np.cos(self.B_phi),
                                                                                     alphaangle=0)
                            # A = np.abs(self.SpatialDeltaMap).T

                        if self.BlockWarnings:
                            warnings.filterwarnings("always")







                    params = dict(a=1e-9, e=self.e, Delta_0=Delta_0_dis, EZ=EZ_dis, TB=TunnelBarrier_dis,
                                  V=V_dis, VG=VGate_dis, alpha=self.alpha * (1e-9) * self.e, hbar=self.hbar,
                                  m=self.m,beta=self.beta * (1e-9) * self.e,
                                  mu=mu_dis, mu_S=self.mu_SC, mu_N=self.mu_N,sin_theta = np.sin(self.B_theta),sin_phi=np.sin(self.B_phi),cos_phi=np.cos(self.B_phi),cos_theta = np.cos(self.B_theta),
                                  EZ_fix=self.gn * self.mu_B * self.B / 2,
                                  V_ref=V_ref_dis, t=t_dis, Delta_0_prime=Delta_0_prime_dis, V_bias=self.Vbias,
                                  Delta_SC_up=self.delta * np.exp(-1j * self.phi / 2),
                                  Delta_SC_dn=self.delta * np.exp(1j * self.phi / 2),
                                  Delta_SC_up_prime=self.delta * np.exp(1j * self.phi / 2),
                                  Delta_SC_dn_prime=self.delta * np.exp(-1j * self.phi / 2),
                                  B=self.B, Y_rl=Y_rl_dis, c=self.c)


                    SMatrix = kwant.solvers.default.smatrix(sys, self.E, params=params, out_leads=[0, 1],
                                                            in_leads=[0, 1])
                    if self.BlockWarnings:
                        warnings.filterwarnings("ignore")
                    self.Delta_induced = np.min(self.Delta_abs_Map.T[:, int(np.shape(self.Delta_abs_Map.T)[1] / 2)])

                    if RunCount%self.PlotbeforeFigures_Ana == 0:
                        try:
                            if self.showBands:
                                kwant.plotter.bands(self.lead_test.finalized(), show=False, params=params)
                                plt.xlabel("momentum [(lattice constant)^-1]")
                                plt.ylabel("energy [t]")
                                plt.title('Superconductor')
                                plt.show()
                                kwant.plotter.bands(self.lead_test_Ham.finalized(), show=False, params=params)
                                plt.xlabel("momentum [(lattice constant)^-1]")
                                plt.ylabel("energy [t]")
                                plt.title('Semiconductor')
                                plt.show()
                                kwant.plotter.bands(self.lead_test_Metal.finalized(), show=False, params=params)
                                plt.xlabel("momentum [(lattice constant)^-1]")
                                plt.ylabel("energy [t]")
                                plt.title('Metal')
                                plt.show()



                            self.Gen_Site_Plot(sys, params)
                            self.fig.savefig(self.SAVEFILENAME +self.LocalSave+ '_' + str(VSwp) + "Sites.png")
                            if self.ShowDensity == 1:
                                self.fig.show()
                            self.Gen_Ana_Plot()
                            self.fig.savefig(self.SAVEFILENAME +self.LocalSave+ '_' + str(VSwp) + "Ana.png")
                            if self.ShowDensity == 1:
                                self.fig.show()
                        except:
                            syst.stdout.write("Site plot not generated")
                            syst.stdout.flush()

                    if self.BlockWarnings:
                        warnings.filterwarnings("always")
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
                    Elapsed = TimeFormat(elapsed)
                    LeftRuns = np.round(self.TotalNrun - self.GlobalRunCount, 0)
                    TimeSpend = np.round(time.time() - self.GlobalStartTime, 2)
                    TimeTXT = 'total:' + TimeFormat(TimeSpend) + '/left:' + TimeFormat(
                        LeftRuns * elapsed_tol / self.GlobalRunCount) + ' ' + TimeFormat(
                        elapsed_tol / self.GlobalRunCount) + '/point'

                    # Mese =self.PBtxt + ';' + self.Proximitytxt + ';' + self.SN + ';TB=' + \
                    #        str(self.TunnelStrength) + "t;Vg=" + self.VStr + ";t=" + str(self.t) + ";E=" +\
                    #       str(self.E)+ ";muN=" + str(self.mu_N)+ ";muSC=" + str(self.mu_SC)+\
                    #       "eV;B=" + str(self.B) + 'T'
                    # print(Mese)

                    # syst.stdout.flush()

                    # syst.stdout.flush()
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
                    self.VarSwp = 1000* self.VarSwp * self.t
                    TitleTxt1 = ["E", "meV", "Excitation_Energy"]
                elif self.SwpID == "B":
                    TitleTxt1 = ["B", "T", "Magnetic_Field"]
                elif self.SwpID == "Phase":
                    TitleTxt1 = ["Theta", "rad", "Phase"]



                if self.GlobalVswpCount%self.PlotbeforeFigures == 0:
                    self.SaveDatatoOrigin(TitleTxt1, Plot=1)
                else:
                    self.SaveDatatoOrigin(TitleTxt1)

        # print('---------------------- All Finished (Total Time:'+TimeFormat(
        #                 TimeSpend)+') ----------------------')
        syst.stdout.write("\r{0}".format('------------------------- All Finished (Total:' + TimeFormat(
            TimeSpend) + '/Ave:' + TimeFormat(elapsed_tol / self.GlobalRunCount) + '/run) -------------------------'))
        syst.stdout.flush()

    def SaveDatatoOrigin(self, TitleTxtX, Plot=0):


        DataState = pd.DataFrame(self.SAVENOTE)
        DataState.to_excel(
            self.OriginFilePath + self.SaveTime + '-SwpDetail.xlsx',
            index=False, header=False)
        TitleTxtY1 = ["G", "2e^2/h", self.SAVEFILENAME_origin + '_Conductance']

        Data = [list(a) for a in zip(TitleTxtX + list(self.VarSwp), TitleTxtY1 + list(self.conductances))]
        if self.SeriesR != 0:
            if self.BlockWarnings:
                warnings.filterwarnings("ignore")
            D_R = (1 / (self.SeriesR + 1 / (7.74809173e-5 * np.array(self.conductances)))) / 7.74809173e-5
            if self.BlockWarnings:
                warnings.filterwarnings("always")
            Data_R = [list(a) for a in zip(TitleTxtX + list(self.VarSwp), TitleTxtY1 + list(D_R))]


        if self.GlobalVswpCount == 1:
            savedata(self.OriginFilePath + self.SaveTime + '.txt',
                     init=True, initdata=Data)
            if self.SeriesR != 0:
                savedata(self.OriginFilePath + self.SaveTime + '-' + str(
                    round(self.SeriesR, 3)) + '.txt', init=True, initdata=Data_R)
        else:
            savedata(self.OriginFilePath + self.SaveTime + '.txt',
                     init=False, newcol=Data)
            if self.SeriesR != 0:
                savedata(self.OriginFilePath + self.SaveTime + '-' + str(
                    round(self.SeriesR, 3)) + '.txt', init=False, newcol=Data_R)
        if self.SwpID == 'E':
            self.Gen_Conduct_Plot(1000*self.VarSwp*self.t, self.conductances, self.SwpID+self.SwpUnit)
        else:
            self.Gen_Conduct_Plot(self.VarSwp, self.conductances, self.SwpID+self.SwpUnit)
        self.fig.savefig(self.SAVEFILENAME +self.LocalSave+ "-Conductance.png")

        if Plot == 1:
            self.fig.show()

        if self.SN == 'SN':

            TitleTxtY2 = ["G", "2e^2/h", self.SAVEFILENAME_origin + '_N-Ree+Reh']

            Data_2 = [list(a) for a in zip(TitleTxtX + list(self.VarSwp), TitleTxtY2 + list(self.conductances2))]

            if self.SeriesR != 0:
                if self.BlockWarnings:
                    warnings.filterwarnings("ignore")

                D_R2 = (1 / (self.SeriesR + 1 / (7.74809173e-5 * np.array(self.conductances2)))) / 7.74809173e-5
                if self.BlockWarnings:
                    warnings.filterwarnings("always")

                Data_R_2 = [list(a) for a in zip(TitleTxtX + list(self.VarSwp), TitleTxtY2 + list(D_R2))]


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
                self.Gen_Conduct_Plot(1000*self.VarSwp*self.t, self.conductances2, self.SwpID+self.SwpUnit)
            else:
                self.Gen_Conduct_Plot(self.VarSwp, self.conductances2, self.SwpID+self.SwpUnit)
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


