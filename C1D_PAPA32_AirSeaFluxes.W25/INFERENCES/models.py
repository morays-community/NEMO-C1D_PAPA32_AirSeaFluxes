"""
Contains User Inference/Analytic Models or Functions.

A model must fit the following requisites and structure :
-------------------------------------------------------
    1. must be a callable function that takes N numpy arrays as inputs
    2. /!\ returns N None for the N awaited outputs if at least one of the input is None /!\
    3. inputs may be freely formatted and transformed into what you want BUT...
    4. ...outputs must be formatted as numpy array for sending back
"""
import numpy as np
import torch
from mlflux.utils import rhcalc
from mlflux.ann_case import open_case


# ============================= #
# -- User Defined Parameters --
# ============================= #
weight_path = '/lustre/fswork/projects/rech/rkm/udp79td/local_libs/morays/NEMO-C1D_PAPA32_AirSeaFluxes/C1D_PAPA32_AirSeaFluxes.W25/INFERENCES/weights'

# ++++++++++++++++++++++++++++ #
#            Utils             #
# ++++++++++++++++++++++++++++ #
def Is_None(*inputs):
    """ Test presence of at least one None in inputs """
    return any(item is None for item in inputs)

def load_model(path):
    try: # in user dir or...
        M = open_case(path+'/M/', 'm.p')
        SH = open_case(path+'/SH/', 'sh.p')
        LH = open_case(path+'/LH/', 'lh.p')
    except: # in repo
        M = open_case('weights/M/', 'm.p')
        SH = open_case('weights/SH/', 'sh.p')
        LH = open_case('weights/LH/', 'lh.p')
    return M, SH, LH


# ------------------------------ #
#       Main Model Routines      #
# ------------------------------ #
# use GPUs if available
if torch.cuda.is_available():
    print("CUDA Available")
    device = torch.device('cuda')
else:
    print('CUDA Not Available')
    device = torch.device('cpu')

# Load model
net = load_model(weight_path)

# save previous time step
tau0, Qs0, Ql0 = None, None, None

def stoch_process(x0,xstd,alpha=0.99):
    """ Generate time-correlated random flucutation from given standard deviation. alpha = 1-dt/T with dt=0.5 hrs and T=60 hrs by default"""
    if x0 == None:
        x0 = np.random.normal(loc=0,scale=xstd)
    return alpha*x0 + (1-alpha**2)**0.5 * np.random.normal(loc=0,scale=xstd)


def W25ann(ux,uy,To,Ta,p,q):
    """ Compute air-sea momentum and heat fluxes from Wu et al. (2025) ANN """
    if Is_None(ux,uy,To,Ta,p,q):
        return None, None, None, None
    else:
        global net, tau0, Qs0, Ql0
        M, SH, LH = net

        # wind speed - rh
        U = (ux**2 + uy**2)**0.5
        cos = ux/U
        sin = uy/U
        rh = rhcalc(Ta, p/100. , q)

        # Format inputs model
        X = np.hstack([U.reshape(-1,1), To.reshape(-1,1), Ta.reshape(-1,1),
                      rh.reshape(-1,1), p.reshape(-1,1)]).astype('float32')
        X = torch.tensor(X)

        # Predict fluxes
        M_mean = M.pred_mean(X).detach().numpy().squeeze()
        M_std = M.pred_var(X).detach().numpy().squeeze() ** 0.5
        SH_mean = SH.pred_mean(X).detach().numpy().squeeze()
        SH_std = SH.pred_var(X).detach().numpy().squeeze() ** 0.5
        LH_mean = LH.pred_mean(X).detach().numpy().squeeze()
        LH_std = LH.pred_var(X).detach().numpy().squeeze() ** 0.5

        # Stochastic fluctuations
        tau0 = stoch_process(tau0,M_std,alpha=0.972)
        taux, tauy = (M_mean + tau0)*cos.ravel() , (M_mean + tau0)*sin.ravel()
        Qs0 = stoch_process(Qs0,SH_std)
        Qs = SH_mean + Qs0
        Ql0 = stoch_process(Ql0,LH_std)
        Ql = LH_mean + Ql0

        return taux.reshape(ux.shape), tauy.reshape(ux.shape), Qs.reshape(ux.shape), Ql.reshape(ux.shape)


if __name__ == '__main__' :

    # Testing inputs
    # --------------
    ux = np.array([[[4]]])         # wind speed in x-dir  [m/s]
    uy = np.array([[[8]]])         # wind speed in y-dir  [m/s]
    q = np.array([[[0.005]]])      # specific humidity    [kg/kg]
    To = np.array([[[12]]])        # ocean temperature    [celsius]
    Ta = np.array([[[10]]])        # air temperature      [celsius]
    p = np.array([[[1.01]]])*10**5 # pressure a sea level [Pa]
    print(f'Inputs -- ux: {ux}, uy: {uy}, Spec.hum: {q}, Toce: {To}, Tair: {Ta}, P: {p}')

    # Run model
    # ---------
    taux, tauy, Qs, Ql = W25ann(ux, uy, To, Ta, p, q)
    print(f'Res    -- taux: {taux}, tauy: {tauy}, Qlatent: {Ql}, Qsensible: {Qs}')
    print(f'Test successful')
