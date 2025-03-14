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
weight_path = '/lustre/fswork/projects/rech/cli/udp79td/local_libs/morays/NEMO-C1D_PAPA32/C1D_PAPA32.W25ANN/INFERENCES/weights'


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


def W25ann(ux,uy,To,Ta,p,q):
    """ Compute air-sea momentum and heat fluxes from Wu et al. (2025) ANN """
    if Is_None(ux,uy,To,Ta,p,q):
        return None, None, None, None
    else:
        global net
        M, SH, LH = net

        # 3D to 1D
        ux = ux.reshape(1)
        uy = uy.reshape(1)
        To = To.reshape(1)
        Ta = Ta.reshape(1)
        p = p.reshape(1)
        q = q.reshape(1)

        # useful variables
        U = (ux**2 + uy**2)**0.5
        cos = ux/U
        sin = uy/U
        rh = rhcalc(Ta, p/100. , q)

        # Format inputs model
        X = np.hstack([U.reshape(-1,1), To.reshape(-1,1), Ta.reshape(-1,1),
                      rh.reshape(-1,1), p.reshape(-1,1)]).astype('float32')
        X = torch.tensor(X)

        # Predict fluxes
        M_mean = M.pred_mean(X)
        M_std = M.pred_var(X) ** 0.5
        SH_mean = SH.pred_mean(X)
        SH_std = SH.pred_var(X) ** 0.5
        LH_mean = LH.pred_mean(X)
        LH_std = LH.pred_var(X) ** 0.5

        # Format outputs
        taux = M_mean.detach().numpy().squeeze().reshape(1,1,1) * cos
        tauy = M_mean.detach().numpy().squeeze().reshape(1,1,1) * sin
        Qs = SH_mean.detach().numpy().squeeze().reshape(1,1,1)
        Ql = LH_mean.detach().numpy().squeeze().reshape(1,1,1)
        return taux, tauy, Qs, Ql


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
