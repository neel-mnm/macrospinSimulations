import numpy as np
from scipy.integrate import solve_ivp
from numba import njit, typed
import matplotlib.pyplot as plt
import copy
import os
import pickle as pkl
import tqdm
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from numba.core.registry import CPUDispatcher

gamma0 = 1.76e11
hbar = 1.054571817e-34
muB = 9.2740100657e-24
q_e = 1.60217663e-19
mu0 = 4*np.pi*1e-7



#=========
#UTILITIES
#=========
@njit
def noEffect(t, shape):
    return np.zeros(shape)
        
def sanitizeParameters(parameters):
    for key in parameters:
        val = parameters[key]
        if isinstance(val, (int, float)):
            parameters[key] = float(val)
        elif isinstance(val, (list, np.ndarray)):
            parameters[key] = np.array(val, dtype=np.float64)
        elif isinstance(val, bool):
            parameters[key] = bool(val)
    return parameters


@njit
def RK4_integrator(J0, t0, tf, dt, func):
    n_step = int((tf-t0)/dt)
    N = J0.shape[1]

    J = np.empty((6,N,n_step+1), dtype = np.float64)
    timesteps = np.empty(n_step+1, dtype = np.float64)

    J[:,:,0]=J0
    timesteps[0] = t0

    for i in range(n_step):
        t = timesteps[i]
        y = J[:,:,i]
        k1 = func(t,y)
        k2 = func(t+0.5*dt, y+0.5*dt*k1)
        k3 = func(t+0.5*dt, y+0.5*dt*k2)
        k4 = func(t+dt, y+dt*k3)

        J[:,:,i+1] = y+dt/6*(k1+2*k2+2*k3+k4)
        timesteps[i+1] = t+dt
    return timesteps, J

@njit
def RK4_stream_integrator(J0, t0, tf, dt, func, save_every = 100):
    n_step = int((tf-t0)/dt)
    N = J0.shape[1]

    y = J0.astype(np.float64)
    t = t0

    n_save = n_step//save_every + 1
    J_save = np.empty((6,N,n_save))
    t_save = np.empty(n_save)

    save_idx = 0
    J_save[:,:,save_idx] = J0
    t_save[save_idx]=t0
    save_idx+=1


    for i in range(n_step):
        k1 = func(t,y)
        k2 = func(t+0.5*dt, y+0.5*dt*k1)
        k3 = func(t+0.5*dt, y+0.5*dt*k2)
        k4 = func(t+dt, y+dt*k3)

        y += dt/6*(k1+2*k2+2*k3+k4)
        t+=dt

        if i%save_every == 0:
            J_save[:,:,save_idx] = y
            t_save[save_idx] = t
            save_idx+=1

    return t_save, J_save



class SimulationParams:
    def __init__(self, plist):
        self.B0 = np.array([p["B0"] for p in plist], dtype=np.float64)
        self.uB = np.array([p["uB"] for p in plist], dtype=np.float64)

        self.M = np.array([p["M"] for p in plist], dtype=np.float64)
        self.N_diag = np.array([p["N_diag"] for p in plist], dtype=np.float64)

        self.Bk = np.array([p["Bk"] for p in plist], dtype=np.float64)
        self.uK = np.array([p["uK"] for p in plist], dtype=np.float64)

        self.anisotropy_affects_S = np.array(
            [p["anisotropy_affects_S"] for p in plist], dtype=np.float64
        )

        self.BOe = np.array([p["BOe"] for p in plist], dtype=np.float64)

        self.Bso = np.array([p["Bso"] for p in plist], dtype=np.float64)

        self.S_fraction = np.array([p["S_fraction"] for p in plist], dtype=np.float64)

        self.gL = np.array([p["gL"] for p in plist], dtype=np.float64)
        self.gS = np.array([p["gS"] for p in plist], dtype=np.float64)
        self.alphaL = np.array([p["alphaL"] for p in plist], dtype=np.float64)
        self.alphaS = np.array([p["alphaS"] for p in plist], dtype=np.float64)

        self.s = np.array([p["s"] for p in plist], dtype=np.float64)
        self.l = np.array([p["l"] for p in plist], dtype=np.float64)
        self.Bfl_l = np.array([p["Bfl_l"] for p in plist], dtype = np.float64)
        self.Bdl_l = np.array([p["Bdl_l"] for p in plist], dtype = np.float64)
        self.Bfl_s = np.array([p["Bfl_s"] for p in plist], dtype = np.float64)
        self.Bdl_s = np.array([p["Bdl_s"] for p in plist], dtype = np.float64)

        self.N = len(plist)

        self.useB0   =     bool(np.any(self.B0 !=0))  
        self.useM    =     bool(np.any(self.M  !=0))
        self.useBk   =     bool(np.any(self.Bk !=0))
        self.useBoe  =     bool(np.any(self.BOe!=0))
        self.useBso  =     bool(np.any(self.Bso!=0))
        self.useBfl_l=     bool(np.any(self.Bfl_l!=0))
        self.useBfl_s=     bool(np.any(self.Bfl_s!=0))
        self.useBdl_l=     bool(np.any(self.Bdl_l!=0))
        self.useBdl_s=     bool(np.any(self.Bdl_s!=0))



def make_effective_field(parameters: SimulationParams):
    useB0   =parameters.useB0   
    useM    =parameters.useM    
    useBk   =parameters.useBk   
    useBoe  =parameters.useBoe  
    useBso  =parameters.useBso  
    useBfl_l=parameters.useBfl_l
    useBfl_s=parameters.useBfl_s
    useBdl_l=parameters.useBdl_l
    useBdl_s=parameters.useBdl_s


    N = parameters.N
    B0_set = parameters.B0
    uB_set = parameters.uB
    M_set = parameters.M
    S_frac_set = parameters.S_fraction
    N_diag_set = parameters.N_diag
    Bk_set = parameters.Bk
    uK_set = parameters.uK
    ani_on_S_set = parameters.anisotropy_affects_S
    BOe_set = parameters.BOe
    BSo_set = parameters.Bso
    gL_set = parameters.gL
    gS_set = parameters.gS
    s_set = parameters.s
    l_set = parameters.l
    Bfl_l_set = parameters.Bfl_l
    Bfl_s_set = parameters.Bfl_s
    Bdl_l_set = parameters.Bdl_l
    Bdl_s_set = parameters.Bdl_s    


    @njit
    def toReturn(I, J):
        out = np.empty((6,N), dtype = np.float64)
        for i in range(N):
            
            Sx = J[0,i]
            Sy = J[1,i]
            Sz = J[2,i]
            Lx = J[3,i]
            Ly = J[4,i]
            Lz = J[5,i]

            out_sx=0
            out_sy=0
            out_sz=0
            out_lx=0
            out_ly=0
            out_lz=0


            #external field
            if useB0:
                B0 = B0_set[i]
                uBx, uBy, uBz = uB_set[i]
                
                out_sx +=  B0 * uBx                
                out_sy +=  B0 * uBy
                out_sz +=  B0 * uBz
                out_lx +=  B0 * uBx
                out_ly +=  B0 * uBy
                out_lz +=  B0 * uBz

            #demag field
            if useM:
                M = M_set[i]
                f_S = S_frac_set[i]
                Nx, Ny, Nz = N_diag_set[i]
                
                Mx = mu0*M*(f_S * Sx + (1-f_S)*Lx)
                My = mu0*M*(f_S * Sy + (1-f_S)*Ly)
                Mz = mu0*M*(f_S * Sz + (1-f_S)*Lz)
                
                out_sx +=  -Mx * Nx
                out_sy +=  -My * Ny
                out_sz +=  -Mz * Nz
                out_lx +=  -Mx * Nx
                out_ly +=  -My * Ny
                out_lz +=  -Mz * Nz

            #uniaxial anisotropy
            if useBk:
                Bk = Bk_set[i]
                uKx, uKy, uKz = uK_set[i]

                uK_dot_L = uKx*Lx + uKy*Ly + uKz*Lz
                uK_dot_S = ani_on_S_set[i]*(uKx*Sx + uKy*Sy + uKz*Sz)
                
                out_sx +=  Bk * uK_dot_S * uKx                
                out_sy +=  Bk * uK_dot_S * uKy
                out_sz +=  Bk * uK_dot_S * uKz
                out_lx +=  Bk * uK_dot_L * uKx
                out_ly +=  Bk * uK_dot_L * uKy
                out_lz +=  Bk * uK_dot_L * uKz

            #Oersted
            if useBoe:
                BOe = BOe_set[i] * I[i]

                out_sy +=  BOe
                out_ly +=  BOe

            
            #spin orbit
            if useBso:
                Bso = BSo_set[i]
                gL = gL_set[i]
                gS = gS_set[i]
                f_S = S_frac_set[i]

                out_sx += Bso * (1-f_S) * Lx/gL
                out_sy += Bso * (1-f_S) * Ly/gL
                out_sz += Bso * (1-f_S) * Lz/gL
                out_lx += Bso * f_S * Sx/gS
                out_ly += Bso * f_S * Sy/gS                    
                out_lz += Bso * f_S * Sz/gS                                
                                    
                                    
            #fieldLike
            if useBfl_s:
                sx,sy,sz = s_set[i]
                Bfl_s = Bfl_s_set[i] * I[i]
                out_sx += Bfl_s * sx 
                out_sy += Bfl_s * sy
                out_sz += Bfl_s * sz

            if useBfl_l:
                lx,ly,lz = l_set[i]
                Bfl_l = Bfl_l_set[i] * I[i]
                out_lx += Bfl_l * lx 
                out_ly += Bfl_l * ly
                out_lz += Bfl_l * lz


            #dampingLike
            if useBdl_s:
                sx,sy,sz = s_set[i]
                cross_x = sz*Sy - sy*Sz
                cross_y = -sz*Sx + sx*Sz
                cross_z = sy*Sx - sx*Sy
                Bdl_s = Bdl_s_set[i] * I[i]
                out_sx += Bdl_s * cross_x 
                out_sy += Bdl_s * cross_y
                out_sz += Bdl_s * cross_z

            if useBdl_l:
                lx,ly,lz = l_set[i]
                cross_x = lz*Ly - ly*Lz
                cross_y = -lz*Lx + lx*Lz
                cross_z = ly*Lx - lx*Ly
                Bdl_l = Bdl_l_set[i] * I[i]
                out_lx += Bdl_l * cross_x 
                out_ly += Bdl_l * cross_y
                out_lz += Bdl_l * cross_z

            
            out[0,i] = out_sx
            out[1,i] = out_sy
            out[2,i] = out_sz
            out[3,i] = out_lx
            out[4,i] = out_ly
            out[5,i] = out_lz

        return out
    return toReturn


def get_LLGs(parameters: SimulationParams, dynamics = "full", currentFunc = None):
    alphaS_set = parameters.alphaS
    alphaL_set = parameters.alphaL
    gS_set = parameters.gS
    gL_set = parameters.gL
    N = parameters.N

    gamma_base = -muB / hbar
    prefactors_S = gS_set * gamma_base / (1 + alphaS_set**2)
    prefactors_L = gL_set * gamma_base / (1 + alphaL_set**2)

    field_func = make_effective_field(parameters)

    fullMode = dynamics=="full"
    dampingMode = dynamics=="damping"
    precessionMode = dynamics=="precession"

    if currentFunc is None:
        @njit
        def currentFunc(t):
            return noEffect(t,N)
        
    if not isinstance(currentFunc, CPUDispatcher):
        raise TypeError(f"{currentFunc.__name__} must be a numba compiled function")

    @njit
    def LLG(t,J):
        dJdt = np.empty((6, N), dtype=np.float64)
        I = currentFunc(t)
        B = field_func(I,J)
        for i in range(N):
            

            

            Sx = J[0,i]
            Sy = J[1,i]
            Sz = J[2,i]
            Lx = J[3,i]
            Ly = J[4,i]
            Lz = J[5,i]

            BSx = B[0,i]
            BSy = B[1,i]
            BSz = B[2,i]
            BLx = B[3,i]
            BLy = B[4,i]
            BLz = B[5,i]

            pref_S = prefactors_S[i]
            pref_L = prefactors_L[i]
            alpha_S = alphaS_set[i]
            alpha_L = alphaL_set[i]



            px = Sy*BSz -  BSy*Sz
            py = -Sx*BSz +  BSx*Sz
            pz = Sx*BSy -  BSx*Sy

            

            if fullMode:
                dx = Sy*pz -  py*Sz
                dy = -Sx*pz +  px*Sz
                dz = Sx*py -  px*Sy

                sx = px + alpha_S * dx
                sy = py + alpha_S * dy
                sz = pz + alpha_S * dz
            elif dampingMode:
                dx = Sy*pz -  py*Sz
                dy = -Sx*pz +  px*Sz
                dz = Sx*py -  px*Sy

                sx = alpha_S * dx
                sy = alpha_S * dy
                sz = alpha_S * dz
            else:  # precession
                sx = px
                sy = py
                sz = pz


            dJdt[0,i]=pref_S * sx
            dJdt[1,i]=pref_S * sy
            dJdt[2,i]=pref_S * sz



            px = Ly*BLz -  BLy*Lz
            py = -Lx*BLz +  BLx*Lz
            pz = Lx*BLy -  BLx*Ly


            if fullMode:
                dx = Ly*pz -  py*Lz
                dy = -Lx*pz +  px*Lz
                dz = Lx*py -  px*Ly

                lx = px + alpha_L * dx
                ly = py + alpha_L * dy
                lz = pz + alpha_L * dz
            elif dampingMode:
                dx = Ly*pz -  py*Lz
                dy = -Lx*pz +  px*Lz
                dz = Lx*py -  px*Ly

                lx = alpha_L * dx
                ly = alpha_L * dy
                lz = alpha_L * dz
            else:  # precession
                lx = px
                ly = py
                lz = pz


            dJdt[3,i]=pref_L * lx
            dJdt[4,i]=pref_L * ly
            dJdt[5,i]=pref_L * lz


        return dJdt
    


    return LLG


def timeEvol(J0, parameters, fmrFieldFunction, tf, dt = 1e-10, t0 = 0, dynamics = "full", save_every = 100, stream = False):
    LLGs = get_LLGs(parameters, dynamics, fmrFieldFunction)
    n_step = int((tf-t0)/dt)
    
    N = J0.shape[1]

    estimated_bytes = 6 * N * (n_step+1) * 8

    if estimated_bytes > 1e9 or stream:  # ~1 GB threshold
        print("Using streaming RK4")
        return RK4_stream_integrator(J0, t0, tf, dt, LLGs, save_every=save_every)
    else:
        print("Using full RK4")
        return RK4_integrator(J0, t0, tf, dt, LLGs)




if __name__ ==   "__main__":
    import time

    t0 = time.perf_counter()

    for s_frac in [.5]:
        print("Starting s_frac = ", s_frac)
        parameters_base={
            "B0":1,
            "uB":np.array([0,1,0]),
            "M": 0/mu0,
            "N_diag": np.array([0,0,1]),
            "Bk": 0,
            "uK": np.array([1,0,0]),
            "anisotropy_affects_S": False,
            "BOe": 1e-4,
            "Bfl_l": 0,
            "Bdl_l": 0,
            "Bfl_s": 0,
            "Bdl_s": 0,
            "s": np.array([0,1,0]),
            "l": np.array([0,1,0]),
            "Bso": 0,
            "S_fraction": s_frac,
            "alphaS": 0.01,
            "alphaL" : 0.01,
            "gL":1,
            "gS":2
        }

        length = 1

        parameters_base = sanitizeParameters(parameters_base)

        parametersList = [parameters_base for i in range(length)]




        B_so_list = np.concatenate([-np.logspace(3, -2,50),np.array([0]),np.logspace(-2, 3,50)])


        parameters_list = [copy.deepcopy(parameters_base) for B in B_so_list]

        for parameter, B in zip(parameters_list,B_so_list):
            parameter["Bso"] = B 

        J0_total = np.array([[1,0,0,1,0,0] for B in B_so_list]).T

        f_max = 100e12
        f_min = 5e9
        T = 10e-9
        max_step = 1/(10*f_max)
        sigma = max_step

        k = np.log(f_max/f_min)
        
        t_0 = 1e-9

        N = len(parameters_list)
        parameters_list = SimulationParams(parameters_list)

        
        @njit
        def gaussian_pulse(t):
            #return np.ones(N)*np.exp(-(t-t_0)**2/(2*sigma**2))
            return np.sin(f_max*(t-t_0))*np.ones(N)

        B = make_effective_field(parameters_list)
        test=get_LLGs(parameters_list)

        #results = timeEvol(J0_total,parameters_list,None,1e-15, dt=1e-15)
        
        results = timeEvol(J0_total,parameters_list,None,1e-8, dt=1e-15, save_every = 100)
        print(np.shape(results[1]))

    t1 = time.perf_counter()

    print(f"sim time: {t1-t0:.6f} s")