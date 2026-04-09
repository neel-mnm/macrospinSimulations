import numpy as np
from scipy.integrate import solve_ivp
from numba import njit, typed
import matplotlib.pyplot as plt
import copy
import os
import pickle as pkl
from scipy.interpolate import interp1d

gamma0 = 1.76e11
hbar = 1.054571817e-34
muB = 9.2740100657e-24
q_e = 1.60217663e-19
mu0 = 4*np.pi*1e-7



#=========
#UTILITIES
#=========
@njit
def cross_inline(ax, ay, az, bx, by, bz):
    return (
        ay*bz - az*by,
        az*bx - ax*bz,
        ax*by - ay*bx
    )

@njit
def dot_inline(a, b):
    return np.array((i*j for i,j in zip(a,b)))

@njit
def noEffect(t):
    return 0
        
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

#===========
# Field and LLG definition

def totalField_multi(parameters_list):
    
    #unpack parameters
    B0_list = np.array([parameters["B0"] for parameters in parameters_list])
    uB_list = np.array([parameters["uB"]  for parameters in parameters_list])
    M_list = np.array([parameters["M"] for parameters in parameters_list])
    N_diag_list = np.array([parameters["N_diag"] for parameters in parameters_list])
    Bk_list = np.array([parameters["Bk"] for parameters in parameters_list])
    uK_list = np.array([parameters["uK"] for parameters in parameters_list])
    anisotropy_affects_S_list = np.array([parameters["anisotropy_affects_S"] for parameters in parameters_list])
    BOe_list = np.array([parameters["BOe"] for parameters in parameters_list])
    Bfl_l_list = np.array([parameters["Bfl_l"] for parameters in parameters_list])
    Bdl_l_list = np.array([parameters["Bdl_l"] for parameters in parameters_list])
    Bfl_s_list = np.array([parameters["Bfl_s"] for parameters in parameters_list])
    Bdl_s_list = np.array([parameters["Bdl_s"] for parameters in parameters_list])
    s_list = np.array([parameters["s"] for parameters in parameters_list])
    l_list = np.array([parameters["l"] for parameters in parameters_list])
    Bso_list = np.array([parameters["Bso"] for parameters in parameters_list])
    S_fraction_list = np.array([parameters["S_fraction"] for parameters in parameters_list])
    gL_list, gS_list = np.array([parameters["gL"] for parameters in parameters_list]),np.array([parameters["gS"]  for parameters in parameters_list])

    lattice_number = len(parameters_list)
    
    @njit
    def toReturn(I_list,J_total):
        out = np.zeros(6*lattice_number, dtype=np.float64)
        for i in range(lattice_number):
            lattice_index = 6*i
            
            #utilities for not evaluating things over too many loops
            B0 = B0_list[i]
            uB = uB_list[i]
            
            
            Ms = M_list[i]
            S_fraction = S_fraction_list[i]
            N_diag = N_diag_list[i]
            
            
            uK = uK_list[i]
            Bk = Bk_list[i]
            anisotropy_affects_S = anisotropy_affects_S_list[i]
            uK_dot_L = J_total[lattice_index+3]*uK[0]+J_total[lattice_index+4]*uK[1]+J_total[lattice_index+5]*uK[2]
            uK_dot_S = anisotropy_affects_S*(J_total[lattice_index]*uK[0]+J_total[lattice_index+1]*uK[1]+J_total[lattice_index+2]*uK[2])
            
            
            BOe = BOe_list[i]*I_list[i]
            out[lattice_index+1] = BOe
            out[lattice_index+4] = BOe
            
            
            Bso = Bso_list[i]
            gL = gL_list[i]
            gS = gS_list[i]
            
            for j in range(3):
                #external field
                val = B0*uB[j]
                out[lattice_index+j]    = out[lattice_index+j]   + val
                out[lattice_index+j+3]  = out[lattice_index+j+3] + val
                
                
                
                #demag field
                M_ij = mu0 * Ms * (S_fraction * J_total[lattice_index+j] + (1-S_fraction) * J_total[lattice_index+j+3])
                val = -N_diag[j]*M_ij
                
                out[lattice_index+j]   = out[lattice_index+j]   + val 
                out[lattice_index+j+3] = out[lattice_index+j+3] + val
                


                #uniaxial anisotropy
                out[lattice_index+j]  = out[lattice_index+j]   + Bk * uK_dot_S * uK[j]
                out[lattice_index+j+3]= out[lattice_index+j+3] + Bk * uK_dot_L * uK[j]
                
                
                
                #Oersted Field is j- independent so it's dealt with in the outer loop

                
                #Spin-Orbit field
                out[lattice_index+j]  = out[lattice_index+j]   + (1-S_fraction) * Bso * J_total[lattice_index+j+3]/gL
                out[lattice_index+j+3]= out[lattice_index+j+3] + S_fraction *     Bso * J_total[lattice_index+j]/gS
                

        #out[np.abs(out) < 1e-10] = 0.0
        return out
        
    return toReturn


def get_LLGs_multi(parameters_list, dynamics = "full"):
    alphaS_list = np.array([parameters["alphaS"] for parameters in parameters_list])
    alphaL_list = np.array([parameters["alphaL"] for parameters in parameters_list])
    gL_list = np.array([parameters["gL"] for parameters in parameters_list])
    gS_list = np.array([parameters["gS"] for parameters in parameters_list])

    gamma_base = -muB / hbar
    prefactors_S = gS_list * gamma_base / (1 + alphaS_list**2)
    prefactors_L = gL_list * gamma_base / (1 + alphaL_list**2)

    
    
    B_func = totalField_multi(parameters_list)

    lattice_number = len(parameters_list)
    dataset_length = 6*len(parameters_list)
    
    @njit
    def fullLLGs(I_list,J_total):
        dJdt = np.empty(dataset_length, dtype=np.float64)
        B = B_func(I_list, J_total)

        for i in range(lattice_number):
            lattice_position = 6*i

            # --- S ---
            px, py, pz = cross_inline(J_total[lattice_position+0], J_total[lattice_position+1], J_total[lattice_position+2],
                                      B[lattice_position+0], B[lattice_position+1], B[lattice_position+2])
            dx, dy, dz = cross_inline(J_total[lattice_position+0], J_total[lattice_position+1], J_total[lattice_position+2],
                                      px, py, pz)

            dJdt[lattice_position+0] = prefactors_S[i] * (px + alphaS_list[i] * dx)
            dJdt[lattice_position+1] = prefactors_S[i] * (py + alphaS_list[i] * dy)
            dJdt[lattice_position+2] = prefactors_S[i] * (pz + alphaS_list[i] * dz)

            # --- L ---
            px, py, pz = cross_inline(J_total[lattice_position+3], J_total[lattice_position+4], J_total[lattice_position+5],
                                      B[lattice_position+3], B[lattice_position+4], B[lattice_position+5])
            dx, dy, dz = cross_inline(J_total[lattice_position+3], J_total[lattice_position+4], J_total[lattice_position+5],
                                      px, py, pz)

            dJdt[lattice_position+3] = prefactors_L[i] * (px + alphaL_list[i] * dx)
            dJdt[lattice_position+4] = prefactors_L[i] * (py + alphaL_list[i] * dy)
            dJdt[lattice_position+5] = prefactors_L[i] * (pz + alphaL_list[i] * dz)
            

        #dJdt[np.abs(dJdt) < 1e-15] = 0.0
        return dJdt
    
    @njit
    def dampingOnly(I_list,J_total):
        dJdt = np.empty(dataset_length, dtype=np.float64)
        B = B_func(I_list, J_total)

        for i in range(len(I_list)):
            lattice_position = 6*i

            # --- S ---
            px, py, pz = cross_inline(J_total[lattice_position+0], J_total[lattice_position+1], J_total[lattice_position+2],
                                      B[lattice_position+0], B[lattice_position+1], B[lattice_position+2])
            dx, dy, dz = cross_inline(J_total[lattice_position+0], J_total[lattice_position+1], J_total[lattice_position+2],
                                      px, py, pz)

            dJdt[lattice_position+0] = prefactors_S[i] * (alphaS_list[i] * dx)
            dJdt[lattice_position+1] = prefactors_S[i] * (alphaS_list[i] * dy)
            dJdt[lattice_position+2] = prefactors_S[i] * (alphaS_list[i] * dz)

            # --- L ---
            px, py, pz = cross_inline(J_total[lattice_position+3], J_total[lattice_position+4], J_total[lattice_position+5],
                                      B[lattice_position+3], B[lattice_position+4], B[lattice_position+5])
            dx, dy, dz = cross_inline(J_total[lattice_position+3], J_total[lattice_position+4], J_total[lattice_position+5],
                                      px, py, pz)

            dJdt[lattice_position+3] = prefactors_L[i] * (alphaL_list[i] * dx)
            dJdt[lattice_position+4] = prefactors_L[i] * (alphaL_list[i] * dy)
            dJdt[lattice_position+5] = prefactors_L[i] * (alphaL_list[i] * dz)

        dJdt[np.abs(dJdt) < 1e-15] = 0.0
        return dJdt
    
    @njit
    def precessionOnly(I_list,J_total):
        dJdt = np.empty(dataset_length, dtype=np.float64)
        B = B_func(I_list, J_total)

        for i in range(len(I_list)):
            lattice_position = 6*i

            # --- S ---
            px, py, pz = cross_inline(J_total[lattice_position+0], J_total[lattice_position+1], J_total[lattice_position+2],
                                      B[lattice_position+0], B[lattice_position+1], B[lattice_position+2])
            
            dJdt[lattice_position+0] = prefactors_S[i] * (px)
            dJdt[lattice_position+1] = prefactors_S[i] * (py)
            dJdt[lattice_position+2] = prefactors_S[i] * (pz)

            # --- L ---
            px, py, pz = cross_inline(J_total[lattice_position+3], J_total[lattice_position+4], J_total[lattice_position+5],
                                      B[lattice_position+3], B[lattice_position+4], B[lattice_position+5])
            
            dJdt[lattice_position+3] = prefactors_L[i] * (px)
            dJdt[lattice_position+4] = prefactors_L[i] * (py)
            dJdt[lattice_position+5] = prefactors_L[i] * (pz)

        dJdt[np.abs(dJdt) < 1e-15] = 0.0
        return dJdt
    
    match dynamics:
       case "full":
           toReturn = fullLLGs
       case "damping":
           toReturn = dampingOnly
       case "precession":
           toReturn = precessionOnly    

    return toReturn



#=========
#Dynamics
#=========

def findEquilibrium_multi_withDamping(J0_total, fmrFieldFunction_list,parameters_list, t_max = 1e-6, torquetol = 1e-9, returnEvolution = False, rel_tol=1e-10, abs_tol=1e-10, method="Radau"):
    parameters_copy = copy.deepcopy(parameters_list)
    for parameter_set in parameters_copy:
        parameter_set["alphaS"] = 1
        parameter_set["alphaL"] = 1

    LLGs = get_LLGs_multi(parameters_copy, "damping")
    t_span = (0, t_max)

    
    def LLG_wrapper(t, J_total):
        I_list = np.array([fun(t) for fun in fmrFieldFunction_list], dtype=np.float64)
        #for i in range(len(fmrFieldFunction_list)):
        #    I_list[i] = fmrFieldFunction_list[i](t)
        
        return LLGs(I_list, J_total)

    def equilibrium_event(t,J):
        dJdt = LLG_wrapper(t,J)
        return np.sqrt(np.sum(dJdt**2))-torquetol
    equilibrium_event.terminal = True
    equilibrium_event.direction = -1

    sol = solve_ivp(
        LLG_wrapper,
        t_span,
        J0_total,
        rtol = rel_tol,
        atol = abs_tol,
        method=method,
        events=equilibrium_event
    )

    if returnEvolution:
        return sol
    else:
        return sol.y[:,-1]


def timeEvolution_multi(J0_total, parameters_list, fmrFieldFunction_list, t_f, t_i=0, rel_tol=1e-10, abs_tol=1e-10, method="Radau", relaxFirst = True, max_step = 1e-8, min_step = None):
    """
    J0 : initial 6-vector
    LLG_numba : Numba'd derivative function LLG(I, J)
    fmrFieldFunction : function of t returning I(t)
    """
    t_span = (t_i, t_f)

    LLG_numba = get_LLGs_multi(parameters_list)
    
    
    def LLG_wrapper(t, J_total):
        I_list = np.array([fun(t) for fun in fmrFieldFunction_list], dtype=np.float64)
        #for i in range(len(fmrFieldFunction_list)):
        #    I_list[i] = fmrFieldFunction_list[i](t)
        
        return LLG_numba(I_list, J_total)
    
    if relaxFirst:
        J0_total = findEquilibrium_multi_withDamping(J0_total, [noEffect for i in range(len(fmrFieldFunction_list))], parameters_list)
        print("equilibrium reached")
        
    sol = solve_ivp(
        LLG_wrapper,
        t_span,
        J0_total,
        rtol=rel_tol,
        atol=abs_tol,
        method=method,
        max_step=max_step,
        min_step = min_step
    )
    return sol


class SimSolution():
    def __init__(self, parameters, solution):
        self.t = solution.t
        self.parameters = parameters
        self.J = solution.y

class MultiSimSolution(SimSolution):
    def __init__(self, parameters_list, solution):
        super().__init__(parameters_list, solution)
        self.num_layers = len(parameters_list)
        self.J = self.J.reshape(self.num_layers, 6, -1)

    def getM(self):
        M_S = np.array([self.parameters[i]["S_fraction"]*self.J[i,:3,:] for i in range(len(self.parameters))])
        M_L = np.array([(1-self.parameters[i]["S_fraction"])*self.J[i,3:,:] for i in range(len(self.parameters))])
        return M_S+M_L

    def getFFT(self, t0 = 0, actOn="M"):

        match actOn:
            case "M":
                data = self.getM
                attributeName = "fft_M"
            case "S":
                data = self.J[:,:3,:]
                attributeName = "fft_S"
            case "L":
                data = self.J[:,3:,:]  
                attributeName = "fft_L"  


        uniform_t = np.linspace(self.t[0], self.t[-1], 10*len(self.t))
        interp = interp1d(self.t, data[:,1,:], kind = "cubic", axis = -1)
        Y_uniform = interp(uniform_t)

        post_idx = uniform_t>t0
        fft_y = Y_uniform[post_idx]
        dt = uniform_t[1]-uniform_t[0]
        freqs = np.fft.fftfreq(fft_y.shape[-1], dt)
        Y_fft = np.fft.fft(fft_y, axis = -1)

        #for index in parameterSet_indeces:
        #    uniform_t = np.linspace(self.t[0], self.t[-1],10*len(self.t))
        #    interp = interp1d(self.t, data[index,1,:], kind = "cubic")
        #    My_uniform = interp(uniform_t)
        #    post_idx = uniform_t > t0
        #    #fft_t = uniform_t[post_idx]
        #    fft_my = My_uniform[post_idx]
        #    dt = uniform_t[1]-uniform_t[0]
        #    freqs = np.fft.fftfreq(len(fft_my), dt)
        #    fftMy = np.fft.fft(fft_my)



        setattr(self,attributeName,Y_fft)
        if not hasattr(self,"fftFreqs"):
            self.fftFreqs = freqs
        




if __name__ ==   "__main__":
    for s_frac in [0,.1,.2,.3,.4,.5,.6,.7,.7,.8,.9,.95,.99]:
        print("Starting s_frac = ", s_frac)
        parameters_base={
            "B0":1,
            "uB":np.array([1,0,0]),
            "M": 0/mu0,
            "N_diag": np.array([0,0,1]),
            "Bk": 0,
            "uK": np.array([1,0,0]),
            "anisotropy_affects_S": False,
            "BOe": 1e-3,
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
        test=get_LLGs_multi(parametersList)
        B = totalField_multi(parametersList)

        #B_so_list = [.01,0,1]
        #B_so_list = B_so_list + [*reversed([-B for B in B_so_list])]

        def tanh_space(n, val_range, val_scale = 3):
            x = np.linspace(-.99,.99,n)
            y = np.tan(x*np.pi/2)
            return val_range * y

        B_so_list = np.concatenate([-np.logspace(3, -2,100),np.array([0]),np.logspace(-2, 3,100)])

        #B_so_list = np.linspace(-100, 100, 100)

        parameters_list = [copy.deepcopy(parameters_base) for B in B_so_list]

        for parameter, B in zip(parameters_list,B_so_list):
            parameter["Bso"] = B 

        J0_total = np.array([[1,0,0,1,0,0] for B in B_so_list]).flatten()

        f_max = 100e12
        t_0 = 5e-9

        @njit
        def sinc_pulse(t):
            return np.sinc(f_max*(t-t_0))

        function_list = [sinc_pulse for i in B_so_list]

        plt.scatter([i for i in range(len(B_so_list))],B_so_list)

        results = timeEvolution_multi(J0_total,parameters_list,function_list,100e-9, relaxFirst=True, max_step=1e-10)
        

        temp = MultiSimSolution(parameters_list,results)

        os.makedirs(r"D:\data\sims", exist_ok=True)

        with open(rf"D:\data\sims/fraction{int(s_frac*100)}%.pkl","wb") as f:
            pkl.dump(temp, f) 