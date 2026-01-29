import numpy as np
from scipy import optimize
import sympy as syp
import matplotlib.pyplot as plt
from tqdm import tqdm

gamma0= 1.76e11   #[s T]^-1
hbar = 	1.054571817e-34 #[J s]
muB = 9.2740100657e-24 #[J T-1]
q_e = 1.60217663e-19 #Coulomb


def get_vector(theta,phi):
    # converts (theta,phi)-->(mx,my,mz)
    theta,phi=np.deg2rad(theta),np.deg2rad(phi)
    mx=np.sin(theta)*np.cos(phi)
    my=np.sin(theta)*np.sin(phi)
    mz=np.cos(theta)
    try:
        toReturn =  np.stack((mx, my, mz), axis=-1)
    except: 
        mz = np.full_like(phi, np.cos(theta))
        toReturn = np.stack((mx, my, mz), axis=-1)
    
    return toReturn    
        
def normConstraint(x,norm):
    return x[0]**2+x[1]**2+x[2]**2-norm

class macrospinSystem():
    def __init__(self, gl=1, gs=2, muB_S=1, muB_L=0, gammaL=0.5*gamma0, gammaS=gamma0, alphaS=0, alphaL=0):
        Sx, Sy, Sz = syp.symbols(r"S_x,S_y,S_z")
        Lx, Ly, Lz = syp.symbols(r"L_x,L_y,L_z")
        g_l,g_s,gamma_L, gamma_S, alpha_L, alpha_S = syp.symbols(r"g_l,g_s,gamma_L, gamma_S, alpha_S, alpha_L")
        
        self.S = syp.Matrix([Sx,Sy,Sz])
        self.L = syp.Matrix([Lx,Ly,Lz])
        

        
        self.M=g_l*self.L+g_s*self.S
        
        self.field=syp.Matrix([0,0,0])
        self.energy=0
        self.field_L=syp.Matrix([0,0,0])
        self.field_S=syp.Matrix([0,0,0])
        
        self.pars={}
        self.variables={}
        if muB_L==0:
            self.pars["L_x"]=0
            self.pars["L_y"]=0
            self.pars["L_z"]=0
            self.unknowns={"S_x":1,"S_y":0,"S_z":0}
        else:
            self.unknowns={"S_x":1,"S_y":0,"S_z":0,"L_x":1,"L_y":0,"L_z":0}

        self.pars["g_l"]=gl
        self.pars["g_s"]=gs
        self.pars["gamma_L"]=gammaL
        self.pars["gamma_S"]=gammaS
        self.pars["alpha_L"]=alphaL
        self.pars["alpha_S"]=alphaS
        self.pars["muB_L"]=muB_L
        self.pars["muB_S"]=muB_S
        
    def externalField(self, B0=0, uB=[1,0,0]):
        
        Bext,ubx,uby,ubz = syp.symbols(r"B_ext, u_bx, u_by, u_bz")
        
        B_ext=Bext*syp.Matrix([ubx,uby,ubz])
        
        if "B_ext" not in self.variables:
            self.field+=B_ext
            self.energy+=-B_ext.dot(self.M)
            
        
        self.variables[r"B_ext"]=B0
        self.variables['u_bx']=uB[0]
        self.variables['u_by']=uB[1]
        self.variables['u_bz']=uB[2]
    
    def demagField(self,mu0_Ms,N_diag):
        Bdem, Nxx, Nyy, Nzz = syp.symbols(r'B_dem,N_xx,N_yy,N_zz')
        N=syp.diag(Nxx,Nyy,Nzz)
        
        B_dem=-Bdem*N@self.M
        
        if "B_dem" not in self.pars:
            self.field+=B_dem
            self.energy+=-0.5*B_dem.dot(self.M)

        self.pars["B_dem"]=mu0_Ms
        self.pars["N_xx"]=N_diag[0]
        self.pars["N_yy"]=N_diag[1]
        self.pars["N_zz"]=N_diag[2]
    
    def uniaxialField(self,Bk,uk):
        Bani,ukx,uky,ukz = syp.symbols(r"B_k, u_kx, u_ky, u_kz")
        
        B_ani=Bani*syp.Matrix([ukx,uky,ukz])
        
        if "B_k" not in self.pars:
            if "L_x" not in self.unknowns:
                self.field+=B_ani
            else:
                self.field_L+=B_ani
            self.energy+=-Bani/2*(self.M.dot(syp.Matrix([ukx,uky,ukz])))**2
            
        
        self.pars[r"B_k"]=Bk
        self.pars['u_kx']=uk[0]
        self.pars['u_ky']=uk[1]
        self.pars['u_kz']=uk[2]
        
    def chargeCurrent(self,BOe=0):
        B_oe, j_c=syp.symbols(f"B_Oe, j_c")
        
        B_Oe=B_oe*j_c*syp.Matrix([0,1,0])
        
        if "B_Oe" not in self.pars:
            self.field+=B_Oe
        
        self.pars["B_Oe"]=BOe
        self.variables["j_c"]=1
        
    def spinAccumulation(self,s):
        if "j_c" not in self.variables:
            print("Error: no charge current")
            return
        else:
            s_x,s_y,s_z,j_c=syp.symbols(r"s_x,s_y,s_z,j_c")
            self.s=j_c*syp.Matrix([s_x,s_y,s_z])
            
            self.pars["s_x"]=s[0]
            self.pars["s_y"]=s[1]
            self.pars["s_z"]=s[2]
    
    def orbitalAccumulation(self,l):
        if "j_c" not in self.variables:
            print("Error: no charge current")
            return
        else:
            l_x,l_y,l_z,j_c=syp.symbols(r"l_x,l_y,l_z,j_c")
            self.l=j_c*syp.Matrix([l_x,l_y,l_z])
            
            self.pars["l_x"]=l[0]
            self.pars["l_y"]=l[1]
            self.pars["l_z"]=l[2]
            
    def fieldLikeTorque(self,Bfl_spin,Bfl_orbital=0):
        if "j_c" not in self.variables:
            print("Error: no charge current")
            return
        B_fls,B_fll=syp.symbols("B_flS,B_flL")
        B=B_fll*self.l+B_fls*self.s
        
        if "B_flS" not in self.pars:
            self.field+=B
        
        self.pars["B_flS"]=Bfl_spin
        self.pars["B_flL"]=Bfl_orbital
    
    def dampingLikeTorque(self,Bdl_spin,Bdl_orbital=0):
        if "j_c" not in self.variables:
            print("Error: no charge current")
            return
        B_dls,B_dll=syp.symbols("B_dlS,B_dlL")
        B=B_dll*self.l.cross(self.M)+B_dls*self.s.cross(self.M)
        
        if "B_dlS" not in self.pars:
            self.field+=B
        
        self.pars["B_dlS"]=Bdl_spin
        self.pars["B_dlL"]=Bdl_orbital
        
    def spinOrbitCoupling(self, lambdaSoc=0.01):
        lambda_soc, muB_L, muB_S = syp.symbols("lambda_LS, muB_L, muB_S")


        B_SOC=lambda_soc*q_e/muB

        if "lambda_LS" not in self.variables:
            self.energy+=lambda_soc*self.L.dot(self.S)
            self.field_L+=B_SOC*muB_S*self.S
            self.field_S+=B_SOC*muB_L*self.L
        
        self.pars["lambda_LS"] = lambdaSoc
    
    def getTimeEvolutions(self):
        Beff_s=self.field+self.field_S
        Beff_l=self.field+self.field_L

        dL_dt = (self.pars["g_l"]*muB/hbar*self.L.cross(Beff_l)+self.pars["g_l"]*muB/hbar*self.pars["alpha_L"]*self.L.cross(self.L.cross(Beff_l)))/(1+self.pars["alpha_L"]**2)
        dS_dt = (self.pars["g_s"]*muB/hbar*self.S.cross(Beff_s)+self.pars["g_s"]*muB/hbar*self.pars["alpha_S"]*self.S.cross(self.S.cross(Beff_s)))/(1+self.pars["alpha_S"]**2)
        return dL_dt, dS_dt     
    
    def getEquilibriumFunction(self, evaluate=True, returnSingle=True):
        TS=self.S.cross(self.field)
        TL=self.L.cross(self.field)
        
        variables = self.variables | self.unknowns
        
        if evaluate:
            TL=TL.subs(self.pars)
            TS=TS.subs(self.pars)
        if not returnSingle:            
            self.TLsqr=TL.dot(TL)
            self.TSsqr=TS.dot(TS)
            simplifiedTLsqr=syp.nsimplify(self.TLsqr, tolerance=1e-10)
            simplifiedTSsqr=syp.nsimplify(self.TSsqr, tolerance=1e-10)

            return syp.lambdify(variables.keys(),simplifiedTLsqr),syp.lambdify(self.variables.keys(),simplifiedTSsqr)
        else:
            T=TS+TL
            self.Tsqr=T.dot(T)
            simplifiedTsqr=self.Tsqr#syp.nsimplify(self.Tsqr,tolerance=1e-10)
            return syp.lambdify(variables.keys(),simplifiedTsqr)

    def sweep_parameters(self,B,u_bx,u_by,u_bz,I, verbose=False, tol=1e-8):
        sweeps=[B,u_bx,u_by,u_bz,I]
        shape=None
        # make all sweeps the right size
        for sweep in sweeps:
            if not isinstance(sweep, (int,float)):
                if shape is not None and shape!=len(sweep):
                    print("Incompatible sweeps")
                    return
                shape=len(sweep)
        for n,sweep in enumerate(sweeps):
            if isinstance(sweep,(int,float)):
                sweeps[n]=[sweep for i in range(shape)]
        self.sweeps = sweeps
        Tsqred=self.getEquilibriumFunction()
        Tsqr=lambda dirs, B0, ubx, uby, ubz, j : Tsqred(B0, ubx, uby, ubz, j, *dirs)
        B, u_bx, u_by, u_bz, I= sweeps
        
        sols=[0]*len(B)
        
        if "L_x" in self.unknowns:
            for n,(B0, ubx, uby, ubz, I) in tqdm(enumerate(sweeps)):
        
        
        
                sol=optimize.minimize(Tsqr,[ubx,uby,ubz,ubx,uby,ubz], args=(B0, ubx, uby, ubz))
                sols[n]=sol.x
        else:
            for n,(B0, ubx, uby, ubz, I) in tqdm(enumerate(zip(*sweeps))):
                constraint  = lambda dirs: normConstraint(dirs,1)
                constraints = [
                    {'type': 'eq', 'fun': constraint}
                    ]
                sol=optimize.minimize(Tsqr,
                                      [ubx,uby,ubz],
                                      args=(B0, ubx, uby, ubz, I),
                                      constraints=constraints,
                                      method="SLSQP",
                                      tol=tol)
                sols[n]=sol.x
        return np.array(sols)
    
    def sweep_parameters_angle(self,B,thetaB, phiB ,I, verbose=False, tol=1e-8):
        uB=get_vector(thetaB,phiB)
        u_bx,u_by,u_bz=uB[:,0],uB[:,1],uB[:,2]
        sweeps=[B,u_bx,u_by,u_bz,I,thetaB,phiB]
        shape=None
        # make all sweeps the right size
        for sweep in sweeps:
            if not isinstance(sweep, (int,float)):
                if shape is not None and shape!=len(sweep):
                    print("Incompatible sweeps")
                    return
                shape=len(sweep)
        for n,sweep in enumerate(sweeps):
            if isinstance(sweep,(int,float)):
                sweeps[n]=[sweep for i in range(shape)]
        self.sweeps = sweeps
        Tsqred=self.getEquilibriumFunction()
        Tsqr=lambda angles, B0, ubx, uby, ubz, j : Tsqred(B0, ubx, uby, ubz, j, *get_vector(*angles))
        B, u_bx, u_by, u_bz, I, thetaB, phiB= sweeps
        
        sols=[0]*len(B)
        
        if "L_x" in self.unknowns:
            for n,(B0, ubx, uby, ubz, I) in tqdm(enumerate(sweeps)):
        
        
        
                sol=optimize.minimize(Tsqr,[ubx,uby,ubz,ubx,uby,ubz], args=(B0, ubx, uby, ubz))
                sols[n]=sol.x
        else:
            for n,(B0, ubx, uby, ubz, I, theta, phi) in tqdm(enumerate(zip(*sweeps))):
                sol=optimize.minimize(Tsqr,
                                      [theta,phi],
                                      args=(B0, ubx, uby, ubz, I),
                                      method="SLSQP",
                                      tol=tol)
                sols[n]=sol.x
        return np.array(sols)
        