## Library for macrospin simulations. Written by Niccolo Davitti, write if you need bugFixing


import numpy as np
from scipy import optimize
import sympy as syp
import matplotlib.pyplot as plt
from IPython import display
from tqdm import tqdm
np.seterr(divide='raise', invalid='raise')

def get_m(theta,phi):
    # converts (theta,phi)-->(mx,my,mz)
    theta,phi=np.deg2rad(theta),np.deg2rad(phi)
    mx=np.sin(theta)*np.cos(phi)
    my=np.sin(theta)*np.sin(phi)
    mz=np.cos(theta)
    return np.array([mx,my,mz])

def getSymVersor(theta,phi):
    # (theta,phi)-->(mx,my,mz) (symbolic)
    x=syp.sin(theta)*syp.cos(phi)
    y=syp.sin(theta)*syp.sin(phi)
    z=syp.cos(theta)
    return syp.Matrix([x,y,z])

def get_e_theta(theta,phi):
    # outputs the (symbolic) unit vector for theta
    x=syp.cos(theta)*syp.cos(phi)
    y=syp.cos(theta)*syp.sin(phi)
    z=-syp.sin(theta)
    return syp.Matrix([x,y,z])
def get_e_phi(theta,phi):
    # outputs the (symbolic) unit vector for phi
    x=-syp.sin(phi)
    y=syp.cos(phi)
    z=0
    return syp.Matrix([x,y,z])


def RH(theta,phi,B=0,thetaB=0,phiB=0,OHE=0,PHE=1,AHE=1):
    # Standard Hall resistance
    mx,my,mz=get_m(theta,phi)
    bx,by,bz=get_m(thetaB,phiB)
    return AHE*mz+PHE*mx*my+bz*OHE*B


theta, phi = syp.symbols(r'theta,phi')
m=getSymVersor(theta,phi)

e_theta=get_e_theta(theta,phi)
e_phi=get_e_phi(theta,phi)



def convertForMinimize(angles,function,thetaB,phiB,B0,alpha,jc):
        # takes the input function and coverts them to be a function of the _values_ of the relevant sin/cos instead of leaving the sin/cos to be evaluated.
        # This makes computation more efficient (evaluating the trig function only once instead of every time they appear)
        theta,phi=angles
        
        sinphi=np.sin(np.deg2rad(phi))
        cosphi=np.cos(np.deg2rad(phi))
        sintheta=np.sin(np.deg2rad(theta))
        costheta=np.cos(np.deg2rad(theta))
        sinphiB=np.sin(np.deg2rad(phiB))
        cosphiB=np.cos(np.deg2rad(phiB))
        sinthetaB=np.sin(np.deg2rad(thetaB))
        costhetaB=np.cos(np.deg2rad(thetaB))
        sinalpha=np.sin(np.deg2rad(alpha))
        cosalpha=np.cos(np.deg2rad(alpha))
        sinPhiMinAlpha=np.sin(np.deg2rad(phi-alpha))
        cosPhiMinAlpha=np.cos(np.deg2rad(phi-alpha))
        
        return function(B0,jc,sinphi,cosphi,sintheta,costheta,sinphiB,cosphiB,sinthetaB,costhetaB,sinalpha,cosalpha,sinPhiMinAlpha,cosPhiMinAlpha)

def check_variables(var1, var2, var3):
    # Check if they are all the same shape
    same_shape = (hasattr(var1, 'shape') and hasattr(var2, 'shape') and hasattr(var3, 'shape') and
                  var1.shape == var2.shape == var3.shape)

    # Count how many are of type int or float
    count_numeric = sum(isinstance(var, (int, float)) for var in [var1, var2, var3])

    # Check if two or more are numeric
    at_least_two_numeric = count_numeric >= 2

    return same_shape or at_least_two_numeric


mu0=4*np.pi*1e-7

class  macrospinSystem():
    def __init__(self,theta0=0,phi0=0,verbose=True) -> None:
        # defines magnetization angles for the system
        self.theta=theta0
        self.phi=phi0
        theta, phi = syp.symbols(r'theta,phi')
        self.thetavec,self.phivec=syp.symbols(r'theta,phi')
        self.m=getSymVersor(theta,phi)
        self.mvec=syp.symbols(r'\vec{m}')
        
        self.verbose=verbose

        self.field=syp.Matrix([0,0,0])
        self.effectiveField=0
        self.effectiveField_expanded=[]
        self.energy=0
        
    def externalField(self,B=0,thetaB=0,phiB=0):
        # introduces an external field to the system

        #symbols required for the definition of the field
        B0,theta_Bvec,phi_Bvec=syp.symbols(fr'B_ext,theta_B,phi_B')
        Bvec=syp.symbols(r'\vec{B}_{ext}')
        B_EXT=B0*getSymVersor(theta_Bvec, phi_Bvec)
        
        #symbols required for the dysplay of the effective field
        B_EXT_expanded=syp.MatMul(B0,getSymVersor(theta_Bvec,phi_Bvec),evaluate=False)
        self.Bext_vec=B_EXT
        self.energy+=-B_EXT.dot(self.m)

        ## if the system does not have the 'field' attributes, create it and set it equal to the external field
        #if not hasattr(self,'field'):
        #    self.field=B_EXT
        #    self.effectiveField=Bvec
        #    self.effectiveField_expanded=B_EXT_expanded
        #    print(f'added value for Bext to {B}T, phiB={phiB} degrees, thetaB= {thetaB} degrees')
        #    self.Bext=B
        #    self.thetaB=thetaB
        #    self.phiB=phiB
        #    self.B0vec=B0
        #    self.thetaBvec=theta_Bvec
        #    self.phiBvec=phi_Bvec
        #    return
        
        # if the system does have the 'field' attributes, but not the external field, add it
        if not hasattr(self,'Bext'):
            self.field+=B_EXT
            self.effectiveField+=Bvec
            self.effectiveField_expanded.append(B_EXT_expanded)
            #if self.effectiveField_expanded is None:
            #    self.effectiveField_expanded=B_EXT_expanded
            #else:
            #    self.effectiveField_expanded=syp.MatAdd(self.effectiveField_expanded,B_EXT_expanded,evaluate=False)
            if self.verbose:
                print(f'added value for Bext to {B}T, phiB={phiB} degrees, thetaB= {thetaB} degrees')
        else:
            if self.verbose:
                print(f'updated value for Bext to {B}T, phiB={phiB} degrees, thetaB= {thetaB} degrees')

        #update the values
        self.Bext=B
        self.thetaB=thetaB
        self.phiB=phiB
        self.B0vec=B0
        self.thetaBvec=theta_Bvec
        self.phiBvec=phi_Bvec
        
    def uniaxialField(self,Ban=0,thetaAn=0,phiAn=0):
        B_an,theta_an,phi_an=syp.symbols(fr'B_an,theta_An,phi_An')
        Bvec=syp.symbols(r'B_k(\vec{u}_k\cdot\vec{m})\vec{u}_k')
        
        B=B_an*getSymVersor(theta_an, phi_an).dot(self.m)*getSymVersor(theta_an, phi_an)
        B_expanded=syp.MatMul(B_an*getSymVersor(theta_an, phi_an).dot(self.m),getSymVersor(theta_an,phi_an),evaluate=False)
        self.Ban_vec=B
        self.energy+=-B_an/2*(self.m.dot(getSymVersor(theta_an,phi_an)))**2
        #if not hasattr(self,'field'):
        #    self.field=B
        #    print(f'Added value for Ban to {Ban}T, phiB={phiAn} degrees, thetaB= {thetaAn} degrees')
        #    self.effectiveField=Bvec
        #    self.effectiveField_expanded=B_expanded
        #    self.Ban=Ban
        #    self.thetaAn=thetaAn
        #    self.phiAn=phiAn
        #    return
        if not hasattr(self,'thetaAn'):
            self.field+=B
            if self.verbose:
                print(f'Added value for Ban to {Ban}T, phiB={phiAn} degrees, thetaB= {thetaAn} degrees')
            
            self.effectiveField+=Bvec
            self.effectiveField_expanded.append(B_expanded)
            #if self.effectiveField_expanded is None:
            #    self.effectiveField_expanded=B_expanded
            #else:
            #    self.effectiveField_expanded=syp.MatAdd(self.effectiveField_expanded,B_expanded,evaluate=False)
        else:
            if self.verbose:
                print(f'updated value for Ban to {Ban}T, phiB={phiAn} degrees, thetaB= {thetaAn} degrees')
        self.Ban=Ban
        self.thetaAn=thetaAn
        self.phiAn=phiAn
    
    def demagField(self,Bdem=0,demagTensor=[0,0,1]):
        B_dem, Nxx, Nyy, Nzz = syp. symbols(r'B_dem,Nxx,Nyy,Nzz')
        N=syp.diag(Nxx,Nyy,Nzz)
        B=-B_dem*N@self.m
        Nvec=syp.symbols(r'\bold{N}')
        Bvec=-B_dem*Nvec*self.mvec*int(bool(Bdem))
        self.Bdem_vec=B
        B_expanded=syp.MatMul(-B_dem,N,self.m,evaluate=False)
        self.energy+=B_dem*self.m.dot(N@self.m)
        #if not hasattr(self,'field'):
        #    self.field=B
        #    print(f'Added value for Bdem to {Bdem}T, demag tensor diagonal to {demagTensor}')
        #    self.effectiveField=Bvec
        #    self.effectiveField_expanded=B_expanded
        #    self.Bdem=Bdem
        #    self.Nxx,self.Nyy,self.Nzz=demagTensor
        #    self.N=np.diag([self.Nxx,self.Nyy,self.Nzz])
        #    return
        if not hasattr(self,'Bdem'):
            self.field+=B
            if self.verbose:
                print(f'Added value for Bdem to {Bdem}T, demag tensor diagonal to {demagTensor}')
            self.effectiveField+=Bvec
            self.effectiveField_expanded.append(B_expanded)
            #if self.effectiveField_expanded is None:
            #    self.effectiveField_expanded=B_expanded
            #else:
            #    self.effectiveField_expanded=syp.MatAdd(self.effectiveField_expanded,B_expanded,evaluate=False)

        else:
            if self.verbose:
                print(f'updatedvalue for Bdem to {Bdem}T, demag tensor diagonal to {demagTensor}')
        self.Bdem=Bdem
        self.Nxx,self.Nyy,self.Nzz=demagTensor
        self.N=np.diag([self.Nxx,self.Nyy,self.Nzz])
    
    def fourFoldField(self,B4=0,alpha_val=0):
        B4_amplitude,alpha=syp.symbols(r'B_4,alpha')
        theta,phi=self.thetavec,self.phivec
        B=B4_amplitude*syp.Matrix([syp.cos(phi-alpha)**3*syp.cos(alpha)-syp.sin(alpha)*syp.sin(phi-alpha)**3
                            ,syp.cos(phi-alpha)**3*syp.sin(alpha)+syp.cos(alpha)*syp.sin(phi-alpha)**3,
                            0])*syp.sin(theta)**3
        Bvec=syp.symbols(r'\vec{B}_{4}')
        self.B4_vec=B
        B_expanded=syp.MatMul(B4_amplitude,syp.sin(theta)**3,syp.Matrix([syp.cos(phi-alpha)**3*syp.cos(alpha)-syp.sin(alpha)*syp.sin(phi-alpha)**3
                            ,syp.cos(phi-alpha)**3*syp.sin(alpha)+syp.cos(alpha)*syp.sin(phi-alpha)**3,
                            0]),evaluate=False)
        #if not hasattr(self,'field'):
        #    self.field=B
        #    print(f'Added value for B4 to {B4}T, alpha to {alpha}')
        #    self.effectiveField=Bvec
        #    self.B4=B4
        #    self.alpha=alpha_val
        #    self.alphaVec=alpha
        #    return
        self.energy+=-B4/4*(    (self.m.dot(getSymVersor(0,alpha)))**4 + (self.m.dot(getSymVersor(0,alpha+90)))**4       )
        if not hasattr(self,'B4'):
            self.field+=B
            if self.verbose:
                print(f'Added value for B4 to {B4}T, alpha to {alpha}')
            self.effectiveField+=Bvec
            self.effectiveField_expanded.append(B_expanded)
            #if self.effectiveField_expanded is None:
            #    self.effectiveField_expanded=B_expanded
            #else:
            #    self.effectiveField_expanded=syp.MatAdd(self.effectiveField_expanded,B_expanded,evaluate=False)
        else:
            if self.verbose:
                print(f'updated value for B4 to {B4}T, alpha to {alpha}')
        self.B4=B4
        self.alpha=alpha_val
        self.alphaVec=alpha
    
    def biasField(self,Bbias=0,thetabias=0,phibias=0):
        B_bias,theta_bias,phi_bias=syp.symbols(fr'B_bias,theta_bias,phi_bias')
        Bvec=syp.symbols(r'B_bias\vec{u}_bias}')
        
        B=B_bias*getSymVersor(theta_bias, phi_bias)
        self.Bbias_vec=B
        B_expanded=syp.MatMul(B_bias,getSymVersor(theta_bias,phi_bias),evaluate=False)

        self.energy+=-B.dot(self.m)
        #if not hasattr(self,'field'):
        #    self.field=B
        #    print(f'Added value for Bbias to {Bbias}T, phiBias={phibias} degrees, thetaBias= {thetabias} degrees')
        #    self.effectiveField=Bvec
        #    self.Bbias=Bbias
        #    self.thetabias=thetabias
        #    self.phibias=phibias
        #    return
        if not hasattr(self,'thetabias'):
            self.field+=B
            if self.verbose:
                print(f'Added value for Bbias to {Bbias}T, phiBias={phibias} degrees, thetaBias= {thetabias} degrees')
            self.effectiveField+=Bvec
            self.effectiveField_expanded.append(B_expanded)
            #if self.effectiveField_expanded is None:
            #    self.effectiveField_expanded=B_expanded
            #else:
            #    self.effectiveField_expanded=syp.MatAdd(self.effectiveField_expanded,B_expanded,evaluate=False)
        else:
            if self.verbose:
                print(f'updated value for Bbias to {Bbias}T, phiBias={phibias} degrees, thetaBias= {thetabias} degrees')
        self.Bbias=Bbias
        self.thetabias=thetabias
        self.phibias=phibias

    def chargeCurrent(self,B_Oe=0):
        self.jc=1
        BOe,jsim=syp.symbols(fr'B_Oe,j_c')
        Bvec=syp.symbols(r'\vec{B}_{Oe}')
        BOE=BOe*jsim*getSymVersor(syp.pi/2,syp.pi/2)
        self.BOe_vec=BOe
        B_expanded=syp.MatMul(BOe*jsim,getSymVersor(syp.pi/2,syp.pi/2),evaluate=False)
        #if not hasattr(self,'field'):
        #    self.field=BOE
        #    self.effectiveField=Bvec
        #    print(f'added value for BOe to {B_Oe*1e3}mT')
        #    self.BOe=B_Oe
        #    self.BOe_vec=BOe
        #    return
        if not hasattr(self,'BOe'):
            self.field+=BOE
            self.effectiveField+=Bvec
            if self.verbose:
                print(f'added value for BOe to {B_Oe*1e3}mT')
            self.effectiveField_expanded.append(B_expanded)
            #if self.effectiveField_expanded is None:
            #    self.effectiveField_expanded=B_expanded
            #else:
            #    self.effectiveField_expanded=syp.MatAdd(self.effectiveField_expanded,B_expanded,evaluate=False)
        else:
            if self.verbose:
                print(f'updated value for BOe to {B_Oe*1e3}mT')
        self.BOe=B_Oe


        
    
    def spinAccumulation(self,thetaS=90,phiS=90):
        theta_S,phi_S,j_sim=syp.symbols(r'theta_S,phi_S,j_c')
        self.thetaS=thetaS
        self.phiS=phiS
        self.s=j_sim*getSymVersor(theta_S,phi_S)
        self.svec=syp.symbols(r'\vec{s}')
    
    def fieldLikeTorque(self,Bfl=0):
        B_fl = syp.symbols(r'B_fl')
        Bvec=syp.symbols(r'B_{fl}\vec{s}')
        B=B_fl*self.s
        B_expanded=syp.MatMul(B_fl,self.s,evaluate=False)
        #if not hasattr(self,'field'):
        #    self.field=B
        #    print(f'Added value for Bfl to {Bfl*1e3}mT')
        #    self.effectiveField=Bvec
        #    self.Bfl=Bfl
        #    return
        if not hasattr(self,'Bfl'):
            self.field+=B
            if self.verbose:
                print(f'Added value for Bfl to {Bfl*1e3}mT')
            self.effectiveField+=Bvec
            self.effectiveField_expanded.append(B_expanded)
            #if self.effectiveField_expanded is None:
            #    self.effectiveField_expanded=B_expanded
            #else:
            #    self.effectiveField_expanded=syp.MatAdd(self.effectiveField_expanded,B_expanded,evaluate=False)
        else:
            if self.verbose:
                print(f'updated value for Bfl to {Bfl*1e3}mT')
        self.Bfl=Bfl

    def dampingLikeTorque(self,Bdl=0):
        B_dl = syp.symbols(r'B_dl')
        Bvec=syp.symbols(r'B_{dl}\vec{s}\times\vec{m}')
        B=B_dl*self.s.cross(self.m)
        B_expanded=syp.MatMul(B_dl,self.s.cross(self.m),evaluate=False)
        #if not hasattr(self,'field'):
        #    self.field=B
        #    print(f'Added value for Bdl to {Bdl*1e3}mT')
        #    self.effectiveField=Bvec
        #    self.Bdl=Bdl
        #    return
        if not hasattr(self,'Bdl'):
            self.field+=B
            if self.verbose:
                print(f'Added value for Bdl to {Bdl*1e3}mT')
            self.effectiveField+=Bvec
            self.effectiveField_expanded.append(B_expanded)
            #if self.effectiveField_expanded is None:
            #    self.effectiveField_expanded=B_expanded
            #else:
            #    self.effectiveField_expanded=syp.MatAdd(self.effectiveField_expanded,B_expanded,evaluate=False)
        else:
            if self.verbose:
                print(f'updated value for Bdl to {Bdl*1e3}mT')
        self.Bdl=Bdl
    
    def getAttributes(self,lockBext=False):
        attributes={}
        match lockBext:
            case False:
                variables=['phi','phiB','thetaB','theta','Bext','alpha','jc']
            case True:
                variables=[]
            case 'forDisplay':
                variables=['phi','phiB','thetaB','theta','Bext','alpha']
            case 'LLG':
                variables=['phi','theta','jc']
            case 'wolfhart':
                variables=['phi','theta']
        self.variables=variables
        for attr in dir(self):
            value=getattr(self,attr)
            if isinstance(value,(float,int)) and attr not in variables and attr:
                if attr != 'phi_B' and attr != 'theta_B' and attr != 'thetaB' and attr != 'phiB':
                    attr=attr.replace('B','B_')
                attr=attr.replace('phi','phi_')
                attr=attr.replace('theta','theta_')
                attr=attr.replace('j','j')
                if 'theta' in attr or 'phi' in attr or 'alpha' in attr:
                    attributes[attr]=np.deg2rad(value)
                else:
                    attributes[attr]=value
        if lockBext=='forDisplay':
            return attributes    
        self.staticAttributes=attributes
        
        
    def getSubsDict(self):
        returnDict = {}

        # Use hasattr to check for the presence of attributes
        if hasattr(self, 'phivec'):
            #print('Substituted cos(phi) and sin(phi)')
            returnDict[syp.sin(self.phivec)] = 'sinphi'
            returnDict[syp.cos(self.phivec)] = 'cosphi'
        
        if hasattr(self, 'thetavec'):
            #print('Substituted cos(theta) and sin(theta)')
            returnDict[syp.sin(self.thetavec)] = 'sintheta'
            returnDict[syp.cos(self.thetavec)] = 'costheta'
        
        if hasattr(self, 'phiBvec'):
            #print('Substituted cos(phiB) and sin(phiB)')
            returnDict[syp.sin(self.phiBvec)] = 'sinphiB'
            returnDict[syp.cos(self.phiBvec)] = 'cosphiB'
        
        if hasattr(self, 'thetaBvec'):
            #print('Substituted cos(thetaB) and sin(thetaB)')
            returnDict[syp.sin(self.thetaBvec)] = 'sinthetaB'
            returnDict[syp.cos(self.thetaBvec)] = 'costhetaB'
        
        if hasattr(self, 'alphaVec'):
            #print('Substituted cos(alpha) and sin(alpha)')
            returnDict[syp.sin(self.alphaVec)] = 'sinalpha'
            returnDict[syp.cos(self.alphaVec)] = 'cosalpha'
            if hasattr(self, 'phivec'):
                #print('Substituted cos(phi-alpha) and sin(phi-alpha)')
                returnDict[syp.sin(self.phivec - self.alphaVec)] = 'sinPhiMinAlpha'
                returnDict[syp.cos(self.phivec - self.alphaVec)] = 'cosPhiMinAlpha'
                returnDict[syp.sin(-self.phivec + self.alphaVec)] = '-sinPhiMinAlpha'
                returnDict[syp.cos(-self.phivec + self.alphaVec)] = 'cosPhiMinAlpha'

        return returnDict
    
    def totalEffectiveField(self,evaluate=False):
        return syp.MatAdd(*self.effectiveField_expanded,evaluate=evaluate)

    def get_effectiveFields_forLLG(self):
        B_theta=self.field.dot(e_theta)
        B_phi=self.field.dot(e_phi)
        
        self.getAttributes()
        subsDict=self.getSubsDict()
        
        variables=['B_ext','j_c','sinphi','cosphi','sintheta','costheta','sinphiB','cosphiB','sinthetaB','costhetaB','sinalpha','cosalpha','sinPhiMinAlpha','cosPhiMinAlpha']
        
        B_theta=B_theta.subs(self.staticAttributes)
        B_phi=B_phi.subs(self.staticAttributes)
        correctedBtheta=B_theta.subs(subsDict)
        correctedBphi=B_phi.subs(subsDict)
        
        Btheta=syp.lambdify(variables,correctedBtheta)
        Bphi=syp.lambdify(variables,correctedBphi)
    
        return Btheta,Bphi

    def simulateLLG(self,B_sweep=0,thetaB_sweep=0,phiB_sweep=0,j_modulation=1,timeSweep=0,initialPosition=[0,0],showSweep=False,alpha_G=1e-2,gamma=1.76e11):
        prevShape=None
        j_sweep=j_modulation
        sweeps=[B_sweep,thetaB_sweep,phiB_sweep,j_sweep,timeSweep]
        for sweep in [B_sweep,thetaB_sweep,phiB_sweep,j_sweep,timeSweep]:
            if not isinstance(sweep,(int,float)):
                currentShape=len(sweep)
                if prevShape is not None:
                    if prevShape!=currentShape:
                        print('Two or more of the provided sweeps do not have a compatible shape')
                        return
                prevShape=currentShape

        for n,sweep in enumerate([B_sweep,thetaB_sweep,phiB_sweep,j_sweep,timeSweep]):
            if isinstance(sweep,(int,float)):
                sweeps[n]=[sweep for i in range(currentShape)]
        
        B_sweep,thetaB_sweep,phiB_sweep,j_sweep,time=sweeps

        j_sweep=self.jc*j_sweep

        BthetaRaw,BphiRaw=self.get_effectiveFields_forLLG()
        Btheta=lambda angles,thetaB,phiB,B0,alpha,jc: convertForMinimize(angles,BthetaRaw,thetaB,phiB,B0,alpha,jc)    
        Bphi=lambda angles,thetaB,phiB,B0,alpha,jc: convertForMinimize(angles,BphiRaw,thetaB,phiB,B0,alpha,jc)    
        
        mult=gamma/(1+alpha_G**2)

        def dAngles(angles,thetaB,phiB,B0,alpha,jc):
            dTheta=mult*np.rad2deg(alpha_G*Btheta(angles,thetaB,phiB,B0,alpha,jc)+Bphi(angles,thetaB,phiB,B0,alpha,jc))
            try:
                dPhi=np.rad2deg(mult/np.sin(np.deg2rad(angles[0]))*(-Btheta(angles,thetaB,phiB,B0,alpha,jc)+alpha_G*Bphi(angles,thetaB,phiB,B0,alpha,jc)))
            except:
                print("Set dphi to 0")
                dPhi=0
            return np.array([dTheta,dPhi])
        for n,sweep in enumerate([B_sweep,thetaB_sweep,phiB_sweep,j_sweep,timeSweep]):
            if isinstance(sweep,(int,float)):
                sweeps[n]=[sweep for i in range(currentShape)]
            
        angles=[np.array(initialPosition)]*len(B_sweep)
        if hasattr(self,'alpha'):
            alphaVal=self.alpha
        else:
            alphaVal=0
        if self.verbose:
            for n,(B,thetaB,phiB,j_c) in tqdm(enumerate(zip(B_sweep,thetaB_sweep,phiB_sweep,j_sweep)),total=len(B_sweep)):
                if n==len(B_sweep)-1:
                    continue
                
                dt=timeSweep[n+1]-timeSweep[n]
                k1=dt*dAngles(angles[n],thetaB,phiB,B,alphaVal,j_c)
                k2=dt*dAngles(angles[n]+dt*k1/2,(thetaB+thetaB_sweep[n+1])/2,(phiB+phiB_sweep[n+1])/2,(B+B_sweep[n+1])/2,alphaVal,(j_c+j_sweep[n+1])/2)
                k3=dt*dAngles(angles[n]+dt*k2/2,(thetaB+thetaB_sweep[n+1])/2,(phiB+phiB_sweep[n+1])/2,(B+B_sweep[n+1])/2,alphaVal,(j_c+j_sweep[n+1])/2)
                k4=dt*dAngles(angles[n]+dt*k3,thetaB_sweep[n+1],phiB_sweep[n+1],B_sweep[n+1],alphaVal,j_sweep[n+1])

                angles[n+1]=angles[n]+(k1+2*k2+2*k3+k4)/6
        else:
            for n,(B,thetaB,phiB,j_c) in enumerate(zip(B_sweep,thetaB_sweep,phiB_sweep,j_sweep)):
                if n==len(B_sweep)-1:
                    continue
                
                dt=timeSweep[n+1]-timeSweep[n]
                k1=dt*dAngles(angles[n],thetaB,phiB,B,alphaVal,j_c)
                k2=dt*dAngles(angles[n]+dt*k1/2,(thetaB+thetaB_sweep[n+1])/2,(phiB+phiB_sweep[n+1])/2,(B+B_sweep[n+1])/2,alphaVal,(j_c+j_sweep[n+1])/2)
                k3=dt*dAngles(angles[n]+dt*k2/2,(thetaB+thetaB_sweep[n+1])/2,(phiB+phiB_sweep[n+1])/2,(B+B_sweep[n+1])/2,alphaVal,(j_c+j_sweep[n+1])/2)
                k4=dt*dAngles(angles[n]+dt*k3,thetaB_sweep[n+1],phiB_sweep[n+1],B_sweep[n+1],alphaVal,j_sweep[n+1])
                
                angles[n+1]=angles[n]+(k1+2*k2+2*k3+k4)/6
        
        angles= np.array(angles)
        theta,phi=angles[:,0],angles[:,1]
        if showSweep:

            sweep=timeSweep
            m=get_m(theta,phi)
            col = np.arange(len(m[0,:]))
            mx=m[0,:]
            my=m[1,:]
            mz=m[2,:]
            
            plt.plot(sweep,mx,label='mx')
            plt.plot(sweep,my,label='my')
            plt.plot(sweep,mz,label='mz')
            #plt.plot(time,I,color='Red',label='I')
            plt.legend()
            # 3D Plot
            fig = plt.figure()
            ax3D = fig.add_subplot(projection='3d')
            cm = plt.get_cmap('Greys')
            r = 1
            u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
            xs = np.cos(u) * np.sin(v)
            ys = np.sin(u) * np.sin(v)
            zs = np.cos(v)
            ax3D.plot_surface(xs, ys, zs, alpha=0.3, color="white")
            ax3D.set_box_aspect((np.ptp(xs), np.ptp(ys), np.ptp(zs)))
            p3d_scatter = ax3D.scatter(mx, my, mz, s=30, c=col, marker='o') 
            p3d_plot = ax3D.plot(mx, my, mz)                                                                                
            plt.show()
            
        return angles[:,0],angles[:,1]

    def stonerWolfhart(self,B_sweep=0,thetaB_sweep=0,phiB_sweep=0,showSweep=True):
        prevShape=None
        sweeps=[B_sweep,thetaB_sweep,phiB_sweep]
        for sweep in [B_sweep,thetaB_sweep,phiB_sweep]:
            if not isinstance(sweep,(int,float)):
                currentShape=len(sweep)
                if prevShape is not None:
                    if prevShape!=currentShape:
                        print('Two or more of the provided sweeps do not have a compatible shape')
                        return
                prevShape=currentShape

        for n,sweep in enumerate([B_sweep,thetaB_sweep,phiB_sweep]):
            if isinstance(sweep,(int,float)):
                sweeps[n]=[sweep for i in range(currentShape)]

        B_sweep,thetaB_sweep,phiB_sweep=sweeps

        finalValues=[[0,0]]*len(B_sweep)

        theta, phi = syp.symbols(r'theta,phi')


        
        for n,(B,thetaB,phiB) in tqdm(enumerate(zip(B_sweep,thetaB_sweep,phiB_sweep)),total=len(B_sweep)):
            self.Bext=B
            self.phiB=phiB
            self.thetaB=thetaB
            self.getAttributes(lockBext='wolfhart')
            totalEnergy=self.energy.subs(self.staticAttributes)
            diffEnergy_theta=syp.diff(totalEnergy,theta)
            diffEnergy_phi=syp.diff(totalEnergy,phi)

            solutions=syp.solve((diffEnergy_theta,diffEnergy_phi),(theta,phi))
            print(solutions)
            finalValues[n]=[solutions[theta].evalf(),solutions[phi].evalf()]
        
        angles = np.array(finalValues)

        theta,phi=angles[:,0],angles[:,1]
        if showSweep:
            match showSweep:
                case 'B':
                    sweep=B_sweep
                case 'theta':
                    sweep=thetaB_sweep
                case 'phi':
                    sweep=phiB_sweep
            m=get_m(theta,phi)
            col = np.arange(len(m[0,:]))
            mx=m[0,:]
            my=m[1,:]
            mz=m[2,:]
            plt.figure()
            plt.plot(sweep,mx,label='mx')
            plt.plot(sweep,my,label='my')
            plt.plot(sweep,mz,label='mz')
            #plt.plot(time,I,color='Red',label='I')
            plt.legend()
            # 3D Plot
            fig = plt.figure()
            ax3D = fig.add_subplot(projection='3d')
            cm = plt.get_cmap('Greys')
            r = 1
            u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
            xs = np.cos(u) * np.sin(v)
            ys = np.sin(u) * np.sin(v)
            zs = np.cos(v)
            ax3D.plot_surface(xs, ys, zs, alpha=0.3, color="white")
            ax3D.set_box_aspect((np.ptp(xs), np.ptp(ys), np.ptp(zs)))
            p3d_scatter = ax3D.scatter(mx, my, mz, s=30, c=col, marker='o') 
            p3d_plot = ax3D.plot(mx, my, mz)                                                                                
            plt.show()
            plt.close()
        return angles[:,0],angles[:,1]




class quasiEquilibriumSystem(macrospinSystem):
    
    def getTargetFunction_symbolic(self):
        T=self.m.cross(self.field)
        self.Tsqr=T.dot(T)
        self.jac=syp.derive_by_array(self.Tsqr,[self.thetavec,self.phivec])
        self.hess=syp.derive_by_array(self.jac,[self.thetavec,self.phivec])

    def getTargetFunction(self):
        self.getTargetFunction_symbolic()
        self.getAttributes()
        subsDict=self.getSubsDict()
        variables=['B_ext','j_c','sinphi','cosphi','sintheta','costheta','sinphiB','cosphiB','sinthetaB','costhetaB','sinalpha','cosalpha','sinPhiMinAlpha','cosPhiMinAlpha']
        targetFunction=self.Tsqr.subs(self.staticAttributes).evalf()
        simplifiedTargetFunction=syp.nsimplify(targetFunction,tolerance=1e-10)
        correctedTargetFcn=targetFunction.subs(subsDict)
        return syp.lambdify(variables,correctedTargetFcn),simplifiedTargetFunction,syp.nsimplify(correctedTargetFcn,tolerance=1e-10)
    
    def getJacobian(self):
        self.getTargetFunction_symbolic()
        self.getAttributes()
        subsDict=self.getSubsDict()
        variables=['B_ext','j_c','sinphi','cosphi','sintheta','costheta','sinphiB','cosphiB','sinthetaB','costhetaB','sinalpha','cosalpha','sinPhiMinAlpha','cosPhiMinAlpha']
        targetFunction=self.jac.subs(self.staticAttributes)
        simplifiedTargetFunction=syp.nsimplify(targetFunction,tolerance=1e-10)
        correctedTargetFcn=targetFunction.subs(subsDict)
        return syp.lambdify(variables,correctedTargetFcn),simplifiedTargetFunction,syp.nsimplify(correctedTargetFcn,tolerance=1e-10)
        
    def getHessian(self):
        self.getTargetFunction_symbolic()
        self.getAttributes()
        subsDict=self.getSubsDict()
        variables=['B_ext','j_c','sinphi','cosphi','sintheta','costheta','sinphiB','cosphiB','sinthetaB','costhetaB','sinalpha','cosalpha','sinPhiMinAlpha','cosPhiMinAlpha']
        targetFunction=self.hess.subs(self.staticAttributes)
        simplifiedTargetFunction=syp.nsimplify(targetFunction,tolerance=1e-10)
        correctedTargetFcn=targetFunction.subs(subsDict)
        return syp.lambdify(variables,correctedTargetFcn),simplifiedTargetFunction,syp.nsimplify(correctedTargetFcn,tolerance=1e-10)
    
    def sweep_equilibrium(self,B_sweep=0,thetaB_sweep=0,phiB_sweep=0,j_modulation=1,timeSweep=0,initialGuess=[0,0],showSweep=False,switchGuess='Bext',tol=1e-10,B_coer=1e10,title=''):
        prevShape=None
        j_sweep=j_modulation
        sweeps=[B_sweep,thetaB_sweep,phiB_sweep,j_sweep,timeSweep]
        for sweep in [B_sweep,thetaB_sweep,phiB_sweep,j_sweep,timeSweep]:
            if not isinstance(sweep,(int,float)):
                currentShape=len(sweep)
                if prevShape is not None:
                    if prevShape!=currentShape:
                        print('Two or more of the provided sweeps do not have a compatible shape')
                        return
                prevShape=currentShape
        #print(f'Sweeping over {currentShape} points')
        for n,sweep in enumerate([B_sweep,thetaB_sweep,phiB_sweep,j_sweep,timeSweep]):
            if isinstance(sweep,(int,float)):
                sweeps[n]=[sweep for i in range(currentShape)]

        
        target,_,_=self.getTargetFunction()
        jac,_,_=self.getJacobian()
        hess,_,_=self.getHessian()
        
        B_sweep,thetaB_sweep,phiB_sweep,j_sweep,time=sweeps

        j_sweep=self.jc*j_sweep
        
        targetMin=lambda angles,thetaB,phiB,B0,alpha,jc: convertForMinimize(angles,target,thetaB,phiB,B0,alpha,jc)    
        jac=lambda angles,thetaB,phiB,B0,alpha,jc: convertForMinimize(angles,jac,thetaB,phiB,B0,alpha,jc)  
        hess=lambda angles,thetaB,phiB,B0,alpha,jc: convertForMinimize(angles,hess,thetaB,phiB,B0,alpha,jc)  

        angles=[initialGuess]*len(B_sweep)
        if hasattr(self,'alpha'):
            alphaVal=self.alpha
        else:
            alphaVal=0
        if self.verbose:
            for n,(B,thetaB,phiB,j_c) in tqdm(enumerate(zip(B_sweep,thetaB_sweep,phiB_sweep,j_sweep)),total=len(B_sweep)):
                if B_sweep[n-1]<B_coer<B_sweep[n] or B_sweep[n-1]>-B_coer>B_sweep[n]:

                    match switchGuess:
                        case 'Bext':
                            if B>=0:
                                newGuess=[thetaB,phiB]
                            else:
                                newGuess=[180-thetaB,180+phiB]
                        case 'theta':
                            newGuess=[180-angles[n-1][0],angles[n-1][1]]
                        case 'phi':
                            newGuess=[angles[n-1][0],180+angles[n-1][1]]
                        case 'both':
                            newGuess=[180-angles[n-1][0],180+angles[n-1][1]]
                        case 'y':
                            newGuess=[angles[n-1][0],360-angles[n-1][1]]
                        case 'x':
                            newGuess=[angles[n-1][0],180-angles[n-1][1]]
                    print(f'Coercive field reached. Attemping switching from {angles[n-1]} to {newGuess}')

                    sol=optimize.minimize(targetMin,newGuess,args=(thetaB,phiB,B,alphaVal,j_c),tol=tol)
                else:
                    sol=optimize.minimize(targetMin,angles[n-1],args=(thetaB,phiB,B,alphaVal,j_c),tol=tol)
                if sol.success:
                    angles[n]=sol.x
                else:
                    print(f'No equilibrium found for B={B}, phiB={phiB}, thetaB={thetaB} attempting switching')
                    match switchGuess:
                        case 'Bext':
                            if B>=0:
                                newGuess=[thetaB,phiB]
                            else:
                                newGuess=[180-thetaB,180+phiB]
                        case 'theta':
                            newGuess=[180-angles[n-1][0],angles[n-1][1]]
                        case 'phi':
                            newGuess=[angles[n-1][0],180+angles[n-1][1]]
                        case 'both':
                            newGuess=[180-angles[n-1][0],180+angles[n-1][1]]
                        case 'y':
                            newGuess=[angles[n-1][0],360-angles[n-1][1]]
                        case 'x':
                            newGuess=[angles[n-1][0],180-angles[n-1][1]]

                    sol=optimize.minimize(targetMin,newGuess,args=(thetaB,phiB,B,alphaVal,j_c),tol=tol)
                    if sol.success:
                        angles[n]=sol.x
                    else:
                        print(f'Switching attempt failed for B={B},phiB={phiB}, thetaB={thetaB}, try another switch guess')
                        return
        else:
            for n,(B,thetaB,phiB,j_c) in enumerate(zip(B_sweep,thetaB_sweep,phiB_sweep,j_sweep)):
                if B_sweep[n-1]<B_coer<B_sweep[n] or B_sweep[n-1]>-B_coer>B_sweep[n]:

                    match switchGuess:
                        case 'Bext':
                            if B>=0:
                                newGuess=[thetaB,phiB]
                            else:
                                newGuess=[180-thetaB,180+phiB]
                        case 'theta':
                            newGuess=[180-angles[n-1][0],angles[n-1][1]]
                        case 'phi':
                            newGuess=[angles[n-1][0],180+angles[n-1][1]]
                        case 'both':
                            newGuess=[180-angles[n-1][0],180+angles[n-1][1]]
                        case 'y':
                            newGuess=[angles[n-1][0],360-angles[n-1][1]]
                        case 'x':
                            newGuess=[angles[n-1][0],180-angles[n-1][1]]
                    print(f'Coercive field reached. Attemping switching from {angles[n-1]} to {newGuess}')

                    sol=optimize.minimize(targetMin,newGuess,args=(thetaB,phiB,B,alphaVal,j_c),tol=tol)
                else:
                    sol=optimize.minimize(targetMin,angles[n-1],args=(thetaB,phiB,B,alphaVal,j_c),tol=tol)
                if sol.success:
                    angles[n]=sol.x
                else:
                    print(f'No equilibrium found for B={B}, phiB={phiB}, thetaB={thetaB} attempting switching')
                    match switchGuess:
                        case 'Bext':
                            if B>=0:
                                newGuess=[thetaB,phiB]
                            else:
                                newGuess=[180-thetaB,180+phiB]
                        case 'theta':
                            newGuess=[180-angles[n-1][0],angles[n-1][1]]
                        case 'phi':
                            newGuess=[angles[n-1][0],180+angles[n-1][1]]
                        case 'both':
                            newGuess=[180-angles[n-1][0],180+angles[n-1][1]]
                        case 'y':
                            newGuess=[angles[n-1][0],360-angles[n-1][1]]
                        case 'x':
                            newGuess=[angles[n-1][0],180-angles[n-1][1]]

                    sol=optimize.minimize(targetMin,newGuess,args=(thetaB,phiB,B,alphaVal,j_c),tol=tol)
                    if sol.success:
                        angles[n]=sol.x
                    else:
                        print(f'Switching attempt failed for B={B},phiB={phiB}, thetaB={thetaB}, try another switch guess')
                        return
            
        
        angles= np.array(angles)
        
        theta,phi=angles[:,0],angles[:,1]
        if showSweep:
            match showSweep:
                case 'B':
                    sweep=B_sweep
                case 'theta':
                    sweep=thetaB_sweep
                case 'phi':
                    sweep=phiB_sweep
                case 'time':
                    sweep=timeSweep
            m=get_m(theta,phi)
            col = np.arange(len(m[0,:]))
            mx=m[0,:]
            my=m[1,:]
            mz=m[2,:]
            plt.figure()
            plt.title(title)
            plt.plot(sweep,mx,label='mx')
            plt.plot(sweep,my,label='my')
            plt.plot(sweep,mz,label='mz')
            #plt.plot(time,I,color='Red',label='I')
            plt.legend()
            # 3D Plot
            fig = plt.figure()
            ax3D = fig.add_subplot(projection='3d')
            cm = plt.get_cmap('Greys')
            r = 1
            u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
            xs = np.cos(u) * np.sin(v)
            ys = np.sin(u) * np.sin(v)
            zs = np.cos(v)
            ax3D.plot_surface(xs, ys, zs, alpha=0.3, color="white")
            ax3D.set_box_aspect((np.ptp(xs), np.ptp(ys), np.ptp(zs)))
            p3d_scatter = ax3D.scatter(mx, my, mz, s=30, c=col, marker='o') 
            p3d_plot = ax3D.plot(mx, my, mz)                                                                                
            plt.show()
            plt.close()
        return angles[:,0],angles[:,1]

    def getHarmonicSignal_fast(self,B_sweep=0,thetaB_sweep=0,phiB_sweep=0,timeSweep=0,initialGuess=[0,0],showSweep=False,switchGuess='Bext',tol=1e-10,B_coer=1e10,resistanceFunction=RH,returnAngles=False):
        """Resistance function must be a fucntion of theta,phi,B,thetaB,phiB IN THAT ORDER SPECIFICALLY"""
        prevShape=None
        sweeps=[B_sweep,thetaB_sweep,phiB_sweep,timeSweep]
        for sweep in [B_sweep,thetaB_sweep,phiB_sweep,timeSweep]:
            if not isinstance(sweep,(int,float)):
                currentShape=len(sweep)
                if prevShape is not None:
                    if prevShape!=currentShape:
                        print('Two or more of the provided sweeps do not have a compatible shape')
                        return
                prevShape=currentShape
        #print(f'Sweeping over {currentShape} points')
        for n,sweep in enumerate([B_sweep,thetaB_sweep,phiB_sweep,timeSweep]):
            if isinstance(sweep,(int,float)):
                sweeps[n]=[sweep for i in range(currentShape)]

       
        thetaP,phiP=self.sweep_equilibrium(B_sweep=B_sweep,thetaB_sweep=thetaB_sweep,phiB_sweep=phiB_sweep,j_modulation=1,timeSweep=0,initialGuess=initialGuess,showSweep=False,switchGuess='Bext',tol=tol,B_coer=B_coer)
        thetaM,phiM=self.sweep_equilibrium(B_sweep=B_sweep,thetaB_sweep=thetaB_sweep,phiB_sweep=phiB_sweep,j_modulation=-1,timeSweep=0,initialGuess=initialGuess,showSweep=False,switchGuess='Bext',tol=tol,B_coer=B_coer)
        
        Rp=resistanceFunction(thetaP,phiP,B_sweep,thetaB_sweep,phiB_sweep)
        Rm=resistanceFunction(thetaM,phiM,B_sweep,thetaB_sweep,phiB_sweep)
        R1w=(Rp+Rm)/2
        R2w=(Rp-Rm)/4
        if returnAngles:
            return R1w,R2w,(thetaP+thetaM)/2,(phiP+phiM)/2
        return R1w,R2w
    
    
if __name__=='__main__':
    test=quasiEquilibriumSystem()
    test.externalField()
    test.demagField(0.4577)
    test.fourFoldField(0.38,15)
    test.chargeCurrent()
    test.spinAccumulation()
    test.fieldLikeTorque(1e-3)
    #B_sweep=np.array([6.96,6.80706,6.60713,6.4074,6.2077,6.00783,5.80814,5.60842,5.4086,5.20888,5.00922,4.8095,4.60991,4.41019,4.21068,4.01081,3.81083,3.61076,3.41065,3.21092,3.01047,2.81031,2.61016,2.4098,2.20957,2.00906,1.80852,1.60798,1.40743,1.20654,1.00528,0.80517,0.6034,0.40243,0.2015,0.00106,-0.2002,-0.40045,-0.60134,-0.80337,-1.00359,-1.20478,-1.40594,-1.60675,-1.8077,-2.00858,-2.20943,-2.40988,-2.61046,-2.81079,-3.01103,-3.21159,-3.41151,-3.61174,-3.81188,-4.01202,-4.21195,-4.41191,-4.61177,-4.81162,-5.01121,-5.21105,-5.4106,-5.61046,-5.81007,-6.00958,-6.20908,-6.40856,-6.60823,-6.80836,-6.99,-6.01017,-5.01244,-4.01368,-3.01284,-2.01013,-1.00452,-7.33507E-4,-1.00475,-2.00994,-3.01237,-4.01317,-5.01231,-6.01013,-6.20992,-6.01055,-5.81134,-5.61192,-5.4122,-5.2128,-5.01306,-4.81349,-4.61397,-4.41408,-4.21437,-4.01441,-3.81438,-3.61417,-3.41396,-3.21396,-3.01351,-2.81309,-2.61273,-2.41216,-2.21156,-2.01057,-1.8096,-1.6085,-1.40749,-1.20624,-1.00476,-0.80437,-0.60211,-0.40102,-0.20031,-7.07496E-4,0.20167,0.40252,0.60364,0.80552,1.00572,1.20698,1.40801,1.60866,1.80935,2.00989,2.21049,2.41081,2.61121,2.81151,3.01166,3.2122,3.41202,3.61221,3.81222,4.01224,4.21209])
    B_sweep=1
    thetaB=90
    phiB=np.linspace(0,360,360)
    deltaPhi=-8

    def RH_CMR(theta,phi,B=0,thetaB=0,phiB=0,OHE=0,PHE=1,AHE=1):
        # Standard Hall resistance
        mx,my,mz=get_m(theta,phi)
        #bx,by,bz=get_m(thetaB,phiB)
        return AHE*mz+PHE*np.sin(np.deg2rad(theta))**2*np.sin(2*np.deg2rad(phi-deltaPhi))

    resistanceFunction=lambda theta,phi,B,thetaB,phiB: RH_CMR(theta,phi,B,thetaB,phiB,OHE=0,AHE=-0.92156,PHE=0.15)


    R1w,R2w,theta,phi=test.getHarmonicSignal_fast(B_sweep,thetaB,phiB,initialGuess=[thetaB,phiB[0]],showSweep=True,resistanceFunction=resistanceFunction,returnAngles=True)

    plt.figure()
    plt.scatter(phiB,R1w)
    plt.draw()

    plt.figure()
    plt.scatter(phi,R1w)
    plt.draw()

    plt.figure()
    plt.scatter(phi,R2w)
    plt.draw()

    plt.figure()
    plt.scatter(phiB,R2w)
    plt.show()