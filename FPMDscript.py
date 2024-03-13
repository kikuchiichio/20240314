#!/usr/bin/env python
# coding: utf-8

# # Ab-inito molecular dynamics by quantum algorithm:
# ## The simultaneous determination of wavefunctions and nuclei positions
# 
# This notebook shows a proof of concept of ab-inito molecular dynamics by quantum algorith, which
# simulataneously determin the wavefunctions and the nuclei positions.
# 
# - We simulate a simple molecule H3+.
# 
# - We use Hartree-Fock functional.
# 
# - We use the STO-3G basis.
# 
# - We express the total energy by an analytic function.
# 
# - The optima are the roots of the system of polynomial equations.
# 
# - The unknown variables in the system of polynomial equations represent the wavefunction and the the nuclei positions.
# 
# - By symbolic computation, we transform the system of polynomial equations into a form suitable to quantum algorithm.
# 
# - This notebook shows the computational steps up to the matrix-opeator prepations.
#   
# - The preparation of quantum circuits are the topics of other notebook.

# In[ ]:


#\
#  This cell defines the analytic formulas of molecular integrals.\
#\

import sympy
from sympy import symbols, Function,expand,sqrt
import numpy as np
import copy
from itertools import combinations_with_replacement, permutations,product

    
def S000000(A,B,RA,RB):
#
#   OVERLAP INTEGRAL between exp(-A*|r-RA|^2) and exp(-B*|r-RB|^2)
#
    RAB2=sum([(c1-c2)**2 for c1,c2 in zip(RA,RB)])
    PI=sympy.pi
    #print(A,B,RAB2,(PI/(A+B))**1.5)
    return (PI/(A+B))**1.5*sympy.exp(-A*B*RAB2/(A+B))


def K000000(A,B,RA,RB):
#
#  KINETIC INTEGRAL between exp(-A*|r-RA|^2) and exp(-B*|r-RB|^2)
#  Namely, the matrix element of the kinetic operator.
#
    RAB2=sum([(c1-c2)**2 for c1,c2 in zip(RA,RB)])
    PI=sympy.pi
    return A*B/(A+B)*(3.0-2.0*A*B*RAB2/(A+B))*(PI/(A+B))**1.5*sympy.exp(-A*B*RAB2/(A+B))   

def F0(ARG):
#
# BOYS F0 FUNCTIOM
#
    PI = sympy.pi
    if  type(ARG)==float and ARG < 1.0e-6:
        return 1 -ARG/3.
    if  type(ARG)==sympy.core.numbers.Zero and ARG < 1.0e-6:
        return 1 -ARG/3.
    else:
        #print("F0:ARG",ARG)
        if ARG!=sympy.S.Zero:
            return sympy.sqrt(PI/ARG)*sympy.erf(sqrt(ARG))/2
        else:
            return 1
            
def BOYS(N,X):
#
#  BOYS FUNCTION F_N(X)
#
    if N==0:
        return F0(X)
    else:
        Z=Symbol("Z")
        f=F0(Z)
        for _ in range(N):
            f=f.diff(Z)*(-1)
        return f.subs(Z,X)
    
def KXYZAB(a,b,RA,RB):
    mu=a*b/(a+b)
    RAB=[(c1-c2)**2 for c1,c2 in zip(RA,RB)]
    RAB2=sum(RAB)
    return sympy.exp(-mu*RAB2)

def T0000(a,b,c,d,RA,RB,RC,RD):
#
# CALCULATES TWO-ELECTRON INTEGRALS FOR UN-NORMALIZED PRIMITIVES
# a,b,c,d ARE THE EXPONENTS ALPHA, BETA, ETC.
# RAB2 EQUALS SQUARED DISTANCE BETWEEN CENTER A AND CENTER B, ETC.
#
# The integrand is the product of those functions.
#   exp(-a*|r1-RA|^2),exp(-b*|r1-RB|^2),1/|r1-r2|, exp(-c*|r2-RC|^2),exp(-d*|r2-RD|^2)
#
#
    def MF0(ARG):
    #
    # BOYS F0 FUNCTIOM
    #
        PI = sympy.pi
        if  type(ARG)==float and ARG < 1.0e-6:
            return 1 -ARG/3.
        if  type(ARG)==sympy.core.numbers.Zero and ARG < 1.0e-6:
            return 1 -ARG/3.
        else:
            #print("F0:ARG",ARG)
            if ARG!=sympy.S.Zero:
                return sympy.sqrt(PI/ARG)*sympy.erf(sqrt(ARG))/2
            else:
                return 1
    p=a+b
    q=c+d
    RP=[(a*c1+b*c2)/(a+b) for c1,c2 in zip(RA,RB)]
    RQ=[(c*c1+d*c2)/(c+d) for c1,c2 in zip(RC,RD)]
    alpha=p*q/(p+q)
    PI=sympy.pi
    RPQ=[(c1-c2)**2 for c1,c2 in zip(RP,RQ)]
    RPQ2=sum(RPQ)
    return 2*PI**(2.5)/p/q/sympy.sqrt(p+q)*KXYZAB(a,b,RA,RB)*KXYZAB(c,d,RC,RD)*MF0(alpha*RPQ2)

def V000000(a,b,RA,RB,RC):
#
# CALCULATES ONE-ELECTRON INTEGRALS FOR UN-NORMALIZED PRIMITIVES
#
#  The integrand is the product of those functions.
#   exp(-a*|r-RA|^2),exp(-b*|r-RB|^2),1/|r-RC|
#  
    p=a+b
    PI=sympy.pi
    RP=[(a*c1+b*c2)/(a+b) for c1,c2 in zip(RA,RB)]
    RPC=[(c1-c2)**2 for c1,c2 in zip(RP,RC)]
    RPC2=sum(RPC)
    return 2*PI/p*KXYZAB(a,b,RA,RB)*BOYS(0,p*RPC2)


def GetRP(a,b,RA,RB):
    AX,AY,AZ=RA
    BX,BY,BZ=RB
    PX=(a*AX+b*BX)/(a+b)
    PY=(a*AY+b*BY)/(a+b)
    PZ=(a*AZ+b*BZ)/(a+b)
    return [PX,PY,PZ]


# In[ ]:


z1,z2,z3,z4=sympy.symbols("z1 z2 z3 z4")
AX,AY,AZ,BX,BY,BZ,CX,CY,CZ,DX,DY,DZ=sympy.symbols("AX AY AZ BX BY BZ CX CY CZ DX DY DZ")


# In[ ]:


#
#  This cell defines 
#  the symbolic Formulas of the molecular integrals
#  for arbitrary atomic positions.
#
RA=[AX,AY,AZ]
RB=[BX,BY,BZ]
RC=[CX,CY,CZ]
RD=[DX,DY,DZ]


T0000expr=T0000(z1,z2,z3,z4,RA,RB,RC,RD)
S000000expr=S000000(z1,z2,RA,RB)
K000000expr=K000000(z1,z2,RA,RB)
V000000expr=V000000(z1,z2,RA,RB,RC)

#
print("Latex output")
#
for exprs in [T0000expr,S000000expr,K000000expr,V000000expr]:
    print("\\begin{align}\n",sympy.latex(exprs),"\n\\end{align}\n")


# In[ ]:


#
#  This cell defines the molecular integrals as an analytic function, 
#  using STO-3G basis.
#  The function is specified to H3+, dependent on the bond length R.
#  
#
import numpy
import numpy as np
R_length=0.8649/0.529177 
R_length=0.9/0.529177 

R=sympy.symbols("R",positive=True)
R_length=R


# Molecular Structure. 
# Three hydrogen atoms (A,B,and C) are placed on an equilateral triangle.
atomA=[0,0,0]
atomB=[R_length,0,0]
atomC=[R_length*1/2,0,R_length*sympy.sqrt(3)/2]



# Atomic Charge
ZA=1
ZB=1
ZC=1
atoms=[atomA,atomB,atomC]
ZS=[ZA,ZB,ZC]


# ZETA, COEFF, EXPOM 
#  ~ The parameters in Gaussian type atomic basis functions.

ZETA1=1.24
ZETA2=ZETA1
ZETA3=ZETA1
ZETAS=[ZETA1,ZETA2,ZETA3]
COEF=[0.444635,0.535328,0.154329]
EXPON=[0.109818,0.405771,2.22766]

# DA,CA,DB,CB,DC,CC,DD,CD: Working arrays.
DA=[0]*3
CA=[0]*3
DB=[0]*3
CB=[0]*3
DC=[0]*3
CC=[0]*3
DD=[0]*3
CD=[0]*3
PI=sympy.pi

#
# TT: Two-electron repulsion 
#
TT=[[[[0 for _ in range(3)] for _  in range(3)] for _ in range(3)] for _ in range(3)]
for ID in range(3):
    for JD in range(3):
        for KD in range(3):
            for LD in range(3):
                RA=[AX,AY,AZ]=atoms[ID]
                RB=[BX,BY,BZ]=atoms[JD]
                RC=[CX,CY,CZ]=atoms[KD]
                RD=[DX,DY,DZ]=atoms[LD]
                N=3
                for i in range(N):
                    #print(i,EXPON[N-1][i],COEF[N-1][i],ZETA1,ZETA2)
                    CA[i]=EXPON[i]*ZETAS[ID]**2
                    DA[i]=COEF[i]*((2*CA[i]/PI)**0.75)
                    CB[i]=EXPON[i]*(ZETAS[JD]**2)
                    DB[i]=COEF[i]*((2*CB[i]/PI)**0.75)
                    CC[i]=EXPON[i]*(ZETAS[KD]**2)
                    DC[i]=COEF[i]*((2*CC[i]/PI)**0.75)
                    CD[i]=EXPON[i]*(ZETAS[LD]**2)
                    DD[i]=COEF[i]*((2*CD[i]/PI)**0.75)
                    
                N=3
                V=0
                for I in range(N):            
                    for J in range(N):
                        for K in range(N):
                            for L in range(N):
                                ca=CA[I]
                                cb=CB[J]
                                cc=CC[K]
                                cd=CD[L]
                                #print(ca,cb,cc,cd,sympy.N(T0000(ca,cb,cc,cd,RA,RB,RC,RD)))
                                
                                V=V+T0000(ca,cb,cc,cd,RA,RB,RC,RD)*DA[I]*DB[J]*DC[K]*DD[L]
                V=sympy.N(V)
                #print(RA,RB,RC,RD,V)
                TT[ID][JD][KD][LD]=V

#
# VAB_C: The matrix element of the coulomb potential from atom "C"  
#

VAB_C=[[[0 for _ in range(3)] for _  in range(3)] for _ in range(3)]
for ID in range(3):
    for JD in range(3):
        for KD in range(3):
                RA=[AX,AY,AZ]=atoms[ID]
                RB=[BX,BY,BZ]=atoms[JD]
                RC=[CX,CY,CZ]=atoms[KD]
                ZC=ZS[KD]
                N=3
                for i in range(N):
                    #print(i,EXPON[N-1][i],COEF[N-1][i],ZETA1,ZETA2)
                    CA[i]=EXPON[i]*ZETAS[ID]**2
                    DA[i]=COEF[i]*((2.0*CA[i]/PI)**0.75)
                    CB[i]=EXPON[i]*(ZETAS[JD]**2)
                    DB[i]=COEF[i]*((2.0*CB[i]/PI)**0.75)
                    CC[i]=EXPON[i]*(ZETAS[KD]**2)
                    DC[i]=COEF[i]*((2.0*CC[i]/PI)**0.75)
                    CD[i]=EXPON[i]*(ZETAS[LD]**2)
                    DD[i]=COEF[i]*((2.0*CD[i]/PI)**0.75)
                    
                N=3
                V=0
                for I in range(N):            
                    for J in range(N):
                        
                                ca=CA[I]
                                cb=CB[J]
                                #V000000(a,b,RA,RB,RC)
                                V=V+V000000(ca,cb,RA,RB,RC)*DA[I]*DB[J]*(-ZC)
                V=sympy.N(V)
                #print(RA,RB,RC,RD,V)
                VAB_C[ID][JD][KD]=V

#
#  The overlap integral between two atomic basis placed at centers A and B
#
SAB=[[0 for _ in range(3)] for _  in range(3)]              
for ID in range(3):
    for JD in range(3):
        RA=[AX,AY,AZ]=atoms[ID]
        RB=[BX,BY,BZ]=atoms[JD]
        for i in range(N):
            #print(i,EXPON[N-1][i],COEF[N-1][i],ZETA1,ZETA2)
            CA[i]=EXPON[i]*ZETAS[ID]**2
            DA[i]=COEF[i]*((2.0*CA[i]/PI)**0.75)
            CB[i]=EXPON[i]*(ZETAS[JD]**2)
            DB[i]=COEF[i]*((2.0*CB[i]/PI)**0.75)
        V=0
        for I in range(N):            
            for J in range(N):
                        ca=CA[I]
                        cb=CB[J]
                        #V000000(a,b,RA,RB,RC)
                        V=V+S000000(ca,cb,RA,RB)*DA[I]*DB[J]
        V=sympy.N(V)
        #print(RA,RB,V)
        SAB[ID][JD]=V

#
#  The Kinetic integral between two atomic basis placed at centers A and B
#
KAB=[[0 for _ in range(3)] for _  in range(3)]              
for ID in range(3):
    for JD in range(3):
        RA=[AX,AY,AZ]=atoms[ID]
        RB=[BX,BY,BZ]=atoms[JD]
        for i in range(N):
            #print(i,EXPON[N-1][i],COEF[N-1][i],ZETA1,ZETA2)
            CA[i]=EXPON[i]*ZETAS[ID]**2
            DA[i]=COEF[i]*((2.0*CA[i]/PI)**0.75)
            CB[i]=EXPON[i]*(ZETAS[JD]**2)
            DB[i]=COEF[i]*((2.0*CB[i]/PI)**0.75)
        V=0
        for I in range(N):            
            for J in range(N):
                        ca=CA[I]
                        cb=CB[J]
                        #V000000(a,b,RA,RB,RC)
                        V=V+K000000(ca,cb,RA,RB)*DA[I]*DB[J]
        V=sympy.N(V)
        #print(RA,RB,V)
        KAB[ID][JD]=V


# ## The formulas used in the construction of the objective function
# 
# 
# ## Functions used to describe physical interactions.
# 
# TT(z1,z2,z3,z4,RA,RB,RC,RD):
# \begin{align}
#  \frac{\pi^{3.0} \sqrt{\frac{z_{1} + z_{2} + z_{3} + z_{4}}{\left(z_{1} + z_{2}\right) \left(z_{3} + z_{4}\right) \left(\left(- \frac{CX z_{3} + DX z_{4}}{z_{3} + z_{4}} + \frac{AX z_{1} + BX z_{2}}{z_{1} + z_{2}}\right)^{2} + \left(- \frac{CY z_{3} + DY z_{4}}{z_{3} + z_{4}} + \frac{AY z_{1} + BY z_{2}}{z_{1} + z_{2}}\right)^{2} + \left(- \frac{CZ z_{3} + DZ z_{4}}{z_{3} + z_{4}} + \frac{AZ z_{1} + BZ z_{2}}{z_{1} + z_{2}}\right)^{2}\right)}} e^{- \frac{z_{1} z_{2} \left(\left(AX - BX\right)^{2} + \left(AY - BY\right)^{2} + \left(AZ - BZ\right)^{2}\right)}{z_{1} + z_{2}}} e^{- \frac{z_{3} z_{4} \left(\left(CX - DX\right)^{2} + \left(CY - DY\right)^{2} + \left(CZ - DZ\right)^{2}\right)}{z_{3} + z_{4}}} \operatorname{erf}{\left(\sqrt{\frac{\left(z_{1} + z_{2}\right) \left(z_{3} + z_{4}\right) \left(\left(- \frac{CX z_{3} + DX z_{4}}{z_{3} + z_{4}} + \frac{AX z_{1} + BX z_{2}}{z_{1} + z_{2}}\right)^{2} + \left(- \frac{CY z_{3} + DY z_{4}}{z_{3} + z_{4}} + \frac{AY z_{1} + BY z_{2}}{z_{1} + z_{2}}\right)^{2} + \left(- \frac{CZ z_{3} + DZ z_{4}}{z_{3} + z_{4}} + \frac{AZ z_{1} + BZ z_{2}}{z_{1} + z_{2}}\right)^{2}\right)}{z_{1} + z_{2} + z_{3} + z_{4}}} \right)}}{\left(z_{1} + z_{2}\right) \left(z_{3} + z_{4}\right) \sqrt{z_{1} + z_{2} + z_{3} + z_{4}}} 
# \end{align}
# 
# S(z1,z2,RA,RB):
# \begin{align}
#  \pi^{1.5} \left(\frac{1}{z_{1} + z_{2}}\right)^{1.5} e^{- \frac{z_{1} z_{2} \left(\left(AX - BX\right)^{2} + \left(AY - BY\right)^{2} + \left(AZ - BZ\right)^{2}\right)}{z_{1} + z_{2}}} 
# \end{align}
# 
# K(z1,z2,RA,RB):
# \begin{align}
#  \frac{\pi^{1.5} z_{1} z_{2} \left(- \frac{2.0 z_{1} z_{2} \left(\left(AX - BX\right)^{2} + \left(AY - BY\right)^{2} + \left(AZ - BZ\right)^{2}\right)}{z_{1} + z_{2}} + 3.0\right) \left(\frac{1}{z_{1} + z_{2}}\right)^{1.5} e^{- \frac{z_{1} z_{2} \left(\left(AX - BX\right)^{2} + \left(AY - BY\right)^{2} + \left(AZ - BZ\right)^{2}\right)}{z_{1} + z_{2}}}}{z_{1} + z_{2}} 
# \end{align}
# 
# VABC(z1,z2,RA,RB):
# \begin{align}
#  \frac{\pi^{\frac{3}{2}} \sqrt{\frac{1}{\left(z_{1} + z_{2}\right) \left(\left(- CX + \frac{AX z_{1} + BX z_{2}}{z_{1} + z_{2}}\right)^{2} + \left(- CY + \frac{AY z_{1} + BY z_{2}}{z_{1} + z_{2}}\right)^{2} + \left(- CZ + \frac{AZ z_{1} + BZ z_{2}}{z_{1} + z_{2}}\right)^{2}\right)}} e^{- \frac{z_{1} z_{2} \left(\left(AX - BX\right)^{2} + \left(AY - BY\right)^{2} + \left(AZ - BZ\right)^{2}\right)}{z_{1} + z_{2}}} \operatorname{erf}{\left(\sqrt{\left(z_{1} + z_{2}\right) \left(\left(- CX + \frac{AX z_{1} + BX z_{2}}{z_{1} + z_{2}}\right)^{2} + \left(- CY + \frac{AY z_{1} + BY z_{2}}{z_{1} + z_{2}}\right)^{2} + \left(- CZ + \frac{AZ z_{1} + BZ z_{2}}{z_{1} + z_{2}}\right)^{2}\right)} \right)}}{z_{1} + z_{2}} 
# \end{align}
# 
# ## Numerically given parameters  {C(i),EXPON(i),COEF(I),D(i)|i=1,2,3} and zeta.
# \begin{align}
# C(i)=EXPON(i)\cdot\zeta^2\ (i=1,2,3)
# \end{align}
# 
# \begin{align}
# D(i)=COEF(i)\cdot\left(\frac{2C(i)}{\pi}\right)^{\frac{3}{4}}\ (i=1,2,3)
# \end{align}
# 
# 
# ##  The integrals describing physical interactions
# \begin{align}
# T(R_a,R_b,R_c,R_d)=\sum_{i=1}^{3}\sum_{j=1}^{3}\sum_{k=1}^{3}\sum_{l=1}^{3}
# T_{0000}(C(i),C(j),C(k),C(l),R_a,R_b,R_c,R_d)\cdot D(i)D(j)D(k)D(l)
# \end{align}
# 
# \begin{align}
# V(R_a,R_b,R_c)=\sum_{i=1}^{3}\sum_{j=1}^{3}V_{000000}(C(i),C(j),R_a,R_b,R_c)D(i)D(j)\cdot(-Z_c)
# \end{align}
# 
# \begin{align}
# Kin(R_a,R_b)=\sum_{i=1}^{3}\sum_{j=1}^{3}K_{000000}(C(i),C(j),R_a,R_b)D(i)D(j)
# \end{align}
# 
# 
# \begin{align}
# S(R_a,R_b)=\sum_{i=1}^{3}\sum_{j=1}^{3}S_{000000}(C(i),C(j),R_a,R_b)D(i)D(j)
# \end{align}
# 
# ### The parameters $C(i)$ and $D(i)$ satisfy the relation:
# \begin{align}
# S(R_a,R_a)=1
# \end{align}
# 
# ## For brevity, $R_a$ $R_b$ $R_c$ and $R_d$ are replaced with index I,J,K and L.
# \begin{align}
# S(R_a,R_b) &\rightarrow S(I,J) \\
# Kin(R_a,R_b) & \rightarrow Kin(I,J) \\
# VABC(R_a,R_b,R_c) & \rightarrow VABC(I,J,K) \\
# T(R_a,R_b,R_c,R_d) & \rightarrow T(I,J,K,L) 
# \end{align}
# 
# 
# ## Density matrix
# \begin{align}
# P=
# \begin{pmatrix}
# 2x^2 & 2xy & 2xz\\
# 2xy & 2y^2 & 2yz \\
# 2xz & 2yz & 2z^2 
# \end{pmatrix}
# \end{align}
# 
# ## Bare Hamiltonian Matrix
# \begin{align}
# H(I,J)=Kin(I,J)+\sum_{K=1}^3 VABC(I,J,K)
# \end{align}
# 
# ## Electron-Electron Potential 
# \begin{align}
# G(I,J)=\sum_{L=1}^3\sum_{k=1}^3 P(K,L)\cdot(TT(I,J,K,L)-\frac{1}{2}TT(I,L,J,K))
# \end{align}
# 
# ## Fock Matrix
# \begin{align}
# F(I,J)=H(I,J)+G(I,J)
# \end{align}
# 
# ## Total Energy (Electronic Part)
# \begin{align}
# EN=\frac{1}{2}\sum_{I=1}^3\sum_{J=1}^3P(I,J)\cdot(H(I,J)+F(I,J))
# \end{align}
# 
# ## Total Energy (Electronic Part + Nuclear Interaction)
# \begin{align}
# EN=\frac{1}{2}\sum_{I=1}^3\sum_{J=1}^3P(I,J)\cdot(H(I,J)+F(I,J))+\frac{3}{R}
# \end{align}
# 
# ## Normalization Condition of the wavefunction
# \begin{align}
# \sum_{I=1}^3\sum_{J=1}^3 P(I,J)S(I,J)=2
# \end{align}
# 
# 
# ## The obnective function (with a Lagrange multiplier e)
# \begin{align}
# f(x,y,z,R)&=\frac{1}{2}\sum_{I=1}^3\sum_{J=1}^3P(I,J)*(H(I,J)+F(I,J))+\frac{3}{R}\\& -e\cdot\left(\sum_{I=1}^3\sum_{J=1}^3 P(I,J)S(I,J)-2\right)
# \end{align}
# 
# 
# ## Symmetry
# From the definition, there are following relations
# 
# \begin{align}
# T(I,J,K,L)=T(J,I,K,L)
# \end{align}
# 
# \begin{align}
# T(I,J,K,L)=T(J,I,L,K)
# \end{align}
# 
# \begin{align}
# T(I,J,K,L)=T(K,L,I,J)
# \end{align}
# 
# 
# 

# ## Reduction of complexity in symbolic computation
# 
# In principle, it is able to optimize all parameters in the energy functional.
# However, the comutation of the Grobner basiis is still time-consuming 
# and we should reduce the complexity of the problem. To this end, we simply utilize 
# the geomertic symmetry of the ground state of the molecule. The LCAO wavefunction is
# given by 
# \begin{align}
# \phi(r)=\sum_{i=A,B,C} x\, \phi(r-R_i)
# \end{align}
# with one unknown $x$. The three  atoms in the molecules are located at the vertices of an equilateral triangle with the edge length $R$. 

# ## The preparation of the energy functional

# In[ ]:


#
#  This cell computes the analytic representation of the total ebergy as 'OBJE'
#  which depends on three variables (x,R,e), 
#  namely, the wavefunction, the atomic coordinate, and the orbital energy.
#
def FORMG():
#
# Calculates G matrix from density matrix and two-electron integrals
#
    for I in range(3):
        for J in range(3):
            G[I][J]=0
            for K in range(3):
                for L in range(3):
                    #print(I,J,K,L,P[K][L],TT[I][J][K][L],TT[I][L][J][K],G[I][J])
                    G[I][J]+=P[K][L]*(TT[I][J][K][L]-0.5*TT[I][L][J][K])
    #print("G",G)
    return 


def SCF_SYMBOL3():
#
# PREPARES THE ANALYTIC FORMULA OF THE TOTAL ENERGY.
#   Currently
    P=[[0 for _ in range(3)] for _ in range(3)]
    P2=[[0 for _ in range(3)] for _ in range(3)]
    G=[[0 for _ in range(3)] for _ in range(3)]
    G2=[[0 for _ in range(3)] for _ in range(3)]
    F=[[0 for _ in range(3)] for _ in range(3)]
    F2=[[0 for _ in range(3)] for _ in range(3)]
    H=[[0 for _ in range(3)] for _ in range(3)]
    x,y,z,u,v,w=symbols("x y z u v w")
    y=x
    z=x
    PI=np.pi
    CRIT=1.0e-4
    MAXIT=25
    ITER=0
    for I in range(3):
        for J in range(3):
            P[I][J]=0.
    P[0][0]=x*x*2
    P[0][1]=x*y*2
    P[0][2]=x*z*2
    P[1][0]=y*x*2
    P[1][1]=y*y*2
    P[1][2]=y*z*2
    P[2][0]=z*x*2
    P[2][1]=z*y*2
    P[2][2]=z*z*2

    for I in range(3):
        for J in range(3):
            G[I][J]=0
            G2[I][J]=0
            for K in range(3):
                for L in range(3):
                    G[I][J]+=P[K][L]*TT[I][J][K][L]-0.5*P[K][L]*TT[I][L][J][K]
    H=[[0 for _ in range(3)] for _ in range(3)]
    for I in range(3):
        for J in range(3):
            H[I][J]=KAB[I][J]
            for K in range(3):
                H[I][J]+=VAB_C[I][J][K]

    for i in range(3):
        for j in range(3):
            F[i][j]=H[i][j]+G[i][j]

    EN=0
    for i in range(3):
        for j in range(3):
            EN+=0.5*P[i][j]*(H[i][j]+F[i][j])
    ENT=EN
    for i in range(3):
        for j in range(i+1,3):
            RPQ=[(c1-c2)**2 for c1,c2 in zip(atoms[i],atoms[j])]
            RPQ2=sum(RPQ)
            #print(RPQ2)
            ENT+=ZS[i]*ZS[j]/sympy.sqrt(RPQ2)
            
    # EN : Electronic Energy
    # ENT : Electronic Energy + Nuclear Repulsion
    # H : Bare Hamiltonian
    # F : Hamiltonian containing electron-electron interaction
    # F2 : Currently zero. 
    # P : Density matrix

    return EN,ENT,F,F2,H,P

def GetNormS(vec,SAB):
    # (v|S|v): The overlap. 
    #  It should be 1 (as a constraint) 
    V=0
    for i in range(len(vec)):
        for j in range(len(vec)):
            V+=vec[i]*SAB[i][j]*vec[j]
    return V

#
# (x,y,z) Wavefunction of the molecular orbita;
#
x,y,z=symbols("x y z")
#
#  ENT: Total Energy of the molecule. 
#  Currently it is a function of R (the atomic distance)
#
EN,ENT,FM,FM2,HM,PM=SCF_SYMBOL3()
#
# e: Lagrange multiplier, or the orbital energy
#
e=symbols("e")
#
#  OBJ: The objective function a function of (x,y,z,e,R) -> simplified to that of (x,e,R)
SAB[0][0]=1
SAB[1][1]=1
SAB[2][2]=1


#
# The objective function. It is a function of R,x,y,z
#
x,y,z=sympy.symbols("x y z")
y=x
z=x
OBJ=ENT-2*e*(GetNormS([x,y,z],SAB)-1)
#
#OBJ=ENT-2*e*(GetNormS([x,y,z],SAB)-1)


# In[ ]:


OBJE=OBJ.subs(y,x).subs(z,x).expand()
#OBJE


# In[ ]:


from sympy import latex, Integral, sqrt
#print(latex(OBJE))


# ## The polynomial approximation of the energy funcional
# 
# Using the Taylor expansion around R=R_c up to the 4th order, we make the polynomial approximation of the energy funcional

# In[ ]:


#  4th order series expansion of OBJE=E(x,R) with respect to R.
#
#OBJEP=sympy.series(OBJE, R, 1.8, 4, "+")
#OBJEP=OBJEP.removeO()
def myseries(f,v,vc,n):
    r=f.subs(v,vc)
    k=1
    for i in range(1,n+1):
        DF=sympy.diff(f,v,i).subs(v,vc)
        k*=i
        r+=DF*(v-vc)**i/k
    return r.expand()
#4th order series expansion of OBJE=E(x,R) with respect to R　around Rc=6.4
OBJEP=myseries(OBJE, R, 1.8, 3)    


# In[ ]:


#
# This cell prepares the objective function with integer coefficients.
#  The coefficients {C_i} in the total energy 
#  are approximated by fractional numbers: {D_i/10^n} 
#  which are given by integers {D_i} and a common denominator 10^n. 
#  We choose n=4.
#
#  'getF' is the polynomial, defined as 
#   (the approximated total energy) * 10^n. 
#   Henceforce 'getF' is the objective function. 
#
def getINT(x,N):
#
#  ROWND DOWN x*N AFTER THE DECIMAL POINT and get AN INTEGER M
#  THEN X is approximated by M/N. 
#
    return (int(np.floor(x*N)))

def getPowerProduct(args,tm):
    V=1
    for w,p in zip(list(args[1:]),tm):
        V*= w**p
    return V
    
OBJ2=OBJEP.expand()
ENTS3=sympy.poly(sympy.N(OBJ2))
AT=ENTS3.terms()


getF=0
for tm in AT:
    #p0,p1,p2,p3,p4,p5,p6,p7=tm[0]
    cf=tm[1]
    #print(p0,p1,p2,cf)
    #getF+=x**p0* y**p1* z**p2* u**p3* v**p4* w**p5* e**p5* f**p7*getINT(cf,10000)
    getF+=getPowerProduct(ENTS3.args,tm[0])*getINT(cf,10**8)


# ## Validity check of the polynomial approximation.
# 
# We check the validity of the polynomial approximation, by comparing the total energies of the exact and the approximated functionals at various values of $R$. The plot shows that the polynomial approximation shall imitate the behaviour of the exact function well.

# In[ ]:


# Wavefunction is currently given by phi:=x*f(r) with the unknown x.
# <x*f|SAB|x*f> -> M_AB= <f|SAB|f>  
M_AB=(GetNormS([x,x,x],SAB).expand()/(x*x)).expand()
#
# N_CD = <x*phi|SAB|x*phi> - 
N_CD=GetNormS([x,x,x],SAB)-1
ZS=list()
for RP in np.linspace(1.5,2.5,51,endpoint=True):
    xv2=1/M_AB.subs(R,RP)
    xv=sympy.sqrt(xv2)
    #
    #  so that <xv*f|SAB|xv*f>=1
    #
    t=(RP,N_CD.subs(x,xv).subs(R,RP),OBJE.subs(R,RP).subs(x,xv).subs(e,0),\
        sympy.N(OBJEP.subs(R,RP).subs(x,xv).subs(e,0)),\
        sympy.N(getF.subs(R,RP).subs(x,xv).subs(e,0))/10**8,\
      )
    ZS.append(list(t))
import pandas as pd
ZSDF=pd.DataFrame(ZS)


# In[ ]:


import matplotlib.pyplot as plt
#plot(ZS[0],ZS[1])

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111)

ax.plot(ZSDF[0], ZSDF[2],label='Exact E(R)')
ax.plot(ZSDF[0], ZSDF[3],label='3rd Order Polynomial Approximation of E(R)')
ax.scatter(ZSDF[0], ZSDF[4],label='3rd Order Polynomial Approximation of E(R) with fractional coefficients')
ax.set_xlabel('R')
ax.set_ylabel('ENERGY')
plt.legend(loc='best')
plt.show()
plt.savefig("ENAPAPROX.pdf")


# In[ ]:


getF


# In[ ]:


str(getF)


# In[ ]:


#
#  There remain three valiables (x,e,R):
#    the wavefunction, the orbital energy, and the bond length
#
print(str(ENTS3.args[1:]))


# ## We solve the set of polynomial equations using Singular (a symbolic computation package)
# 
# We prepare a script of Singular that conduct following computations.
# 
# 1. The ring (r) definition, with degree reverse lexicographic monomial ordering
# 2. The ideal (I) definition
# 3. The Groebner basis (SI) computation
# 4. The Groebner basis (j) computation, with the change of monomial ordering, from "degree reverse lexicographic" to "lexocographic" type
# 6. The numerical solving using the triangular decomposition of the Groebner basis (j).
# 7. The display of the obtained solutions (x,e,R) and the corresponding values of the energy functional E(x,e,R)
# 8. The computation of transformation matrices that represent the action of (x,e,R) in the quotient ring.
# 
# We evoke Singular from this notebook and retrieve the computed result. 

# In[ ]:


getF
Fargs=str(ENTS3.args[1:])
stringQ='option(noredefine);LIB "solve.lib";option(redSB);\n'
stringQ+='ring r=0,'+Fargs+',dp;\n'+'poly OBJ='+str(getF)+';\n'
stringQ+='list diffs;\n'
stringQ+='for(int i=1;i<=nvars(r); i=i+1){diffs=insert(diffs,diff(OBJ,var(i)));}\n'
stringQ+='ideal I=0;\n'
stringQ+='for(int i=1;i<=nvars(r); i=i+1){I=I+diff(OBJ,var(i));}\n'
stringQ+='print(I);'
stringQ+='ideal SI=std(I);\n'
stringQ+='print(SI);'
stringQ+='ring s=0,'+Fargs+',lp;\n'
stringQ+='setring s;\n'
stringQ+='ideal j=fglm(r,SI);\n;j;'
stringQ+='def RS=triang_solve(j,50);\n'
stringQ+='setring RS;rlist;'
stringQ+='poly OBJ=fetch(r,OBJ);\
ideal I=fetch(r,I);\
OBJ;\
for (i=1;i<=size(rlist);i=i+1)\
{\
list substv;\
\
poly OBJ2=OBJ;\
for (int k=1; k<=nvars(RS); k=k+1)\
{\
 OBJ2=subst(OBJ2,var(k), rlist[i][k]);\
}\
substv=insert(substv,OBJ2);\
\
for (int l=1;l<size(I);l=l+1)\
{\
poly OBJ2=I[l];\
 for (int k=1; k<=nvars(RS); k=k+1)\
 {\
  OBJ2=subst(OBJ2,var(k), rlist[i][k]);\
 }\
 substv=insert(substv,OBJ2);\
}\
print("CHECK THE SOLUTION");\
print(substv);\
write(":w save_i.txt",rlist);\
}\
LIB "rootsmr.lib";\
setring r;\
ideal b = qbase(SI);\
matrix mx = matmult(x,b,SI);\
matrix me = matmult(e,b,SI);\
matrix mr = matmult(R,b,SI);\
write(":w saved_mx.txt",string(mx));\
write(":w saved_me.txt",string(me));\
write(":w saved_mr.txt",string(mr));\
write(":w saved_b.txt",string(b));\
'

text_file = open("SCRIPT.txt", "w")
on = text_file.write(stringQ)
 
#close file
text_file.close()


# In[ ]:


import subprocess
cp = subprocess.run(" Singular<SCRIPT.txt", shell=True,capture_output=True,text=True)
#print("stdout:", cp.stdout)
#print("stderr:", cp.stderr)


# In[ ]:


cp = subprocess.run(" Singular<SCRIPT.txt > outputs.txt", shell=True,capture_output=True,text=True)


# ## The result computed by Singular 
# 
# For the sake of clarity, the output from Singular is arranged in the Table given below. 
# The rows show the computed value of (x,e,R) and ETOT. ETOT is defined by (10^4)*(total energy). 
# 
# Note that only the real solutions are admissible. Additionally, as we use the series expansion with respect to $R$, the most plausible solutions would be located at a certain value of $R$ closest to the center of the series expansion ($R_c=1.8$). From this criterion, we judge that the solutions (No 0 and 1) are the optima.

# In[ ]:


with open("outputs.txt") as f:
    for i,x in enumerate(f):
        print(i,x)


# In[ ]:


with open("outputs.txt") as f:
    SOLS=list()
    for i,x in enumerate(f):
        if i>=36 and i<=189:
            SOLS.append(x.replace("\n",""))

SOLS=np.array(SOLS).reshape(22,7)

with open("outputs.txt") as f:
    EOLS=list()
    ic=0
    for i,x in enumerate(f):
        if i>=191 and i<=344:
            #print(i, (i-135)%6, x)  
            ic+=1
            EOLS.append(x.replace("\n",""))
EOLS=np.array(EOLS).reshape(22,7)
pd.set_option("display.float_format", "{:.4f}".format)
SOLS=pd.DataFrame(SOLS)
EOLS=pd.DataFrame(EOLS)
result = SOLS.join(EOLS,lsuffix='_l',rsuffix='_r')
result.iloc[0]


# In[ ]:


def stox(s):
    try:
        return float(s)
    except:
        s.replace("I","1j")
        return "Complex"

import copy
r2=pd.DataFrame(columns=["x","e","R","ETOT"])
for ci,di in zip(["2_l","4_l","6_l","6_r"],["x","e","R","ETOT"]):
    
    r2[di]=[stox(x) for x in result[ci]]

print(r2)


# In[ ]:


# IN THE CASE OF WSL+LINUX
#import subprocess
#cp = subprocess.run("wsl Singular<SCRIPT.txt", shell=True,capture_output=True,text=True)
#print("stdout:", cp.stdout)
#print("stderr:", cp.stderr)


# ## The numerical solving of the equation using the transformation matrices.
# 
# Let $I$ be the system of polynomial equations, whose roots give the optima of the molecular structure.
# 
# Let $G$ be the Groebner basis of the system of the polynomial $I$.
# 
# $G$ is defined in the ring $S=Q[x,e,R]$, where $Q$ is an arbitrary field.
# 
# The quotient ring $S / G$ is compsed of the finite numer of monomial bases $B$. 
# 
# In other words, it is a linear space over $B$.
# 
# The multiplication of $(x,e,R)$ to $B$ gives the linear transformation of $B$:
# \begin{align}
# x \cdot B =  B\cdot M_x
# \end{align}
# \begin{align}
# e \cdot B =  B\cdot M_r
# \end{align}
# \begin{align}
# R \cdot B =  B\cdot M_R
# \end{align}
# 
# Namely, in the above multiplications, the enries in the row vector $B$ is transormed by matrices $M_x, M_y$, and $M_R$. Those matrices do not include (x,e, R).
# 
# We could read the above relations as eigenvalue problems, where the actual values of $x, y$ and $R$ are given as the eigenvalues of  $M_x, M_y$, and $M_R$. Once these matrices are numerically defined, (x,y,R) are determined numerically. In other words, the given problem (to solve the system of polynomial equations) is restated as the set of eigenvalue problems.
# 
# Since $M_x, M_y$, and $M_R$ commute with eath other, they share common eigenvectors. Hence, if the eigenvalue problem for one of those matrices is solved and the eigenvectors $B$ are computed, the other eigenvalues are solved by these $B$.
# 
# As we have computed the Groebner basis of the given problem, $M_x, M_y$, and $M_R$ are computed without ambiguity. The computation by Singular, conducted above, have already gotten them and saved the data in three files.
# 
# Let us assert the above statesments.
# 

# In[ ]:


import numpy as np
import csv
def getmat(filename):
    rowmx=list()
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for t in reader:
            #print(t)
            rowmx.append(t)
    x=rowmx[0]
    y=np.array([eval(t) for t in x])
    return np.array(np.array_split(y,22))


# In[ ]:


mx=getmat('saved_mx.txt')    
mr=getmat('saved_mr.txt')    
me=getmat('saved_me.txt')   
#
#  The eigenvalue problem assumed in Singular is of the form v * M = p * v
#  On the other hand, in physics, we use M^T * w = p * w
#
mx=mx.transpose()
mr=mr.transpose()
me=me.transpose()


# In[ ]:


es,vs=np.linalg.eig(mx)
vs=vs.transpose()
es


# In[ ]:


for m in [mx,mr,me]:
    print(np.real(np.dot(vs[-1],np.dot(m,vs[-1]))))


# In[ ]:


Z=list()
for w in vs:
    ww=list()
    for m in [mx,mr,me]:
        ww.append(np.dot(w,np.dot(m,w)))
    Z.append(ww)


# In[ ]:


pd.set_option("display.float_format", "{:.4f}".format)
ZPD=pd.DataFrame(Z,columns=["x","R","e"])


# In the Table shown below, the eigenvalues of the transformation matrices are given. 
# The rows in the table show the sequential numbers of the normalized eigenvevtors $|i)$ of the transformation matrix $m_x$ (for i=0,...,13) , the expectation values $(i|m_x|i)$,
# $(i|m_R|i)$, and $(i|m_e|i)$. 
# 
# We should discard the solutions of complex values and adopt only the real solutions. Moreover, we should assess the real solutions. As the most hazardous approximation in the computation is the series expansion with respect to $R$, we should seek the solutions that fall on the proper range of this approximation.  We could guess that $|4)$ and $|9)$ would be appropriate, because $(4|m_R|4)$ and $(9|m_R|9)$ are located at the point closest to the center of the series expansion (given by $R_c=1.8$). Indeed they are the points where the energy functional takes the minimum value.

# In[ ]:


print(ZPD.to_latex())


# ## What is next?
# 
# Now we arrived at the end of symboloc computation, and we now have the transformation matrices which encode the wavefunction and the nuclei position.  The eigenvalue problems of those matrices are also solved by quantum algorithms. 
# 
# The next notebook demonstrates how to design the quantum circuits for this purpose.
# 
# - The practicable algorithm is quantum phase estimation (QPE). 
# 
# - As the transformation matrices we have now are not hermitian, we should use the trick of block-encoding. It leads to the more complicated quantum circuits.
# 
# - We do not step into the design of the quantum circuits for the QPE for the following reasons. First, the QPE requires long, complicated quantum circuit, which would not be practicable by emulators. Second, the QPE is basically a repetation of a gate that conducts a unitary transformation; the important point is how to implement such a function in a gate. 
# 
# - Instead, we investigate the construction of the gates that carry out the block-encodings for the transformation matrices we have now possessed.
# - 
# 
# 
# 
# 
# 
# 
# 

# In[ ]:


import numpy as np
from scipy.linalg import expm, sinm, cosm,logm
import scipy
from qiskit_aer import AerSimulator
import numpy as np
from qiskit import QuantumCircuit


# In[ ]:


#
#  The functions given below in this cell is taken from FABLE library.
#
# [*FABLE: Fast Approximate Quantum Circuits for Block-Encodings*]
# (https://ieeexplore.ieee.org/abstract/document/9951292), 
# Daan Camps, Roel Van Beeumen,
# 2022 2022 IEEE International Conference on Quantum Computing and Engineering (QCE), 
# [arXiv:2205.00081](https://arxiv.org/abs/2205.00081).
#

def test_fable_real():
    n = 3
    a = np.random.randn(2**n, 2**n)++ 1j * np.random.randn(2**n, 2**n)

    simulator = AerSimulator(method="unitary")

    circ, alpha = fable(a)
    circ.save_state()
    u_be = simulator.run(circ).result().get_unitary().data
    np.testing.assert_array_almost_equal(
        a/alpha/2**n,  np.real(u_be[:2**n, :2**n])
    )


def test_fable_complex():
    n = 3
    a = np.random.randn(2**n, 2**n) + 1j * np.random.randn(2**n, 2**n)

    simulator = AerSimulator(method="unitary")

    circ, alpha = fable(a)
    circ.save_state()
    u_be = simulator.run(circ).result().get_unitary().data
    np.testing.assert_array_almost_equal(
        a/alpha/2**n,  u_be[:2**n, :2**n]
    )


def gray_code(b):
    '''Gray code of b.
    Args:
        b: int:
            binary integer
    Returns:
        Gray code of b.
    '''
    return b ^ (b >> 1)


def gray_permutation(a):
    '''Permute the vector a from binary to Gray code order.

    Args:
        a: vector
            1D NumPy array of size 2**n
    Returns:
        vector:
            Gray code permutation of a
    '''
    b = np.zeros(a.shape[0])
    for i in range(a.shape[0]):
        b[i] = a[gray_code(i)]
    return b


def sfwht(a):
    '''Scaled Fast Walsh-Hadamard transform of input vector a.

    Args:
        a: vector
            1D NumPy array of size 2**n.
    Returns:
        vector:
            Scaled Walsh-Hadamard transform of a.
    '''
    n = int(np.log2(a.shape[0]))
    for h in range(n):
        for i in range(0, a.shape[0], 2**(h+1)):
            for j in range(i, i+2**h):
                x = a[j]
                y = a[j + 2**h]
                a[j] = (x + y) / 2
                a[j + 2**h] = (x - y) / 2
    return a


def compute_control(i, n):
    '''Compute the control qubit index based on the index i and size n.'''
    if i == 4**n:
        return 1
    return 2*n - int(np.log2(gray_code(i-1) ^ gray_code(i)))


def compressed_uniform_rotation(a, ry=True):
    '''Compute a compressed uniform rotation circuit based on the thresholded
    vector a.

    Args:
        a: vector:
            A thresholded vector a a of dimension 2**n
        ry: bool
            uniform ry rotation if true, else uniform rz rotation
    Returns:
        circuit
            A qiskit circuit representing the compressed uniform rotation.
    '''
    n = int(np.log2(a.shape[0])/2)
    circ = QuantumCircuit(2*n + 1)

    i = 0
    while i < a.shape[0]:
        parity_check = 0

        # add the rotation gate
        if a[i] != 0:
            if ry:
                circ.ry(a[i], 0)
            else:
                circ.rz(a[i], 0)

        # loop over sequence of consecutive zeros
        while True:
            ctrl = compute_control(i+1, n)
            # toggle control bit
            parity_check = (parity_check ^ (1 << (ctrl-1)))
            i += 1
            if i >= a.shape[0] or a[i] != 0:
                break

        # add CNOT gates
        for j in range(1, 2*n+1):
            if parity_check & (1 << (j-1)):
                circ.cx(j, 0)

    return circ


def fable(a, epsilon=None):
    '''FABLE - Fast Approximate BLock Encodings.

    Args:
        a: array
            matrix to be block encoded.
        epsilon: float >= 0
            (optional) compression threshold.
    Returns:
        circuit: qiskit circuit
            circuit that block encodes A
        alpha: float
            subnormalization factor
    '''
    epsm = np.finfo(a.dtype).eps
    alpha = np.linalg.norm(np.ravel(a), np.inf)
    if alpha > 1:
        alpha = alpha + np.sqrt(epsm)
        a = a/alpha
    else:
        alpha = 1.0

    n, m = a.shape
    if n != m:
        k = max(n, m)
        a = np.pad(a, ((0, k - n), (0, k - m)))
        n = k
    logn = int(np.ceil(np.log2(n)))
    if n < 2**logn:
        a = np.pad(a, ((0, 2**logn - n), (0, 2**logn - n)))
        n = 2**logn

    a = np.ravel(a)
    #print("a",a)
    if all(np.abs(np.imag(a)) < epsm):  # real data
        a = gray_permutation(
                sfwht(
                    2.0 * np.arccos(np.real(a))
                )
            )
        # threshold the vector
        #print(a)
        if epsilon:
            a[abs(a) <= epsilon] = 0
        # compute circuit
        OA = compressed_uniform_rotation(a)
    else:  # complex data
        # magnitude
        a_m = gray_permutation(
                sfwht(
                    2.0 * np.arccos(np.abs(a))
                )
            )
        if epsilon:
            a_m[abs(a_m) <= epsilon] = 0

        # phase
        a_p = gray_permutation(
                sfwht(
                    -2.0 * np.angle(a)
                )
            )
        if epsilon:
            a_p[abs(a_p) <= epsilon] = 0

        # compute circuit
        OA = compressed_uniform_rotation(a_m).compose(
                compressed_uniform_rotation(a_p, ry=False)
            )

    circ = QuantumCircuit(2*logn + 1)

    # diffusion on row indices
    for i in range(logn):
        circ.h(i+1)

    # matrix oracle
    circ = circ.compose(OA)

    # swap register
    for i in range(logn):
        circ.swap(i+1,  i+logn+1)

    # diffusion on row indices
    for i in range(logn):
        circ.h(i+1)

    # reverse bits because of little-endiannes
    circ = circ.reverse_bits()

    return circ, alpha

#  The functions given above in this cell is taken from FABLE library.
#
# [*FABLE: Fast Approximate Quantum Circuits for Block-Encodings*]
# (https://ieeexplore.ieee.org/abstract/document/9951292), 
# Daan Camps, Roel Van Beeumen,
# 2022 2022 IEEE International Conference on Quantum Computing and Engineering (QCE), 
# [arXiv:2205.00081](https://arxiv.org/abs/2205.00081).
#
#circ, alpha = fable(a)


# In[ ]:


import numpy as np
import csv
def getmat(filename,N):
    rowmx=list()
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for t in reader:
            #print(t)
            rowmx.append(t)
    x=rowmx[0]
    y=np.array([eval(t) for t in x])
    return np.array(np.array_split(y,N))
#
#  Optimization of H3+, HF ground state
#
mx=getmat('saved_mx.txt',22)    
mr=getmat('saved_mr.txt',22)    
me=getmat('saved_me.txt',22)   
#mr=getmat('MRsave.txt')
mx=mx.transpose()
me=me.transpose()
mr=mr.transpose()


# In[ ]:


import array_to_latex as a2l
a2l.to_ltx(mx, frmt = '{:4.2f}', arraytype = 'array')


# The table shows the accuracy of the block encodings, evaluated by the square norms of the difference betwen the transformation matrices and the correspoinding block-encodings:
# $\|A-A_{BL}\|^2$
# where $A_{BL}$ is the top-left diagonal part of the Hermitian matrices generated by FABLE method.

# In[ ]:


simulator = AerSimulator(method="unitary")
circlist=list()
alphalist=list()
import copy
def HOWDIFFER(A,B):
    nx,ny=A.shape
    s=0
    for i in range(nx):
        for j in range(ny):
            s+=np.abs(A[i,j]-B[i,j])**2
    return s
            
   
DL=list()
circs=list()
for a,sa in zip([mx,me,mr],["$m_x$","$m_e$","$m_r$"]):
    n=5
    #a=expm(-1j*a)
    print(sa)
    circ, alpha = fable(a)
    circd=copy.deepcopy(circ)
    circs.append(circd)
    circ.save_state()
    u_be = simulator.run(circ).result().get_unitary().data
    circlist.append(u_be)
    alphalist.append(alpha)
    np.testing.assert_array_almost_equal(
        a/alpha/2**n,  u_be[:22, :22]
    )
    DL.append([sa,HOWDIFFER(a,  alpha*(2**n)*u_be[:22, :22])])
for a,sa in zip([mx,me,mr],["$exp(-\sqrt{-1}m_x)$","$exp(-\sqrt{-1}m_y)$","$exp(-\sqrt{-1}m_r)$"]):
    n=5
    a=expm(-1j*a)
    print(sa)
    circ, alpha = fable(a)
    circd=copy.deepcopy(circ)
    circs.append(circd)
    circ.save_state()
    u_be = simulator.run(circ).result().get_unitary().data
    circlist.append(u_be)
    alphalist.append(alpha)
    np.testing.assert_array_almost_equal(
        a/alpha/2**n,  u_be[:22, :22]
    )
    DL.append([sa,HOWDIFFER(a,  alpha*(2**n)*u_be[:22, :22])])
import pandas as pd
DL=pd.DataFrame(DL,columns=["M (GS)","$\|M-M_{BL}\|^2$"])
print(DL)


# In[ ]:


print(DL.to_latex())


# In[ ]:


print("DRAW THE QUANTUM CIRCUIT FOR BLOCK ENCODING")
print("THE UPPPER PART: THE HEAD OF THE QUANTUM CIRCUIT/ THE LOWER PART: THE TAIL")
cp=QuantumCircuit(circs[0].num_qubits)
for i in range(0,28):
    cp.append(circs[0]._data[i])
#cp.barrier()
##cp.append(circs[0],range(circs[0].num_qubits))
#cp.barrier()

for i in range(-28+len(circs[0]._data),len(circs[0]._data)):
    cp.append(circs[0]._data[i])
cp.draw(output='mpl', style={'backgroundcolor': '#EEEEEE'},filename="circ.pdf")

