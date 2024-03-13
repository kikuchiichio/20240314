# 20240314

Quantum Computation and Algebraic Molecular Geometry


We present small programs, which are the implementation of the algorithm used in our article.
The details are explained by our article that referrs to this repository.


(1) FPMDscript.py
 >> python3 FPMDscript.py

  First, the script generates a polynmial ideal I. 
  
   To compute the zeros of the ideal I is to derermine the nuclei and the wavefunction of a molecule simultaneously.
   In other words, it enables us to conduct First-Principles Molecular Dynamics (FPMD) 
   by computational algebraic geometry and quantum algorithms. 

  Second, it yields the Groebner basis and the transformation matrices in the quotient ring, using "SCRIPT_N.txt".

   The eigenvalues of the transformation matrices give the zeros of the given ideal I.

  Third, it composes the essential parts of quantum circuits for quantum phase estimation (with block-encoding), 
  by which the eigenvaues are computed and registered in quantum states.

(2) SCRIPT_N.txt output.txt

 >> Singular <SCRIPT_N.txt > outputs.txt

    This script conducts the computations related to computational algebraic geometry, using SINGULAR package.
    It is used as a subroutine duing the computation of FPMDscript.py.


(3) The raw data of transformation matrices (mx,me,mR)
     'saved_mx.txt', 'saved_me.txt' and 'saved_mr.txt'

The matrix data in these files are transformed to python numpy arrays.

To this end, use the following function. The matrices should be transposed.
     
def getmat(filename,N):
    rowmx=list()
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for t in reader:
            rowmx.append(t)
    x=rowmx[0]
    y=np.array([eval(t) for t in x])
    return np.array(np.array_split(y,N))

mx=getmat('saved_mx.txt',22)    
mr=getmat('saved_mr.txt',22)    
me=getmat('saved_me.txt',22)   
mx=mx.transpose()
me=me.transpose()
mr=mr.transpose()    

    
