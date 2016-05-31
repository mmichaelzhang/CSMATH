import numpy as np
import matplotlib.pyplot as plt 
x_global = []
y_global = []


def InfNorm(Vec):
    maxEle = abs(Vec[0])
    for ele in Vec:
        if abs(ele)>maxEle:
            maxEle = ele
    return maxEle
def MaxDiagEle(Mat):
    Mat = np.array(Mat)
    dim = Mat.shape
    max_diagEle = Mat[0][0]
    for i in range(1,dim[0]):
        if Mat[i][i]>max_diagEle:
            max_diagEle = Mat[i][i]
    return max_diagEle
def LM(calValue,calHession,initPara,x,errorThres1 = 0.00000001,errorThres2 = 0.00000001,errorThres3 = 0.00000001,tao = 0.001,maxIteTime = 40):
    iterator_count = 0
    v = 2
    currentPara = initPara.copy()
    J = calHession(currentPara)
    A = J.T*J
    currentError = x - calValue(currentPara)
    tmp_Error = (np.mat(currentError)).T
    g = J.T*tmp_Error
    stop_flag = (InfNorm(g)<=errorThres1)
    u = tao*MaxDiagEle(A)
    dim = A.shape[0]
    I = np.identity(dim)
    while (not stop_flag) and (iterator_count<maxIteTime):
        iterator_count += 1
        while 1:
            del_p = np.array(np.linalg.solve(A+u*I,g))
            del_p = np.reshape(del_p,initPara.shape)
            if np.linalg.norm(del_p)<=errorThres2*np.linalg.norm(currentPara):
                stop_flag = 1
            else:
                newPara = currentPara+del_p

                del_pT =  np.mat(del_p)             
                xxx = del_pT*(u*del_pT.T+g)
                xxx = (np.array(xxx)).reshape(1)
                #print xxx
                
                minus_resi = (np.linalg.norm(currentError)*np.linalg.norm(currentError)-np.linalg.norm(x-calValue(newPara))*np.linalg.norm(x-calValue(newPara)))/(xxx[0])
                print minus_resi                
                if minus_resi > 0 :
                    currentPara = newPara.copy()
                    J = calHession(currentPara)
                    A = J.T*J
                    currentError = x - calValue(currentPara)
                    tmp_Error = (np.mat(currentError)).T
                    g = J.T*tmp_Error
                    stop_flag = (InfNorm(g)<=errorThres1) or (np.linalg.norm(currentError)*np.linalg.norm(currentError)<=errorThres3)
                    ratio = 0.0                    
                    if 1.0/3.0 > 1-(2*minus_resi)**3:
                        ratio = 1.0/3.0
                    else:
                        ratio = 1-(2*minus_resi)**3
                    u = u*ratio
                    v = 2
                else:
                    u = u*v
                    v = 2*v
            if minus_resi>0 or stop_flag:
                break
            
    return currentPara

def f(para):
    value = []
    for i in range(0,len(x_global)):
        value.append( para[0]*x_global[i]*x_global[i]*x_global[i]+para[1]*x_global[i]*x_global[i]+para[2]*x_global[i] + para[3]- y_global[i])
    return  np.array(value)
    
def Hession(para):
    hession = []
    for i in range(0,len(x_global)):
        oneline = []
        oneline.append(x_global[i]*x_global[i]*x_global[i])
        oneline.append(x_global[i]*x_global[i])
        oneline.append(x_global[i])
        oneline.append(1.0)
        hession.append(oneline)
    return np.mat(hession)
    
x = np.arange(0,1,0.05)  
y = [np.sin(2*np.pi*a) for a in x] 
noise = np.random.normal(0.0,0.1,len(x))
y = y+noise 

noise = np.random.normal(0.0,0.02,len(x))
x +=noise
fig = plt.figure()  
plot1 = fig.add_subplot(111)  
plt.ylabel('t')
plt.xlabel('x')
plot1.plot(x,y,'bo')

iniPara = []
for i in range(0,4):
    iniPara.append(0.0)   
iniPara = np.array(iniPara)

x_int = []
for i in range(0,len(x)):
    x_int.append(0.0)
    
x_int = np.array(x_int)
x_global = x
y_global = y
para = LM(f,Hession,iniPara,x_int)

print 2**3

print para

order = para.size -1
x_new = np.arange(0,1,0.01)
y_new = []
for index_x in range(0,len(x_new)):
    y_in_loop = 0.0
    for index_ele in range(0,order+1):
        ele_in_loop = 1.0
        for i in range(0,index_ele):
            ele_in_loop *= x_new[index_x]
        y_in_loop += ele_in_loop*para[order-index_ele]
    y_new.append(y_in_loop)
    
plot1.plot(x_new,y_new,color='g',linestyle='-',marker='')  
  
plt.show()



        
        