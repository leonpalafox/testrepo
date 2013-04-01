#This is the main file for the IMM
import numpy as np
import pylab as plt
import IMMUtils
import Filter as fl
#Define probability mixin matrix (2 models)
p_ij=np.array([[0.95,0.05],[0.05,0.95]])
MixMat=np.atleast_2d([0.5,0.5])
r=2#Number of possible Filters
fillist=[]
fillist.append(fl.Kalman(2))
fillist.append(fl.Kalman(2))
fillist[0].A=1
fillist[1].A=2
obse=np.random.rand(100,2)
for j in range(100):
        print np.shape(MixMat)
        obs=obse[j,:]
        modelsaposte=np.array([])
        modelscov=np.array([])
        for fil in fillist:
                modelsaposte=np.append(modelsaposte,fil.mu_hat)
                modelscov=np.append(modelscov,fil.cov)
                fil.update(obs)
        mixing_mu,normal_c=IMMUtils.MixingProbs(p_ij,MixMat,r,0)
        MixingPrior_0=IMMUtils.MixingStep(modelsaposte,mixing_mu,r)
        Covariance_0=IMMUtils.MixingCov(mixing_mu,modelsaposte,modelscov,MixingPrior_0,r)
        Likeli=np.array([])
        for i in range(r):
                #print i
                Temp=IMMUtils.CalcLikeli(fillist[i].H,fillist[i].R,Covariance_0[i],MixingPrior_0[i],obs)
                Likeli=np.append(Likeli,Temp)#Need debuging the Cov
        MixMat=IMMUtils.UpdateMixMat(Likeli,normal_c)
        #print MixMat
        #Final update of the model
        modelsaposte=np.array([])
        modelscov=np.array([])
        for fil in fillist:
                modelsaposte=np.append(modelsaposte,fil.mu_hat)
                modelscov=np.append(modelscov,fil.cov)
        Estimates=IMMUtils.CalculateEstim(modelsaposte,MixMat,r)#Review this covariance
        Covarri=IMMUtils.CalculateCovari(modelsaposte,MixMat,Estimates,modelscov,r)
        MixMat=MixMat[None,:]
        print np.shape(MixMat)
