import fvcb
import stomatal
import util
import pandas as pd
import torch
import matplotlib.pyplot as plt

# util.selftest() # Check if all models are working

# FvCB model fitting
dftest = pd.read_csv('data/dfMAGIC043_lr.csv')
lcd = fvcb.initLicordata(dftest, preprocess=True, lightresp_id = [118])
fvcbm = fvcb.model(lcd, LightResp_type = 2, TempResp_type = 0, onefit = False)
fitresult = fvcb.fit(fvcbm, learn_rate= 0.06, maxiteration = 20000, minloss= 1, fitcorr=False) # If temp type is 0, do not set fitcorr to True
fvcbm = fitresult.model
fvcbm.eval()
A_fit, Ac_fit, Aj_fit, Ap_fit = fvcbm()

#plot all the data based on the ID
plt.figure()
for id in lcd.IDs:
    if id == 118:
        continue
    indices_id = lcd.getIndicesbyID(id)
    A_id = A_fit[indices_id]
    # Ac_id = Ac_fit[indices_id]
    # Aj_id = Aj_fit[indices_id]
    # Ap_id = Ap_fit[indices_id]
    plt.plot(lcd.Ci[indices_id],A_id.detach().numpy())
    plt.plot(lcd.Ci[indices_id],lcd.A[indices_id],'.')
plt.title('Fitted A/Ci curves')
plt.show()
#
plt.figure()
indices_id = lcd.getIndicesbyID(118)
# A_id = A_fit[indices_id]
Ac_id = Ac_fit[indices_id]
Aj_id = Aj_fit[indices_id]
plt.plot(lcd.Q[indices_id],Ac_id.detach().numpy())
plt.plot(lcd.Q[indices_id],Aj_id.detach().numpy())
plt.plot(lcd.Q[indices_id],lcd.A[indices_id],'.')
plt.title('Fitted Light response A/Q curve for ID 118')
plt.show()

# Stomatal model fitting
datasc = pd.read_csv('data/steadystate_stomatalconductance.csv')
scd = stomatal.initscdata(datasc)
scm = stomatal.BMF(scd)
fitresult = stomatal.fit(scm, learnrate = 0.5, maxiteration =20000)
scm = fitresult.model
gsw = scm()
gsw_mea = scd.gsw

plt.figure()
plt.plot(gsw_mea)
plt.plot(gsw.detach().numpy(), '.')
plt.title('Fitted gsw')
plt.legend(['Measured gsw', 'Fitted gsw'])
plt.show()
