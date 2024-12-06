import simglucose.patient.t1dpatient
from simglucose.patient.t1dpatient import T1DPatient
import numpy as np

patient = ['adolescent#001', 'adolescent#002', 'adolescent#003', 'adolescent#004', 'adolescent#005', 'adolescent#006',
           'adolescent#007', 'adolescent#008', 'adolescent#009', 'adolescent#010',
           'adult#001', 'adult#002', 'adult#003', 'adult#004', 'adult#005', 'adult#006', 'adult#007', 'adult#008',
           'adult#009', 'adult#010',
           'child#001', 'child#002', 'child#003', 'child#004', 'child#005', 'child#006', 'child#007', 'child#008',
           'child#009', 'child#010']
for idx in range(30):
    p = T1DPatient.withName(patient[idx])
    #basal = p._params.u2ss * p._params.BW/6000
    if(p._params.BW>=90):
        kcal=p._params.BW*3+70
    # elif(p._params.BW>70):
    #     kcal=p._params.BW*3.5+70
    elif(p._params.BW>60):
        kcal=p._params.BW*3.5+70
    else:
        kcal =p._params.BW *4+70
    if 20 > idx >= 10:
        cho=[round(kcal*0.25),round(kcal*0.3),round(kcal*0.25),round(kcal*0.2)]
    else:
        cho=[round(kcal*0.2),round(kcal*0.1),round(kcal*0.25),round(kcal*0.1),round(kcal*0.2),round(kcal*0.15)]
    print(p.name,round(p._params.BW),cho)

