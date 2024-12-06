import pandas as pd
import pkg_resources

PATIENT_PARA_FILE = pkg_resources.resource_filename(
    'simglucose', 'params/vpatient_params.csv')
patient_params = pd.read_csv(PATIENT_PARA_FILE)
patient_names = list(patient_params['Name'].values)
for i, p in enumerate(patient_names):
    print('[{0}] {1}'.format(i + 1, p))
exit(0)