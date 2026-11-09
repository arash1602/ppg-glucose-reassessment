import numpy as np
import pandas as pd
import tissue_model.tissue_params as params
import tissue_model.layer_calc as calc

rng = np.random.default_rng(45)

epi_mu_a_sim_set = []
derm_mu_a_sim_set = []
Jonasson_derm_mu_a_sim_set = []
sub_mu_a_sim_set = []
realizations = 200

for i in range(realizations):
    derm_blood = rng.uniform(params.derm_lower_bound["blood"], params.derm_upper_bound["blood"])
    derm_oxygen_saturation = rng.uniform(params.derm_lower_bound["oxygen_saturation"], params.derm_upper_bound["oxygen_saturation"])
    derm_water = rng.uniform(params.derm_lower_bound["water"], params.derm_upper_bound["water"])
    if(1 - (derm_blood + derm_water) > 0):
        derm_mu_a_sim_set.append(
            calc.derm(
                B = derm_blood, 
                S = derm_oxygen_saturation, 
                W = derm_water, 
                F = 0, 
                M = 0,
                baseline = params.baseline_derm,
        )[0]
        )

for i in range(realizations):
    derm_blood = rng.uniform(params.derm_lower_bound["blood"], params.derm_upper_bound["blood"])
    derm_oxygen_saturation = rng.uniform(params.derm_lower_bound["oxygen_saturation"], params.derm_upper_bound["oxygen_saturation"])
    Jonasson_derm_mu_a_sim_set.append(
        calc.derm(
            B = derm_blood, 
            S = derm_oxygen_saturation, 
            W = 0, 
            F = 0, 
            M = 0,
            baseline = 0,
                
        )[0]
        )

for i in range(realizations):
    sub_blood = rng.uniform(params.sub_lower_bound["blood"], params.sub_upper_bound["blood"])
    sub_oxygen_saturation = rng.uniform(params.sub_lower_bound["oxygen_saturation"], params.sub_upper_bound["oxygen_saturation"])
    sub_water = rng.uniform(params.sub_lower_bound["water"], params.sub_upper_bound["water"])
    sub_fat = rng.uniform(params.sub_lower_bound["fat"], params.sub_upper_bound["fat"])
    if(1 - (sub_blood + sub_water + sub_fat) > 0): # Remove samples with a negative baseline fraction
        sub_mu_a_sim_set.append(
            calc.sub(
                B = sub_blood, 
                S = sub_oxygen_saturation, 
                W = sub_water, 
                F = sub_fat, 
                M = 0,
                baseline = 0
        )[0]
        )

epi_mu_a_sim_df = []
derm_mu_a_sim_df = pd.concat(derm_mu_a_sim_set, axis = 1)
Jonasson_derm_mu_a_sim_df = pd.concat(Jonasson_derm_mu_a_sim_set, axis = 1)
sub_mu_a_sim_df = pd.concat(sub_mu_a_sim_set, axis = 1)