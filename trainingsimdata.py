from LSPI_GBM_analysis import training_sim_data, scoring_sim_data

training_sim_data(expiry=1, num_steps=5, 
num_paths=5, spot_price=10, 
spot_price_frac=0.8, rate=0.6, vol=0.2)


scoring_sim_data(expiry=1, num_steps=5, num_paths=5, spot_price=10, 
rate=0.2, vol=0.6)


