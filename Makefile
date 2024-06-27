NUM_SIMULATIONS := 125

simulate_estimate_dcc_sgt_garch: dcc.py
	python3 simulate_estimate_dcc_sgt_garch.py --num_simulations $(NUM_SIMULATIONS) --num_sample 100 --dim 2
	# python3 simulate_dcc_sgt_garch.py --num_simulations $(NUM_SIMULATIONS) --num_sample 100 --dim 5
	#
	# python3 simulate_dcc_sgt_garch.py --num_simulations $(NUM_SIMULATIONS) --num_sample 300 --dim 2
	# python3 simulate_dcc_sgt_garch.py --num_simulations $(NUM_SIMULATIONS) --num_sample 300 --dim 5
	#
	# python3 simulate_dcc_sgt_garch.py --num_simulations $(NUM_SIMULATIONS) --num_sample 500 --dim 2
	# python3 simulate_dcc_sgt_garch.py --num_simulations $(NUM_SIMULATIONS) --num_sample 500 --dim 5

