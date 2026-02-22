import sys
sys.path.append("../src")
from acoupipe_extensions import VariableArrayConfig, VogelHansen

from acoupipe.datasets.synthetic import DatasetSynthetic
import numpy as np

def main():
    # --- setup data set configuration ---
    test_generator = np.random.default_rng(seed=200)
    config = VariableArrayConfig(mpos_fn=VogelHansen, mode="analytic", mic_sig_noise=False, generator=test_generator, min_nsources=1, max_nsources=4, min_num_mics=32, max_num_mics=64)

    d1 = DatasetSynthetic(config=config)

    # --- generate and save data ---
    d1.save_h5(features=["csm", "cartesian_coordinates", "eigmode", "loc", 'source_strength_analytic' ], f=2500, num=0, split="training", size=62500, name="step_2_3.h5")
    
if __name__ == "__main__": 
    main()