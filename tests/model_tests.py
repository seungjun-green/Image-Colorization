import sys
sys.path.append("/Users/seungjunlee/Desktop/MLProjects/Image Colorization/")
from models.models import *
import torch

# test generator
def test_generator():
    sample_input = torch.randn(1, 1, 256, 256)
    
    gen1 = UNetGenerator()
    gen2 = ResNetUNetGenerator()
    gen3 = AttentionUNetGenerator()
    
    sample_output1 = gen1(sample_input)
    sample_output2 = gen2(sample_input)
    sample_output3 = gen3(sample_input)
    
    expected_shape = (1, 2, 256, 256)
    
    # check whether all has shape of (N, 2, 256, 256)
    assert sample_output1.shape == expected_shape, f"gen1 output shape {sample_output1.shape} does not match expected {expected_shape}"
    assert sample_output2.shape == expected_shape, f"gen2 output shape {sample_output2.shape} does not match expected {expected_shape}"
    assert sample_output2.shape == expected_shape, f"gen3 output shape {sample_output3.shape} does not match expected {expected_shape}"
    
    
    print("All generators produce outputs with the expected shape:", expected_shape)
    

if __name__ == "__main__":
    test_generator()
