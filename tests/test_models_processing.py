import unittest
import torch
from src.models import UNetGenerator, PatchGANDiscriminator

class TestModels(unittest.TestCase):
    def test_unet_generator_output_shape(self):
        model = UNetGenerator()
        input_tensor = torch.randn(1, 1, 256, 256)
        output = model(input_tensor)
        self.assertEqual(output.shape, (1, 2, 256, 256))

    def test_patchgan_discriminator_output_shape(self):
        model = PatchGANDiscriminator()
        input_image = torch.randn(1, 1, 256, 256)
        target_image = torch.randn(1, 2, 256, 256)
        output = model(input_image, target_image)
        self.assertEqual(output.shape, (1, 1, 30, 30))  # Based on the final Conv2d parameters

if __name__ == '__main__':
    unittest.main()