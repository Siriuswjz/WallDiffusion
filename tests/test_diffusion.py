"""CPU checks for diffusion algebra, conditioning, padding, and gradients."""
import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch
from src.diffusion import ConditionalUNet, GaussianDiffusion
from src.diffusion.data import make_condition, grid_condition
from predict_diffusion import external_condition


class DiffusionChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_noise_equation_and_oracle_sampling(self):
        diffusion = GaussianDiffusion(20)
        clean = torch.randn(1, 1, 17, 19)
        noise = torch.randn_like(clean)
        timestep = torch.tensor([7])
        noisy = diffusion.add_noise(clean, timestep, noise)
        alpha = diffusion.alpha_bar[7]
        self.assertTrue(torch.allclose((noisy-alpha.sqrt()*clean)/(1-alpha).sqrt(), noise, atol=1e-6))
        def oracle(x, t, condition):
            a = diffusion.alpha_bar[t][:, None, None, None]
            return (x-a.sqrt()*clean)/(1-a).sqrt()
        generated = diffusion.sample(oracle, torch.zeros(1, 4, 17, 19), steps=10)
        self.assertTrue(torch.allclose(generated, clean, atol=1e-5))

    def test_joint_channels_backward_and_sampling(self):
        model=ConditionalUNet(base_channels=8,attention_depth=1,heads=2,
                              target_channels=2,condition_channels=2)
        diffusion=GaussianDiffusion(20,'v')
        clean=torch.randn(2,2,35,47); condition=torch.randn(2,2,35,47)
        loss=diffusion.loss(model,clean,condition);loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(model.output[-1].weight.grad.shape[0],2)
        result=diffusion.sample(model.eval(),condition,steps=3)
        self.assertEqual(result.shape,clean.shape)
        self.assertTrue(torch.isfinite(result).all())
        with self.assertRaises(ValueError):
            diffusion.sample(model,condition,steps=3,enforce_observations=True)

    def test_model_backward_timestep_padding_and_reproducibility(self):
        torch.manual_seed(42)
        model = ConditionalUNet(base_channels=8, attention_depth=1, heads=2)
        diffusion = GaussianDiffusion(10)
        clean = torch.randn(2, 1, 65, 47)
        condition = torch.randn(2, 4, 65, 47)
        condition[:, 2:] = 0
        loss = diffusion.loss(model, clean, condition)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(model.time_mlp[0].weight.grad.abs().sum().item(), 0)
        model.zero_grad(set_to_none=True)
        v_loss = GaussianDiffusion(10, prediction_type='v').loss(model, clean, condition)
        v_loss.backward()
        self.assertTrue(torch.isfinite(v_loss))
        self.assertGreater(model.time_mlp[0].weight.grad.abs().sum().item(), 0)
        model.eval()
        output = diffusion.sample(model, condition, steps=3, seed=42)
        again = diffusion.sample(model, condition, steps=3, seed=42)
        self.assertEqual(output.shape, clean.shape)
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.equal(output, again))
        self.assertFalse(torch.equal(output, diffusion.sample(model, condition, steps=3, seed=43)))
        condition[:, 2, 5, 7] = 1
        condition[:, 3, 5, 7] = 2.5
        projected = diffusion.sample(model, condition, steps=3, enforce_observations=True)
        self.assertTrue(torch.equal(projected[:, 0, 5, 7], torch.full((2,), 2.5)))

    def test_mask_and_external_normalization_without_truth(self):
        quantity = torch.ones(1, 10, 10)
        condition = make_condition(torch.zeros(2, 10, 10), quantity, .1, torch.Generator().manual_seed(42))
        self.assertEqual(condition[2].sum().item(), 10)
        self.assertTrue(torch.equal(condition[2], condition[3]))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'observations.npz'
            mask = np.zeros((10, 10), dtype=np.float32)
            mask[2, 3] = 1
            observed = np.full_like(mask, np.nan); observed[2, 3] = 14
            np.savez(path, u=np.full_like(mask, 3), w=np.full_like(mask, 6), mask=mask, observations=observed)
            result = external_condition(path, {'mean': [1, 2, 10], 'std': [2, 4, 2]})
            self.assertTrue(torch.isfinite(result).all())
            self.assertEqual(result[3, 2, 3].item(), 2)
            self.assertEqual(result[3].sum().item(), 2)

    def test_grid_matches_global_crop_and_covers_boundary(self):
        condition = grid_condition(torch.zeros(2, 1400, 800), torch.ones(1, 1400, 800), 10)
        self.assertEqual(condition[2].sum().item(), 141 * 81)
        self.assertEqual(condition[2, -1, -1].item(), 1)
        for origin in [(192, 192), (1144, 544)]:
            h, w = origin
            patch = grid_condition(torch.zeros(2,256,256), torch.ones(1,256,256), 10, origin)
            self.assertTrue(torch.equal(patch[2], condition[2,h:h+256,w:w+256]))

    def test_zero_density_condition_contains_no_wall_target_information(self):
        velocity = torch.randn(2,32,32)
        first = make_condition(velocity, torch.randn(1,32,32), 0)
        second = make_condition(velocity, torch.randn(1,32,32)*100+30, 0)
        self.assertTrue(torch.equal(first,second))
        self.assertEqual(first[2:].abs().sum().item(),0)
        self.assertTrue(torch.equal(first[:2],velocity))

    def test_v_oracle_sampling_and_terminal_stability(self):
        diffusion = GaussianDiffusion(1000, prediction_type='v')
        clean = torch.randn(1,1,17,19)
        def oracle(x,t,condition):
            a = diffusion.alpha_bar[t][:,None,None,None]
            epsilon = (x-a.sqrt()*clean)/(1-a).sqrt()
            return a.sqrt()*epsilon-(1-a).sqrt()*clean
        result = diffusion.sample(oracle, torch.zeros(1,4,17,19), steps=50)
        self.assertTrue(torch.allclose(result,clean,atol=1e-5))
        # An untrained zero-output predictor must no longer blow up by 1/sqrt(alpha_T).
        result = diffusion.sample(lambda x,t,c: torch.zeros_like(x), torch.zeros(1,4,17,19), steps=50)
        self.assertTrue(torch.isfinite(result).all())
        self.assertLess(result.std().item(),2.)


if __name__ == '__main__':
    unittest.main()
