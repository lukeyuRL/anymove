"""Offline, mock-backbone contracts for 672/192 Human Reader V2."""
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn as nn

from mocap.reader_hr_v2 import (
    HumanReaderHRV2, decode_hr, INPUT_SIZE, HEATMAP_SIZE, PATCH_GRID
)

class FakeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config=SimpleNamespace(hidden_size=32,num_register_tokens=2)
        self.param=nn.Parameter(torch.ones(1))
    def forward(self,pixel_values):
        b=pixel_values.shape[0]
        # Frozen visual encoder yields one token per 14px patch plus CLS/REG.
        t=torch.zeros(b,3+PATCH_GRID*PATCH_GRID,32,dtype=pixel_values.dtype,
                      device=pixel_values.device)
        return SimpleNamespace(last_hidden_state=t+self.param*0.00001)

class ReaderHRV2Contracts(unittest.TestCase):
    def test_dimensions(self):
        self.assertEqual(INPUT_SIZE,672)
        self.assertEqual(HEATMAP_SIZE,192)
        self.assertEqual(PATCH_GRID,48)
    def test_no_pretrained_network_for_unit_tests(self):
        with mock.patch("mocap.reader_hr_v2.AutoModel.from_pretrained",
                        return_value=FakeBackbone()) as load:
            reader=HumanReaderHRV2(2).cpu().train()
            load.assert_called_once()
            self.assertFalse(reader.backbone.training)
            self.assertTrue(all(not p.requires_grad for p in reader.backbone.parameters()))
            with torch.no_grad():
                output=reader(torch.zeros(1,3,INPUT_SIZE,INPUT_SIZE))
            self.assertEqual(tuple(output.shape),(1,2,HEATMAP_SIZE,HEATMAP_SIZE))
            with self.assertRaisesRegex(ValueError,"672"):
                reader(torch.zeros(1,3,448,448))
    def test_decode_heatmap_gaussian_peak(self):
        im=torch.full((1,2,HEATMAP_SIZE,HEATMAP_SIZE),-20.)
        for j,(x,y) in enumerate(((35,58),(178,101))):
            im[0,j,y,x]=8.
            im[0,j,y,x-1]=6.
            im[0,j,y,x+1]=6.
            im[0,j,y-1,x]=6.
            im[0,j,y+1,x]=6.
        xy,confidence=decode_hr(im)
        self.assertEqual(tuple(xy.shape),(1,2,2))
        torch.testing.assert_close(xy[0,0],torch.tensor((35.,58.))*3.5)
        torch.testing.assert_close(xy[0,1],torch.tensor((178.,101.))*3.5)
        torch.testing.assert_close(confidence,torch.tensor([[8.,8.]]))
    def test_decode_boundaries_are_finite(self):
        im=torch.zeros(1,1,HEATMAP_SIZE,HEATMAP_SIZE)
        im[0,0,0,0]=10.
        p,peak=decode_hr(im)
        self.assertTrue(torch.isfinite(p).all())
        self.assertTrue(torch.isfinite(peak).all())
        self.assertTrue((p>=0).all())
    def test_decoder_rejects_legacy_heatmap(self):
        with self.assertRaises(ValueError):
            decode_hr(torch.zeros(1,67,128,128))

if __name__=="__main__":
    unittest.main()
