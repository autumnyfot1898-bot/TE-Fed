import unittest
from copy import deepcopy

try:
    import torch
    from tefed.model import RoutedYOLO,sample_tensor
    from tefed.losses import WeightedNativeLoss,soft_consistency
    from tefed.engine import train_local
    AVAILABLE=True
except ImportError:
    AVAILABLE=False


@unittest.skipUnless(AVAILABLE,'Requires native training dependencies')
class NativeTrainingTests(unittest.TestCase):
    def test_paper_backbone_three_routed_heads_and_inactive_head(self):
        torch.set_num_threads(2);torch.manual_seed(0)
        model=RoutedYOLO({'model':{'architecture':'yolo26s.yaml','classes':2,'specialist_width':.75}})
        unused=deepcopy(model.heads[5].state_dict())
        x=torch.rand(1,5,64,64)
        targets=[{'box':[.1,.2,.4,.7],'class_id':0},{'box':[.55,.3,.9,.8],'class_id':1}]
        outputs=model(x,(0,1,2))
        loss=sum(WeightedNativeLoss(model,m)(outputs[m],targets,[w,w],'cpu')[0] for m,w in [(0,.5),(1,.3),(2,.2)])
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(any(p.grad is not None for p in model.trunk.parameters()))
        self.assertTrue(all(p.grad is None for p in model.heads[5].parameters()))
        for key,value in model.heads[5].state_dict().items():
            self.assertTrue(torch.equal(value,unused[key]))

    def test_empty_labeled_frame_retains_background_loss(self):
        torch.set_num_threads(2)
        model=RoutedYOLO({'model':{'architecture':'yolo26n.yaml','classes':2,'specialists':0}})
        output=model(torch.rand(1,5,64,64),(0,))[0]
        loss,_=WeightedNativeLoss(model,0)(output,[],[],'cpu')
        self.assertGreater(float(loss),0)

    def test_fixed_depth_normalizer_and_nir_intensity(self):
        import numpy as np
        sample={'rgb':np.zeros((4,4,3),np.uint8),'depth':np.full((4,4),2000,np.float32),
                'nir':np.full((4,4),128,np.uint8),'depth_scale':10000}
        x=sample_tensor(sample)
        self.assertAlmostEqual(float(x[0,3,0,0]),.2,places=6)
        self.assertAlmostEqual(float(x[0,4,0,0]),128/255,places=6)

    def test_consistency_uses_detached_teacher_and_finite_student_gradients(self):
        weak={'logits':torch.tensor([[[2.,0.]]],requires_grad=True),
              'boxes':torch.tensor([[[.1,.1,.8,.8]]],requires_grad=True),'scores':torch.tensor([[.8]])}
        strong={'logits':torch.tensor([[[1.,0.]]],requires_grad=True),
                'boxes':torch.tensor([[[.12,.1,.8,.8]]],requires_grad=True),'scores':torch.tensor([[.7]])}
        loss=soft_consistency(weak,strong)
        loss.backward()
        self.assertIsNone(weak['logits'].grad)
        self.assertTrue(torch.isfinite(strong['boxes'].grad).all())


if __name__=='__main__':
    unittest.main()
