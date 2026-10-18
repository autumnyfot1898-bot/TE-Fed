import unittest

from tefed.diagnostics import renderer_target_observation


class RendererAgreementTests(unittest.TestCase):
    def test_missing_responses_are_failures_not_missing_events(self):
        target={'box':[.2,.2,.6,.6],'class_id':0}
        result=renderer_target_observation({},[0,1,2],[.5,.3,.2],target,scope='test')
        self.assertFalse(result['agreement'])
        self.assertIsNone(result['predicted_class'])

    def test_class_mismatch_is_not_filtered_during_association(self):
        target={'box':[.2,.2,.6,.6],'class_id':0}
        outputs={'0':[{'box':target['box'],'posterior':[.1,.9],'class_id':1}]}
        result=renderer_target_observation(outputs,[0,1,2],[.5,.3,.2],target,scope='test')
        self.assertEqual(result['predicted_class'],1)
        self.assertFalse(result['agreement'])
        outputs['0'][0]['posterior']=[.9,.1]
        outputs['0'][0]['class_id']=0
        self.assertTrue(renderer_target_observation(outputs,[0,1,2],[.5,.3,.2],target,scope='test')['agreement'])
