# Copyright (c) Alibaba, Inc. and its affiliates.
from modelscope.utils.constant import Tasks
import torch
# import sys
# sys.argv = ['GPT_eval_multi.py']
from modelscope.models.base import TorchModel
from modelscope.preprocessors.base import Preprocessor
from modelscope.pipelines.base import Model, Pipeline
from modelscope.utils.config import Config
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors.builder import PREPROCESSORS
from modelscope.models.builder import MODELS
import models.vqvae as vqvae
import models.t2m_trans as trans
import clip
import numpy as np
from utils.motion_process import recover_from_ric
import visualization.plot_3d_global as plot_3d
import options.option_transformer as option_trans
import os


@MODELS.register_module('text-driven_motion_generation', module_name='T2M-GPT')
class MyCustomModel(TorchModel):

    def __init__(self, model_dir, *args, **kwargs):
        super().__init__(model_dir, *args, **kwargs)
        self.net,self.trans = self.init_model(**kwargs)
        ckpt = torch.load(os.path.join(model_dir,'net_last.pth'), map_location='cpu')
        self.net.load_state_dict(ckpt['net'], strict=True)
        self.net.eval()
        self.net.cuda()

        ckpt = torch.load(os.path.join(model_dir,'net_best_fid.pth'), map_location='cpu')
        self.trans.load_state_dict(ckpt['trans'], strict=True)
        self.trans.eval()
        self.trans.cuda()

        self.clip, _ = clip.load("ViT-B/32", device=torch.device('cuda'), jit=False, download_root=model_dir)  # Must set jit=False for training
        clip.model.convert_weights(self.clip)  # Actually this line is unnecessary since clip by default already on float16
        self.clip.eval()
        for p in self.clip.parameters():
            p.requires_grad = False

        self.mean = torch.from_numpy(np.load(os.path.join(model_dir,'mean.npy'))).cuda()
        self.std = torch.from_numpy(np.load(os.path.join(model_dir,'std.npy'))).cuda()

    def forward(self, text, **forward_params):
        text = clip.tokenize(text, truncate=True).cuda()
        feat_clip_text = self.clip.encode_text(text).float()
        #from render_final import render
        index_motion = self.trans.sample(feat_clip_text[0:1],True)
        pred_pose = self.net.forward_decoder(index_motion)


        pred_xyz = recover_from_ric((pred_pose*self.std+self.mean).float(), 22)
        xyz = pred_xyz.reshape(1, -1, 22, 3)
        return xyz

    def init_model(self, **kwargs):
        """Provide default implementation based on TorchModel and user can reimplement it.
            include init model and load ckpt from the model_dir, maybe include preprocessor
            if nothing to do, then return lambda x: x
        """
        args = option_trans.get_args_parser()
        args.dataname = 't2m'
        args.down_t = 2
        args.depth = 3
        args.block_size = 51
        net = vqvae.HumanVQVAE(args, ## use args to define different parameters in different quantizers
                       args.nb_code,
                       args.code_dim,
                       args.output_emb_width,
                       args.down_t,
                       args.stride_t,
                       args.width,
                       args.depth,
                       args.dilation_growth_rate)


        trans_encoder = trans.Text2Motion_Transformer(num_vq=args.nb_code,
                                              embed_dim=1024,
                                              clip_dim=args.clip_dim,
                                              block_size=args.block_size,
                                              num_layers=9,
                                              n_head=16,
                                              drop_out_rate=args.drop_out_rate,
                                              fc_rate=args.ff_rate)
        return net,trans_encoder


@PREPROCESSORS.register_module('text-driven_motion_generation', module_name='T2M-GPT-preprocessor')
class MyCustomPreprocessor(Preprocessor):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #self.trainsforms = self.init_preprocessor(**kwargs)

    def __call__(self, results):
        return results

    # def init_preprocessor(self, **kwarg):
    #     """ Provide default implementation based on preprocess_cfg and user can reimplement it.
    #         if nothing to do, then return lambda x: x
    #     """
    #     return lambda x: x


@PIPELINES.register_module('text-driven_motion_generation', module_name='T2M-GPT-pipeline')
class MyCustomPipeline(Pipeline):
    """ Give simple introduction to this pipeline.

    Examples:

    >>> from modelscope.pipelines import pipeline
    >>> input = "Hello, ModelScope!"
    >>> my_pipeline = pipeline('my-task', 'my-model-id')
    >>> result = my_pipeline(input)

    """

    def __init__(self, model, preprocessor=MyCustomPreprocessor(), **kwargs):
        """
        use `model` and `preprocessor` to create a custom pipeline for prediction
        Args:
            model: model id on modelscope hub.
            preprocessor: the class of method be init_preprocessor
        """
        super().__init__(model=model, preprocessor=preprocessor,auto_collate=False)
        self.model = MyCustomModel(model)

    def _sanitize_parameters(self, **pipeline_parameters):
        """
        this method should sanitize the keyword args to preprocessor params,
        forward params and postprocess params on '__call__' or '_process_single' method
        considered to be a normal classmethod with default implementation / output

        Default Returns:
            Dict[str, str]:  preprocess_params = {}
            Dict[str, str]:  forward_params = {}
            Dict[str, str]:  postprocess_params = pipeline_parameters
        """
        return {}, pipeline_parameters, {}

    def _check_input(self, inputs):
        pass

    def _check_output(self, outputs):
        pass

    def forward(self, inputs, **forward_params):
        """ Provide default implementation using self.model and user can reimplement it
        """
        results = self.model(inputs)
        return results
    # def preprocess(self,inputs):
    #     return inputs

    def postprocess(self, inputs):
        """ If current pipeline support model reuse, common postprocess
            code should be write here.

        Args:
            inputs:  input data

        Return:
            dict of results:  a dict containing outputs of model, each
                output should have the standard output name.
        """
        return inputs


# Tips: usr_config_path is the temporary save configuration location， after upload modelscope hub, it is the model_id
# usr_config_path = './'
# config = Config({
#     "framework": 'pytorch',
#     "task": 'text-driven_motion_generation',
#     "model": {'type': 'T2M-GPT'},
#     "pipeline": {"type": "T2M-GPT-pipeline"}
# })
# config.dump('./' + 'configuration.json')

if __name__ == "__main__":
    from modelscope.models import Model
    from modelscope.pipelines import pipeline
    text = 'a man is kicking'
    outdir = './'
    inference = pipeline('text-driven_motion_generation', model='')
    output = inference(text)
    _ = plot_3d.draw_to_batch(output.detach().cpu().numpy(),text, [outdir+text+'.gif'])
