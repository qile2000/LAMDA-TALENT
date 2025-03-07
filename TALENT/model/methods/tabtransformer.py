from TALENT.model.methods.base import Method
import torch

class TabTransformerMethod(Method):
    def __init__(self, args, is_regression):
        super().__init__(args, is_regression)
        assert(args.cat_policy == 'indices')

    def construct_model(self, model_config = None):
        if model_config is None:
            model_config = self.args.config['model']
        from TALENT.model.models.tabtransformer import TabTransformerModel
        self.model = TabTransformerModel(
            categories=self.categories,
            num_continuous=self.d_in,
            dim_out=self.d_out,
            mlp_act=torch.nn.ReLU(),
            mlp_hidden_mults=(4, 2),
            **model_config
        ).to(self.args.device)
        if self.args.use_float:
            self.model.float()
        else:
            self.model.double()