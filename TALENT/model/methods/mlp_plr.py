from TALENT.model.methods.base import Method

class MLP_PLRMethod(Method):
    def __init__(self, args, is_regression):
        super().__init__(args, is_regression)
        assert(args.cat_policy == 'tabr_ohe')

    def construct_model(self, model_config = None):
        from TALENT.model.models.mlp_plr import MLP
        if model_config is None:
            model_config = self.args.config['model']
        self.model = MLP(
                d_in=(self.d_in + len(self.categories)) if self.categories is not None else self.d_in,
                d_num=self.d_in,
                d_out=self.d_out,
                **model_config
                ).to(self.args.device) 
        if self.args.use_float:
            self.model.float()
        else:
            self.model.double()