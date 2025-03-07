from TALENT.model.methods.base import Method

class SaintMethod(Method):
    def __init__(self, args, is_regression):
        super().__init__(args, is_regression)
        assert(args.cat_policy == 'indices')

    def construct_model(self, model_config = None):
        if model_config is None:
            model_config = self.args.config['model']
        from TALENT.model.models.saint import SAINT
        self.model = SAINT(
                categories=self.categories,
                num_continuous=self.d_in,
                y_dim=self.d_out,
                **model_config
                ).to(self.args.device)
        if self.args.use_float:
            self.model.float()
        else:
            self.model.double()