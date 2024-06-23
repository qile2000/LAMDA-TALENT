from model.methods.base import Method

class ResNetMethod(Method):
    def __init__(self, args, is_regression):
        super().__init__(args, is_regression)
        assert(args.cat_policy != 'indices')

    def construct_model(self, model_config = None):
        from model.models.resnet import ResNet
        if model_config is None:
            model_config = self.args.config['model']
        self.model = ResNet(
            d_in=self.d_in,
            d_out=self.d_out,
            **model_config
        ).to(self.args.device)
        self.model.double()