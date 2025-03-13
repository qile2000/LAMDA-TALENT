from TALENT.model.methods.base import Method

class DANetsMethod(Method):
    def __init__(self, args, is_regression):
        super().__init__(args, is_regression)
        assert(args.cat_policy != 'indices')

    def construct_model(self, model_config = None):
        from TALENT.model.models.danets import DANet
        if model_config is None:
            model_config = self.args.config['model']
        self.model = DANet(
            input_dim=self.d_in,
            num_classes=self.d_out,
            virtual_batch_size=self.args.config['general']['virtual_batch_size'],
            k=self.args.config['general']['k'],
            **model_config
            ).to(self.args.device)
        from TALENT.model.lib.danets.AcceleratedModule import AcceleratedCreator
        accelerated_module = AcceleratedCreator(self.d_in, base_out_dim=model_config["base_outdim"], k=self.args.config['general']['k'])
        self.model = accelerated_module(self.model)
        if self.args.use_float:
            self.model.float()
        else:
            self.model.double()