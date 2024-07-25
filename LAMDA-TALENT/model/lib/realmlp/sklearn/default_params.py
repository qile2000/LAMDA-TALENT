class DefaultParams:
    RealMLP_TD_CLASS = dict(
        hidden_sizes=[256] * 3,
        max_one_hot_cat_size=9, embedding_size=8,
        weight_param='ntk', bias_lr_factor=0.1,
        act='selu', use_parametric_act=True, act_lr_factor=0.1,
        block_str='w-b-a-d', p_drop=0.15, p_drop_sched='flat_cos',
        add_front_scale=True,
        scale_lr_factor=6.0,
        bias_init_mode='he+5', weight_init_mode='std',
        wd=2e-2, wd_sched='flat_cos', bias_wd_factor=0.0,
        use_ls=True, ls_eps=0.1,
        num_emb_type='pbld', plr_sigma=0.1, plr_hidden_1=16, plr_hidden_2=4, plr_lr_factor=0.1,
        lr=4e-2,
        tfms=['one_hot', 'median_center', 'robust_scale', 'smooth_clip', 'embedding'],
        n_epochs=256, lr_sched='coslog4', opt='adam', sq_mom=0.95
    )

    RealMLP_TD_S_CLASS = dict(
        hidden_sizes=[256] * 3,
        weight_param='ntk', bias_lr_factor=0.1,
        act='selu',
        block_str='w-b-a',
        add_front_scale=True, scale_lr_factor=6.0,
        bias_init_mode='normal', weight_init_mode='normal',
        last_layer_config=dict(bias_init_mode='zeros', weight_init_mode='zeros'),
        use_ls=True, ls_eps=0.1,
        tfms=['one_hot', 'median_center', 'robust_scale', 'smooth_clip'],
        n_epochs=256, lr=4e-2, lr_sched='coslog4', opt='adam', sq_mom=0.95
    )

    RealMLP_TD_REG = dict(
        hidden_sizes=[256] * 3,
        max_one_hot_cat_size=9, embedding_size=8,
        weight_param='ntk', weight_init_mode='std',
        bias_init_mode='he+5', bias_lr_factor=0.1,
        act='mish', use_parametric_act=True, act_lr_factor=0.1,
        wd=2e-2, wd_sched='flat_cos', bias_wd_factor=0.0,
        block_str='w-b-a-d', p_drop=0.15, p_drop_sched='flat_cos',
        add_front_scale=True, scale_lr_factor=6.0,
        tfms=['one_hot', 'median_center', 'robust_scale', 'smooth_clip', 'embedding'],
        num_emb_type='pbld', plr_sigma=0.1, plr_hidden_1=16, plr_hidden_2=4, plr_lr_factor=0.1,
        clamp_output=True, normalize_output=True,
        lr=0.2,
        n_epochs=256, lr_sched='coslog4', opt='adam', sq_mom=0.95
    )

    RealMLP_TD_S_REG = dict(
        hidden_sizes=[256] * 3,
        weight_param='ntk', bias_lr_factor=0.1,
        bias_init_mode='normal', weight_init_mode='normal',
        last_layer_config=dict(bias_init_mode='zeros', weight_init_mode='zeros'),
        act='mish', normalize_output=True,
        block_str='w-b-a',
        add_front_scale=True, scale_lr_factor=6.0,
        tfms=['one_hot', 'median_center', 'robust_scale', 'smooth_clip'],
        n_epochs=256, lr=7e-2, lr_sched='coslog4', opt='adam', sq_mom=0.95
    )

    # -------- GBDTs ------------

    LGBM_TD_CLASS = dict(
        n_estimators=1000, lr=4e-2, subsample=0.75, colsample_bytree=1.0, num_leaves=50, bagging_freq=1,
        min_data_in_leaf=40, min_sum_hessian_in_leaf=1e-7, max_bin=255, early_stopping_rounds=300,
    )

    LGBM_TD_REG = dict(
        n_estimators=1000, lr=5e-2, subsample=0.7, colsample_bytree=1.0, num_leaves=100, max_bin=255, bagging_freq=1,
        min_data_in_leaf=3, min_sum_hessian_in_leaf=1e-7, early_stopping_rounds=300,
    )

    XGB_TD_CLASS = dict(
        n_estimators=1000, lr=8e-2, min_child_weight=5e-6, reg_lambda=0.0, max_depth=6,
        colsample_bylevel=0.9, subsample=0.65, tree_method='hist', max_bin=256, early_stopping_rounds=300,
    )

    XGB_TD_REG = dict(
        n_estimators=1000, max_depth=9, tree_method='hist', max_bin=256, lr=5e-2, min_child_weight=2.0, reg_lambda=0.0,
        subsample=0.7, early_stopping_rounds=300,
    )

    # from Probst, Boulestix, and Bischl, "Tunability: Importance of ..."
    XGB_PBB_CLASS = dict(
        n_estimators=4168, lr=0.018, min_child_weight=2.06,
        max_depth=13, reg_lambda=0.982, reg_alpha=1.113, subsample=0.839,
        colsample_bytree=0.752, colsample_bylevel=0.585,
        tree_method='hist', max_n_threads=64,
        tfms=['one_hot'], max_one_hot_cat_size=20
    )

    CB_TD_CLASS = dict(
        n_estimators=1000, lr=8e-2, l2_leaf_reg=1e-5, boosting_type='Plain',
        bootstrap_type='Bernoulli', subsample=0.9,
        max_depth=7, random_strength=0.8, one_hot_max_size=15,
        leaf_estimation_iterations=1, max_bin=254, early_stopping_rounds=300,
    )

    CB_TD_REG = dict(
        n_estimators=1000, lr=9e-2, l2_leaf_reg=1e-5, boosting_type='Plain',
        bootstrap_type='Bernoulli', subsample=0.9,
        max_depth=9, random_strength=0.0, max_bin=254,
        one_hot_max_size=20, leaf_estimation_iterations=20, early_stopping_rounds=300,
    )

    # RTDL params

    RESNET_RTDL_D_CLASS = {
        "lr_scheduler": False,
        "module__activation": "reglu",
        "module__normalization": "batchnorm",
        "module__n_layers": 8,
        "module__d": 256,
        "module__d_hidden_factor": 2,
        "module__hidden_dropout": 0.2,
        "module__residual_dropout": 0.2,
        "lr": 1e-3,
        "optimizer__weight_decay": 1e-7,
        "optimizer": "adamw",
        "module__d_embedding": 128,
        "batch_size": 256,
        "max_epochs": 300,
        "use_checkpoints": True,
        "es_patience": 40,
        "lr_patience": 30,
        "verbose": 0,
        'tfms': ['quantile'],
    }

    RESNET_RTDL_D_REG = {**RESNET_RTDL_D_CLASS,
                         "transformed_target": True}

    MLP_RTDL_D_CLASS = {
        "lr_scheduler": False,
        "module__n_layers": 8,
        "module__d_layers": 256,
        "module__d_first_layer": 128,
        "module__d_last_layer": 128,
        "module__dropout": 0.2,
        "lr": 1e-3,
        "optimizer": "adamw",
        "module__d_embedding": 128,
        "batch_size": 256,
        "max_epochs": 300,
        "use_checkpoints": True,
        "es_patience": 40,
        "lr_patience": 30,
        "verbose": 0,
        'tfms': ['quantile'],
    }

    MLP_RTDL_D_REG = {**MLP_RTDL_D_CLASS,
                      "transformed_target": True}
    
    # Default parameters for rtdl models based on https://github.com/naszilla/tabzilla/blob/main/TabZilla/models/rtdl.py
    RESNET_RTDL_D_CLASS_TabZilla = {
        "lr_scheduler": False,
        "module__activation": "relu",
        "module__normalization": "batchnorm",
        "module__n_layers": 2,
        "module__d": 128,
        "module__d_first_layer": 128,
        "module__d_last_layer": 128,
        "module__d_hidden_factor": 2,
        "module__hidden_dropout": 0.25, #DROPOUT_FIRST
        "module__residual_dropout": 0.1, #DROPOUT_SECOND
        "lr": 1e-3,
        "optimizer__weight_decay": 0.01, # for tabzilla they don't set it which means 0.01 (which seems high compared
                                        # to rtdl hp space?)
        "optimizer": "adamw",
        "module__d_embedding": 8,
        "batch_size": 128, # default param in https://github.com/naszilla/tabzilla/blob/4949a1dea3255c1a794d89aa2422ef1f8c9ae265/README.md?plain=1#L129
        "max_epochs": 1000, #same
        "use_checkpoints": True,
        "es_patience": 20, #same
        "lr_patience": 30,
        "verbose": 0,
        'tfms': ['quantile_tabr'],
    }

    RESNET_RTDL_D_REG_TabZilla = {**RESNET_RTDL_D_CLASS_TabZilla,
                         "transformed_target": True}

    MLP_RTDL_D_CLASS_TabZilla = {
        "lr_scheduler": False,
        "module__n_layers": 3,
        "module__d_first_layer": 128,  # ignored by the code since d_layers is a list
        "module__d_last_layer": 128,  # ignored by the code since d_layers is a list
        "module__d_layers": [128, 256, 128],
        "module__dropout": 0.1,
        "lr": 1e-3,
        "optimizer": "adamw",
        "module__d_embedding": 8,
        "batch_size": 128, # default param in https://github.com/naszilla/tabzilla/blob/4949a1dea3255c1a794d89aa2422ef1f8c9ae265/README.md?plain=1#L129
        "max_epochs": 1000, #same
        "use_checkpoints": True,
        "es_patience": 20, #same
        "lr_patience": 30,
        "verbose": 0,
        'tfms': ['quantile_tabr'],
    }

    MLP_RTDL_D_REG_TabZilla = {**MLP_RTDL_D_CLASS_TabZilla,
                    "transformed_target": True}
    
    TABR_S_D_CLASS = {
        "num_embeddings": None,
        "d_main": 265,
        "context_dropout": 0.38920071545944357, #named mixer_dropout sometimes I think
        "d_multiplier": 2.0,
        "encoder_n_blocks": 0,
        "predictor_n_blocks": 1,
        "mixer_normalization": "auto",
        "dropout0": 0.38852797479169876,
        "dropout1": 0.0,
        "normalization": "LayerNorm",
        "activation": "ReLU",
        "batch_size": "auto", # adapt given the dataset size
        "eval_batch_size": 4096, #TODO: automatically infer given memory
        "patience": 16,
        "n_epochs": 100_000, #inf in paper
        "context_size": 96,
        "freeze_contexts_after_n_epochs": None,
        "optimizer": {
            "type": "AdamW",
            "lr": 0.0003121273641315169,
            "weight_decay": 1.2260352006404615e-06
        },
        'tfms': ['quantile_tabr'],
    }
    
    TABR_S_D_REG = {**TABR_S_D_CLASS,
                    "transformed_target": True}
    
    TABR_S_D_CLASS_FREEZE = {
        **TABR_S_D_CLASS,
        "freeze_contexts_after_n_epochs": 4,
    }

    TABR_S_D_REG_FREEZE = {
        **TABR_S_D_REG,
        "freeze_contexts_after_n_epochs": 4,
    }


    # ----- sklearn versions ------

    LGBM_D = dict(
        n_estimators=100,
    )

    XGB_D = dict(
        n_estimators=100, tree_method='hist',
    )

    CB_D = dict(
        n_estimators=1000,
    )

    RF_SKL_D = dict(
        tfms=['ordinal_encoding'], permute_ordinal_encoding=True,
    )

    MLP_SKL_D = dict(
        tfms=['mean_center', 'l2_normalize', 'one_hot']
    )
