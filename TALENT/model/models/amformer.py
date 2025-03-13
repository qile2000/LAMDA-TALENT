import torch
import torch.nn.functional as F
from torch import nn, einsum
from TALENT.model.lib.amformer.blocks import NumericalEmbedder, Transformer
from einops import rearrange, repeat


# adapted from https://github.com/aigc-apps/AMFormer

# main class

class AMFormer(nn.Module):
    def __init__(
        self,
        # args,
        num_cont,
        num_cate,
        categories,
        out,
        dim,
        depth,
        attn_dropout,
        ff_dropout,
        groups,
        sum_num_per_group,
        prod_num_per_group,
        cluster,
        target_mode,
        token_descent,
        num_special_tokens=2,
        qk_relu=False,
        heads=8,
        use_prod=True,
        use_cls_token=True
    ):
        super().__init__()
        '''
        dim: token dim
        depth: Attention block numbers
        heads: heads in multi-head attn
        attn_dropout: dropout in attn
        ff_dropout: drop in ff in attn
        use_cls_token: use cls token in FT-transformer but autoint it should be False
        groups: used in Memory block --> how many cluster prompts
        sum_num_per_group: used in Memory block --> topk to sum in each sum cluster prompts
        prod_num_per_group: used in Memory block --> topk to sum in each prod cluster prompts
        cluster: if True, prompt --> q, False, x --> q
        target_mode: if None, prompt --> q, if mix, [prompt, x] --> q
        token_num: how many token in the input x
        token_descent: use in MUCH-TOKEN dataset
        use_prod: use prod block
        num_special_tokens: =2
        categories: how many different cate in each cate ol
        out: =1 if regressioin else =cls number
        self.num_cont: how many cont col
        num_cont = args.num_cont
        num_cate: how many cate col
        '''
        
        self.use_cls_token = use_cls_token
        self.out = out
        self.num_cont = num_cont
        token_num = num_cont + num_cate
        categories = [] if categories is None else categories
        # self.args = args
        assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'
        assert len(categories) + num_cont > 0, 'input shape must not be null'

        # categories related calculations
        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)

        # create category embeddings table

        total_tokens = self.num_unique_categories + num_special_tokens + 1

        # for automatically offsetting unique category ids to the correct position in the categories embedding table

        if self.num_unique_categories > 0:
            categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value = num_special_tokens)
            categories_offset = categories_offset.cumsum(dim = -1)[:-1]
            self.register_buffer('categories_offset', categories_offset)

            # categorical embedding

            self.categorical_embeds = nn.Embedding(total_tokens, dim)

        # continuous

        

        if self.num_cont > 0:
            self.numerical_embedder = NumericalEmbedder(dim, self.num_cont)

        # cls token

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        # transformer

        self.transformer = Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            use_cls_token=self.use_cls_token,
            groups=groups,
            sum_num_per_group=sum_num_per_group,
            prod_num_per_group=prod_num_per_group,
            cluster=cluster,
            target_mode=target_mode,
            token_num=token_num,
            token_descent=token_descent,
            use_prod=use_prod,
            qk_relu=qk_relu,
        )

        # to logits


        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, out)
        )


        self.pool = nn.Linear(num_cont + num_cate, 1)
        

    def model_name(self):
        return 'ft_trans'
    
    def forward(self, x_num,x_cat):

        x = []
        if self.num_unique_categories > 0:
            x_cat = self.categorical_embeds(x_cat)
            x.append(x_cat)

        # add numerically embedded tokens
        if self.num_cont > 0:
            x_num = self.numerical_embedder(x_num)
            x.append(x_num)

        # concat categorical and numerical

        x = torch.cat(x, dim = 1)

        # append cls tokens
        b = x.shape[0]

        if self.use_cls_token:
            cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
            x = torch.cat((cls_tokens, x), dim = 1)


        x = self.transformer(x)

        if self.use_cls_token:
            x = x[:, 0]
        else:
            x = self.pool(x.transpose(-1, -2)).squeeze(-1)


        logit = self.to_logits(x)
        
        return logit