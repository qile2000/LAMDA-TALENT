import torch
from torch import nn

# Source: https://github.com/avivnur/SwitchTab/blob/main/model.py
# Feature corruption function
def feature_corruption(x, corruption_ratio=0.3):
    # We sample a mask of the features to be zeroed out
    corruption_mask = torch.bernoulli(torch.full(x.shape, 1-corruption_ratio)).to(x.device)
    return x * corruption_mask


# Encoder network with a three-layer transformer
class Encoder(nn.Module):
    def __init__(self, feature_size, num_heads=2):
        super(Encoder, self).__init__()
        self.transformer_layers = nn.Sequential(
            nn.TransformerEncoderLayer(d_model=feature_size, nhead=num_heads),
            nn.TransformerEncoderLayer(d_model=feature_size, nhead=num_heads),
            nn.TransformerEncoderLayer(d_model=feature_size, nhead=num_heads)
        )

    def forward(self, x):
        # Since Transformer expects seq_length x batch x features, we assume x is already shaped correctly
        return self.transformer_layers(x)


# Projector network
class Projector(nn.Module):
    def __init__(self, feature_size):
        super(Projector, self).__init__()
        self.linear = nn.Linear(feature_size, feature_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))


# Decoder network
class Decoder(nn.Module):
    def __init__(self, input_feature_size, output_feature_size):
        super(Decoder, self).__init__()
        self.linear = nn.Linear(input_feature_size, output_feature_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))


# Prediction network for pre-training
class Predictor(nn.Module):
    def __init__(self, feature_size, num_classes):
        super(Predictor, self).__init__()
        self.linear = nn.Linear(feature_size, num_classes)

    def forward(self, x):
        return self.linear(x)


class SwitchTab(nn.Module):
    def __init__(self, feature_size, num_classes, num_heads=2, alpha=1.0):
        super(SwitchTab, self).__init__()
        self.encoder = Encoder(feature_size, num_heads)
        self.projector_s = Projector(feature_size)
        self.projector_m = Projector(feature_size)
        self.decoder = Decoder(2 * feature_size, feature_size)  # Assuming concatenation of salient and mutual embeddings
        self.predictor = Predictor(feature_size, num_classes)
        self.alpha = alpha

    def forward(self, x1, x2):
        if x2 is not None:  # fit
            # Feature corruption is not included in the model itself and should be applied to the data beforehand
            z1_encoded = self.encoder(x1)
            z2_encoded = self.encoder(x2)

            s1_salient = self.projector_s(z1_encoded)
            m1_mutual = self.projector_m(z1_encoded)
            s2_salient = self.projector_s(z2_encoded)
            m2_mutual = self.projector_m(z2_encoded)

            x1_reconstructed = self.decoder(torch.cat((m1_mutual, s1_salient), dim=1))
            x2_reconstructed = self.decoder(torch.cat((m2_mutual, s2_salient), dim=1))
            x1_switched = self.decoder(torch.cat((m2_mutual, s1_salient), dim=1))
            x2_switched = self.decoder(torch.cat((m1_mutual, s2_salient), dim=1))
            
            x1_pred = self.predictor(z1_encoded)
            x2_pred = self.predictor(z2_encoded)

            return x1_reconstructed, x2_reconstructed, x1_switched, x2_switched, x1_pred, x2_pred, self.alpha
        else:  # predict
            z = self.encoder(x1)
            pred = self.predictor(z)
            return pred