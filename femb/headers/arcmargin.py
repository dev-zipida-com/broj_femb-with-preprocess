import torch


class ArcMarginHeader(torch.nn.Module):
    """ ArcMarginHeader class"""

    def __init__(self, in_features, out_features, s, m1, m2, m3, k=0):
        super(ArcMarginHeader, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.s = s
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3

        self.linear = torch.nn.Linear(in_features=in_features, out_features=out_features, bias=False)
        self.normalize = torch.nn.functional.normalize

        self.epsilon = 1e-10


    def forward(self, input, label):
        # multiply normed features (input) and normed weights to obtain cosine of theta (logits)
        self.linear.weight = torch.nn.Parameter(self.normalize(self.linear.weight))
        logits = self.linear(self.normalize(input))

        # apply arccos to get theta
        theta = torch.acos(torch.clamp(logits, -1.0-self.epsilon, 1.0+self.epsilon))
        # add angular margin (m) to theta and transform back by cos
        target_logits = torch.cos(self.m1 * theta + self.m2) - self.m3

        # derive one-hot encoding for label
        one_hot = torch.zeros(logits.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        # build the output logits
        output = one_hot * target_logits + (1.0 - one_hot) * logits
        # feature re-scaling
        output *= self.s

        return output
