import torch
from attrdict import AttrDict
from torch import nn
from copy import deepcopy
from torch.nn import functional as F
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence
from .config import cfg


# default sqair architecture. This is global for convenience
arch = AttrDict({
    # sequence length
    'T': 10,
    # maximum number of objects
    'K': 3,
    
    # network related
    'input_shape': (50, 50),
    'glimpse_shape': (20, 20),
    'input_size': 50 * 50,
    'glimpse_size': 20 * 20,
    'z_what_size': 50,
    'z_where_size': 4,
    'z_pres_size': 1,
    
    # rnns
    'rnn_relation_hidden_size': 256,
    'rnn_temporal_hidden_size': 256,
    'rnn_prior_hidden_size': 256,
    
    # object embedding
    'embedding_size': 256,
    'embedding_hidden_size': 512,
    
    # predict
    'predict_hidden_size': 256,
    
    # encode object into z_what
    'encoder_hidden_size': 256,
    'decoder_hidden_size': 256,
    
    # prior network hidden_size
    'prior_hidden_size': 256,
    
    # Prior parameters. Note that this are used ONLY in discover.
    'z_pres_prob_prior': torch.tensor(cfg.anneal.initial, device=cfg.device),
    
    'z_where_loc_prior': torch.tensor([3.0, 0.0, 0.0], device=cfg.device),
    'z_where_scale_prior': torch.tensor([0.2, 1.0, 1.0], device=cfg.device),
    'z_what_loc_prior': torch.tensor(0.0, device=cfg.device),
    'z_what_scale_prior': torch.tensor(1.0, device=cfg.device),
    
    # output prior
    'x_scale': torch.tensor(0.3, device=cfg.device)
    
})


class ObjectState(AttrDict):
    """
    An object's state.
    
    This is 'z' in the paper, including presence, where and what.
    """
    
    def __init__(self, z_pres, z_where, z_what):
        """
        Args:
            z_pres:  (B, 1)
            z_where: (B, 4)
            z_what: (B, N)
        """
        AttrDict.__init__(
            self,
            z_pres=z_pres,
            z_where=z_where,
            z_what=z_what,
        )
    
    def get_encoding(self):
        """
        Get a feature representation for current object.
        Returns:

        """
        return torch.cat([self.z_pres, self.z_where, self.z_what], dim=-1)
    
    @staticmethod
    def get_initial_state(batch_size):
        B = batch_size
        return ObjectState(
            z_pres=torch.ones(B, 1, device=cfg.device),
            z_where=torch.zeros(B, 4, device=cfg.device),
            z_what=torch.zeros(B, arch.z_what_size, device=cfg.device),
        )

    
    @staticmethod
    def get_size():
        """
        The last dimension of the Tensor returned by get_encoding
        """
        return arch.z_pres_size + arch.z_where_size + arch.z_pres_size
    
# TODO: The following state classes can be avoided if we use a different definition here.

class RelationState(AttrDict):
    """
    LSTM State that will be used in Relation RNN.
    
    This includes ObjectState, plus the LSTM h and c.
    """
    
    def __init__(self, object_state, h, c):
        AttrDict.__init__(self, h=h, c=c)
        self.object_state = object_state
        
    @staticmethod
    def get_initial_state(batch_size, object_state=None):
        B = batch_size
        return RelationState(
            ObjectState.get_initial_state(batch_size) if object_state is None else object_state,
            h=torch.zeros(B, arch.rnn_relation_hidden_size),
            c=torch.zeros(B, arch.rnn_relation_hidden_size),
        )


class TemporalState(AttrDict):
    """
    LSTM State that will be used in Relation RNN.
    
    This includes ObjectState, plus the LSTM h and c.
    """
    
    def __init__(self, object_state, h, c):
        AttrDict.__init__(self, h=h, c=c)
        self.object_state = object_state

    @staticmethod
    def get_initial_state(batch_size, object_state):
        B = batch_size
        return TemporalState(
            ObjectState.get_initial_state(batch_size) if object_state is None else object_state,
            h=torch.zeros(B, arch.rnn_temporal_hidden_size),
            c=torch.zeros(B, arch.rnn_temporal_hidden_size),
        )

class PropagatePriorState(AttrDict):
    """
    LSTM State that will be used in prior RNN.
    
    This includes ObjectState, plus the LSTM h and c.
    """
    
    def __init__(self, object_state, h, c):
        AttrDict.__init__(self, h=h, c=c)
        self.object = object_state

    @staticmethod
    def get_initial_state(batch_size, object_state):
        B = batch_size
        return PropagatePriorState(
            ObjectState.get_initial_state(batch_size) if object_state is None else object_state,
            h=torch.zeros(B, arch.rnn_temporal_hidden_size),
            c=torch.zeros(B, arch.rnn_temporal_hidden_size),
        )

class Embedding(nn.Module):
    """
    Embedding the image into feature
    """
    
    def __init__(self):
        nn.Module.__init__(self)
        self.fc1 = nn.Linear(arch.input_size, arch.embedding_hidden_size)
        self.fc2 = nn.Linear(arch.embedding_hidden_size, arch.embedding_size)
    
    def forward(self, img):
        """
        Args:
            img: (B, C, H, W)

        Returns:
            Embeded feature. (B, N)
        """
        B = img.size(0)
        img = img.view(B, -1)
        return self.fc2(F.elu(self.fc1(img)))


class PropagatePredict(nn.Module):
    """
    Predict pres and where given h^T, h^R, x.
    
    This is used in propagate. I seperate these two because I may use delta in
    propagate later.
    """
    
    def __init__(self):
        nn.Module.__init__(self)
        self.fc1 = nn.Linear(
            arch.embedding_size + arch.temporal_hidden_size
            + arch.relation_hidden_size, arch.predict_hidden_size)
        # 4 + 1: z_where + z_pres
        self.fc2 = nn.Linear(arch.predict_hidden_size, 4 * 2 + 1)
    
    def forward(self, x):
        """
        Args:
            x: feature encoding h^T, h^R, img
        Returns:
            z_where_loc: (B, 4)
            z_where_scale: (B, 4)
            z_pres_prob: (B, 1)
        """
        x = self.fc2(F.elu(self.fc1(x)))
        z_where_loc = x[:, :4]
        z_where_scale = F.softplus(x[:, 4:8])
        z_pres_prob = torch.sigmoid(x[8:])
        # NOTE: for numerical stability, if z_pres_p is 0 or 1, we will need to
        # clamp it to within (0, 1), or otherwise the gradient will explode
        eps = 1e-6
        z_pres_prob = z_pres_prob + eps * (z_pres_prob == 0).float() - eps * (z_pres_prob == 1).float()
        
        return z_where_loc, z_where_scale, z_pres_prob

class DiscoverPredict(nn.Module):
    """
    Predict pres and where given  h^R, x.
    
    This is used in discover. I seperate these two because I may use delta in
    propagate later.
    """
    
    def __init__(self):
        nn.Module.__init__(self)
        self.fc1 = nn.Linear(
            arch.embedding_size + arch.relation_hidden_size, arch.predict_hidden_size)
        # 4 + 1: z_where + z_pres
        self.fc2 = nn.Linear(arch.predict_hidden_size, 4 * 2 + 1)
    
    def forward(self, x):
        """
        Args:
            x: feature encoding h^T, h^R, img
        Returns:
            z_where_loc: (B, 4)
            z_where_scale: (B, 4)
            z_pres_prob: (B, 1)
        """
        x = self.fc2(F.elu(self.fc1(x)))
        z_where_loc = x[:, :4]
        z_where_scale = F.softplus(x[:, 4:8])
        z_pres_prob = torch.sigmoid(x[8:])
        # NOTE: for numerical stability, if z_pres_p is 0 or 1, we will need to
        # clamp it to within (0, 1), or otherwise the gradient will explode
        eps = 1e-6
        z_pres_prob = z_pres_prob + eps * (z_pres_prob == 0).float() - eps * (z_pres_prob == 1).float()
    
        return z_where_loc, z_where_scale, z_pres_prob



class Encoder(nn.Module):
    """
    Encoder glimpse into parameters of z_what distribution
    """
    
    def __init__(self):
        nn.Module.__init__(self)
        self.fc1 = nn.Linear(arch.glimpse_size, arch.encoder_hidden_size)
        # 4 + 1: z_where + z_pres
        self.fc2 = nn.Linear(arch.encoder_hidden_size, 2 * arch.z_what_size)
    
    def forward(self, x):
        """
        Args:
            x: glimpse. Shape (B, 1, H, W)
        Returns:
            z_what_loc: (B, N)
            z_what_scale: (B, N)
        """
        x = x.view(x.size(0), -1)
        # (B, 2*N)
        x = self.fc2(F.elu(self.fc1(x)))
        # (B, N), (B, N)
        z_what_loc, z_what_scale = torch.split(x, 2, dim=-1)
        z_what_scale = F.softplus(z_what_scale)
        
        return z_what_loc, z_what_scale


class PropagatePrior(nn.Module):
    """
    Compute prior parameters for current step given feature encoding for z's of
    all previous steps.
    """
    
    def __init__(self):
        nn.Module.__init__(self)
        self.fc1 = nn.Linear(arch.rnn_prior_hidden_size,
                             arch.prior_hidden_size)
        self.fc2 = nn.Linear(arch.prior_hidden_size, 2 * arch.z_what_size + 4 * 2 + 1)
    
    def forward(self, x):
        """
        Args:
            x: hidden state of the prior RNN. Shape (B, N)

        Returns:
            z_what_loc, z_what_scale, z_where_loc, z_where_scale, z_pres_prob.
            All with shape (B, N).
        """
        z = self.fc2(F.elu(self.fc1(x)))
        z_what_loc, z_what_scale, z_where_loc, z_where_scale, z_pres_prob = \
            torch.split(x, [arch.z_what_size] * 2 + [4] * 2 + [1])
        z_what_scale = F.softplus(z_what_scale)
        z_where_scale = F.softplus(z_where_scale)
        z_pres_prob = torch.sigmoid(z_pres_prob)
        eps = 1e-6
        z_pres_prob = z_pres_prob + eps * (z_pres_prob == 0).float() - eps * (z_pres_prob == 1).float()
        
        return z_what_loc, z_what_scale, z_where_loc, z_where_scale, z_pres_prob


class SpatialTransformer(nn.Module):
    """
    Spatial transformer that transforms between glimpse and whole image.
    """
    
    def __init__(self, input_size, output_size):
        """
        Args:
            input_size: (H, W)
            output_size: (H, W)
        """
        nn.Module.__init__(self)
        self.input_size = input_size
        self.output_size = output_size
    
    def forward(self, x, z_where, inverse=False):
        """
        Transform the image.
        
        Note that z_where is for glimpse-to-image transform. That
        means z[:, 0] should be larger than 1.
        Args:
            x: a batch of image. (B, C, Hin, Win)
            z_where: (B, 4). Each being (sx, sy, x, y)
            inverse: whether glimpse-to-image or the inverse.
            
        Returns:
            out: (B, C, Hout, Wout)
        """
        B = x.size(0)
        # Compute sampling grid
        theta = self.z_where_to_matrix(z_where, inverse)
        grid = F.affine_grid(theta, torch.Size((B, 1)) + self.output_size)
        out = F.grid_sample(x, grid)
        
        return out
    
    @staticmethod
    def z_where_to_matrix(z_where, inverse):
        """
        Tranforms the tuple (sx, sy, x, y) to matrix
        [[sx, 0, x], [0, sy, y]]
        
        Args:
            z_where: shape (B, 4)
            inverse: bool

        Returns:
            Transformed matrix with shape (B, 2, 3)
        """
        
        if inverse:
            z_where = SpatialTransformer.inverse_z_where(z_where)
        
        # Padding to (sx, sy, x, y, 0, 0)
        B = z_where.size(0)
        padding = torch.zeros(B, 2, device=z_where.device)
        z_where = torch.cat((z_where, padding), dim=-1)
        
        # Rearange to (sx, 0, x, 0, sy, y)
        expansion_indices = torch.tensor([0, 4, 2, 5, 1, 3])
        matrix = torch.index_select(z_where, dim=-1, index=expansion_indices)
        matrix = matrix.view(-1, 2, 3)
        
        return matrix
    
    @staticmethod
    def inverse_z_where(z_where):
        """
        Inverse the tranformation
        Args:
            z_where: (B, 4)

        Returns:
            z_where_inv: (B, 4)
        """
        z_where_inv = torch.zeros_like(z_where)
        z_where_inv[:, 2:4] = -z_where[:, 2:4] / (z_where[:, 0:2] + 1e-6)
        z_where_inv[:, 0:2] = 1 / (z_where[:, 0:2] + 1e-6)
        
        return z_where_inv


class SQAIR(nn.Module):
    def __init__(self, arch_update=None):
        nn.Module.__init__(self)
        arch.update(arch_update if arch_update else {})
        
        # Relation RNN
        self.rnn_relation = nn.LSTMCell(ObjectState.get_size(),
                                        arch.rnn_relation_hidden_size)
        
        # Temporal RNN
        self.rnn_temporal = nn.LSTMCell(ObjectState.get_size(),
                                        arch.rnn_temporal_hidden_size)
        
        # Prior RNN
        self.rnn_prior = nn.LSTMCell(ObjectState.get_size(),
                                     arch.rnn_prior_hidden_size)
        
        # Predict z_pres, z_where given h^T, h^T, x
        self.propagate_predict = PropagatePredict()
        self.discover_predict = DiscoverPredict()
        
        # Spatial transformers
        self.glimpse_to_image = SpatialTransformer(arch.glimpse_shape, arch.input_shape)
        self.image_to_glimpse = SpatialTransformer(arch.input_shape, arch.glimpse_shape)
        
        # Encode glimpse into z_what parameters
        self.encoder = Encoder()

        # This is a module, computing current prior parameters from previous
        # priors (encoded in LSTM)
        self.propagate_prior = PropagatePrior()
        
        
        # Initial relation, temporal, relation, object state
        
        
    def forward(self, x):
        pass
    
    def propagate(self, x, x_embed, prev_relation, prev_temporal, prev_prior):
        """
        Propagate step for a single object in a single time step.
        
        In this process, even empty objects are encoded in relation h's. This
        can be avoided by directly passing the previous relation state on to the
        next spatial step. May do this later.
        
        Args:
            x: original image. Size (B, C, H, W)
            x_embed: extracted image feature. Size (B, N)
            prev_relation: see RelationState
            prev_temporal: see TemporalState
            prev_prior: see PriorState
            
        Returns:
            temporal_state: TemporalState
            relation_state:  RelationState
            prior_state: PropagatePriorState
            kl: kl divergence for all z's. Scalar
            z_pres_likelihood: q(z_pres|x). Scalar
        """
        
        # First, encode relation and temporal info to get current h^{T, i}_t and
        # current h^{R, i}_t
        # Each being (B, N)
        h_rel, c_rel = self.rnn_relation(prev_relation.object.get_encoding(),
                                         (prev_relation.h, prev_relation.c))
        h_tem, c_tem = self.rnn_temporal(prev_temporal.object.get_encoding(),
                                         (prev_temporal.h, prev_temporal.c))
        # (B, N)
        # Predict where and pres, using h^T, h^T and x
        predict_input = torch.cat((h_rel, h_tem, x_embed), dim=-1)
        # (B, 4), (B, 4), (B, 1)
        z_where_loc, z_where_scale, z_pres_prob = self.propagate_predict(predict_input)
        
        # Sample from z_pres posterior. Shape (B, 1)
        # NOTE: don't use zero probability otherwise log q(z|x) will not work
        z_pres_post = Bernoulli(z_pres_prob)
        z_pres = z_pres_post.sample()
        # Mask z_pres. You don't have do this for where and what because
        #   - their KL will be masked
        #   - they will be masked when computing the likelihood
        z_pres = z_pres * prev_temporal.object.z_pres
        
        # Sample from z_where posterior, (B, 4)
        z_where_post = Normal(z_where_loc, z_where_scale)
        z_where = z_where_post.rsample()
        # Mask
        z_where = z_where * z_pres
        
        # Extract glimpse from x, shape (B, 1, H, W)
        glimpse = self.image_to_glimpse(x, z_where, inverse=True)
        
        # Compute postribution over z_what and sample
        z_what_loc, z_what_scale = self.encoder(glimpse)
        z_what_post = Normal(z_what_loc, z_what_scale)
        # (B, N)
        z_what = z_where_post.rsample()
        # Mask
        z_what = z_what * z_pres
        
        # Now we compute KL divergence and discrete likelihood. Before that, we
        # will need to compute the recursive prior. This is parametrized by
        # previous object state (z) and hidden states from LSTM.
        
        # Compute current state
        h_prior, c_prior = self.rnn_prior(prev_prior.object.get_encoding(),
                                          (prev_prior.h, prev_prior.c))
        
        # Compute prior for current step
        (z_what_loc_prior, z_what_scale_prior, z_where_loc_prior,
            z_where_scale_prior, z_pres_prob_prior) = (
            self.propagate_prior(h_prior))
        
        # Construct prior distributions
        z_what_prior = Normal(z_what_loc_prior, z_what_scale_prior)
        z_where_prior = Normal(z_where_loc_prior, z_where_scale_prior)
        z_pres_prior = Bernoulli(z_pres_prob_prior)
        
        # Compute KL divergence. Each (B, N)
        kl_z_what = kl_divergence(z_what_post, z_what_prior)
        kl_z_where = kl_divergence(z_where_post, z_where_prior)
        kl_z_pres = kl_divergence(z_pres_post, z_pres_prior)
        
        # Mask these terms.
        # Note for kl_z_pres, we will need to use z_pres of previous time step.
        # This this because even z_pres[t] = 0, p(z_pres[t] | z_prse[t-1])
        # cannot be ignored.
        
        # Also Note that we do not mask where and what here. That's OK since We will
        # mask the image outside of the function later.
        
        kl_z_what = kl_z_what * z_pres
        kl_z_where = kl_z_where * z_pres
        kl_z_pres = kl_z_pres * prev_temporal.object.z_pres
        # (B,) here, after reduction
        kl = kl_z_what.sum(dim=-1) + kl_z_where.sum(dim=-1) + kl_z_pres.sum(dim=-1)
        
        # Finally, we compute the discrete likelihoods.
        z_pres_likelihood = z_pres_post.log_prob(z_pres)
        
        # Note we also need to mask some of this terms since they do not depend
        # on model parameter. (B, 1) here
        z_pres_likelihood = z_pres_likelihood * prev_temporal.object.z_pres.sum(-1)
        
        # Collect terms into new states.
        object_state = ObjectState(z_pres, z_where, z_what)
        temporal_state = TemporalState(object_state, h_tem, c_tem)
        relation_state = RelationState(object_state, h_rel, c_rel)
        prior_state = PropagatePriorState(object_state, h_prior, c_prior)
        
        kl = kl.mean()
        z_pres_likelihood = z_pres_likelihood.mean()
        return temporal_state, relation_state, prior_state, kl, z_pres_likelihood

    def discover(self, x, x_embed, prev_relation):
        """
        Discover step for a single object in a single time step.
        
        This is basically the same as propagate, but without a temporal state
        input. However, to share the same Predict module, we will use an empty
        temporal state instead.
        
        There are multiple code replication here. I do this because refactoring
        will not be a good abstraction.
        
        Args:
            x: original image. Size (B, C, H, W)
            x_embed: extracted image feature. Size (B, N)
            prev_relation: see RelationState
            
        Returns:
            temporal_state: TemporalState
            relation_state:  RelationState
            prior_state: PropagatePriorState
            kl: kl divergence for all z's. Scalar
            z_pres_likelihood: q(z_pres|x). Scalar
        """
        # First, encode relation info to get current h^{R, i}_t
        # Each being (B, N)
        h_rel, c_rel = self.rnn_relation(prev_relation.object.get_encoding(),
                                         (prev_relation.h, prev_relation.c))
        # (B, N)
        # Predict where and pres, using h^R, and x
        predict_input = torch.cat((h_rel, x_embed), dim=-1)
        # (B, 4), (B, 4), (B, 1)
        z_where_loc, z_where_scale, z_pres_prob = self.discover_predict(predict_input)

        # Sample from z_pres posterior. Shape (B, 1)
        # NOTE: don't use zero probability otherwise log q(z|x) will not work
        z_pres_post = Bernoulli(z_pres_prob)
        z_pres = z_pres_post.sample()
        # Mask z_pres. You don't have do this for where and what because
        #   - their KL will be masked
        #   - they will be masked when computing the likelihood
        z_pres = z_pres * prev_relation.object.z_pres

        # Sample from z_where posterior, (B, 4)
        z_where_post = Normal(z_where_loc, z_where_scale)
        z_where = z_where_post.rsample()
        # Mask
        z_where = z_where * z_pres

        # Extract glimpse from x, shape (B, 1, H, W)
        glimpse = self.image_to_glimpse(x, z_where, inverse=True)

        # Compute postribution over z_what and sample
        z_what_loc, z_what_scale = self.encoder(glimpse)
        z_what_post = Normal(z_what_loc, z_what_scale)
        # (B, N)
        z_what = z_where_post.rsample()
        # Mask
        z_what = z_what * z_pres

        # Construct prior distributions
        z_what_prior = Normal(arch.z_what_loc_prior, arch.z_what_scale_prior)
        z_where_prior = Normal(arch.z_where_loc_prior, arch.z_where_scale_prior)
        z_pres_prior = Bernoulli(arch.z_pres_prob_prior)

        # Compute KL divergence. Each (B, N)
        kl_z_what = kl_divergence(z_what_post, z_what_prior)
        kl_z_where = kl_divergence(z_where_post, z_where_prior)
        kl_z_pres = kl_divergence(z_pres_post, z_pres_prior)

        # Mask these terms.
        # Note for kl_z_pres, we will need to use z_pres of previous time step.
        # This this because even z_pres[t] = 0, p(z_pres[t] | z_prse[t-1])
        # cannot be ignored.

        # Also Note that we do not mask where and what here. That's OK since We will
        # mask the image outside of the function later.

        kl_z_what = kl_z_what * z_pres
        kl_z_where = kl_z_where * z_pres
        kl_z_pres = kl_z_pres * prev_relation.object.z_pres
        # (B,) here, after reduction
        kl = kl_z_what.sum(dim=-1) + kl_z_where.sum(dim=-1) + kl_z_pres.sum(dim=-1)

        # Finally, we compute the discrete likelihoods.
        z_pres_likelihood = z_pres_post.log_prob(z_pres)

        # Note we also need to mask some of this terms since they do not depend
        # on model parameter. (B, 1) here
        z_pres_likelihood = z_pres_likelihood * prev_relation.object.z_pres.sum(-1)

        # Collect terms into new states.
        object_state = ObjectState(z_pres, z_where, z_what)
        
        # For temporal and prior state, we will use the initial state.
        
        B = x.size(0)
        temporal_state = TemporalState.get_initial_state(B, object_state)
        prior_state = PropagatePriorState.get_initial_state(B, object_state)
        relation_state = RelationState(object_state, h_rel, c_rel)


        # Mean over batch
        kl = kl.mean()
        z_pres_likelihood = z_pres_likelihood.mean()
        
        return temporal_state, relation_state, prior_state, kl, z_pres_likelihood
    
    def rearange(self, temporal_states, prior_states):
        """
        Some propagation slots are empty. So we move them to the end.
        Args:
            temporal_states: a list of length T. Each item is a TemporalState
            prior_states: a list of length T. Each item is a PropagatePriorState

        Returns:

        """
        

