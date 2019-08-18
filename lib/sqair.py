import torch
from attrdict import AttrDict
from torch import nn
from copy import deepcopy
from torch.nn import functional as F
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence
from .config import cfg
from collections import defaultdict
from .utils import vis_logger


# default sqair architecture. This is global for convenience
arch = AttrDict({
    # sequence length
    'T': cfg.dataset.seq_len,
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
    
    'z_where_loc_prior': torch.tensor([3.0, 3.0, 0.0, 0.0], device=cfg.device),
    'z_where_scale_prior': torch.tensor([0.2, 0.2, 1.0, 1.0], device=cfg.device),
    'z_what_loc_prior': torch.tensor(0.0, device=cfg.device),
    'z_what_scale_prior': torch.tensor(1.0, device=cfg.device),
    
    # output prior
    'x_scale': torch.tensor(0.3, device=cfg.device)
    
})


class ObjectState:
    """
    An object's state.
    
    This is 'z' in the paper, including presence, where and what.
    """
    
    def __init__(self, z_pres, z_where, z_what, id, z_pres_prob):
        """
        Args:
            z_pres:  (B, 1)
            z_where: (B, 4)
            z_what: (B, N)
            id: (B, 1). 0 for empty slots.
        """
        self.z_pres = z_pres
        self.z_where = z_where
        self.z_what = z_what
        self.id = id
        
        # For visualization only
        self.z_pres_prob = z_pres_prob
        
    def __getitem__(self, item):
        return getattr(self, item)
    
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
            id = torch.zeros(B, 1, device=cfg.device),
            z_pres_prob=None
        )

    
    @staticmethod
    def get_size():
        """
        The last dimension of the Tensor returned by get_encoding
        """
        return arch.z_pres_size + arch.z_where_size + arch.z_what_size
    
# TODO: The following state classes can be avoided if we use a different definition here.

class RelationState:
    """
    LSTM State that will be used in Relation RNN.
    
    This includes ObjectState, plus the LSTM h and c.
    """
    
    def __init__(self, object_state, h, c):
        self.h = h
        self.c = c
        self.object = object_state
        
        
    def __getitem__(self, item):
        return getattr(self, item)
    
    def __getattr__(self, item):
        if item not in self.__dict__:
            return getattr(self.object, item)
    
    
    @staticmethod
    def get_initial_state(batch_size, object_state=None):
        B = batch_size
        return RelationState(
            ObjectState.get_initial_state(batch_size) if object_state is None else object_state,
            h=torch.zeros(B, arch.rnn_relation_hidden_size, device=cfg.device),
            c=torch.zeros(B, arch.rnn_relation_hidden_size, device=cfg.device),
        )


class TemporalState:
    """
    LSTM State that will be used in Relation RNN.
    
    This includes ObjectState, plus the LSTM h and c.
    """
    
    def __init__(self, object_state, h, c):
        self.h = h
        self.c = c
        self.object = object_state
        
    def __getattr__(self, item):
        if item not in self.__dict__:
            return getattr(self.object, item)

    def __getitem__(self, item):
        return getattr(self, item)

    @staticmethod
    def get_initial_state(batch_size, object_state):
        B = batch_size
        return TemporalState(
            ObjectState.get_initial_state(batch_size) if object_state is None else object_state,
            h=torch.zeros(B, arch.rnn_temporal_hidden_size, device=cfg.device),
            c=torch.zeros(B, arch.rnn_temporal_hidden_size, device=cfg.device),
        )

class PropagatePriorState:
    """
    LSTM State that will be used in prior RNN.
    
    This includes ObjectState, plus the LSTM h and c.
    """
    
    def __init__(self, object_state, h, c):
        self.h = h
        self.c = c
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
            arch.embedding_size + arch.rnn_temporal_hidden_size
            + arch.rnn_relation_hidden_size, arch.predict_hidden_size)
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
        z_pres_prob = torch.sigmoid(x[:, 8:])
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
            arch.embedding_size + arch.rnn_relation_hidden_size, arch.predict_hidden_size)
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
        z_pres_prob = torch.sigmoid(x[:, 8:])
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
        z_what_loc, z_what_scale = torch.split(x, [arch.z_what_size] * 2, dim=-1)
        z_what_scale = F.softplus(z_what_scale)
        
        return z_what_loc, z_what_scale
    
class Decoder(nn.Module):
    """
    Decoder z_what into glimpse.
    """
    def __init__(self):
        nn.Module.__init__(self)
        self.fc1 = nn.Linear(arch.z_what_size, arch.decoder_hidden_size)
        self.fc2 = nn.Linear(arch.decoder_hidden_size, arch.glimpse_size)
        
    def forward(self, z_what):
        """
        Args:
            z_what: (B, N)

        Returns:
            glimpse: (B, 1, H, W)
        """
        BIAS = -2.0
        B = z_what.size(0)
        x = self.fc2(F.elu(self.fc1(z_what)))
        # This bias makes initial outputs empty
        x = F.sigmoid(x + BIAS)
        x = x.view(B, 1, *arch.glimpse_shape)
        
        return x


class PropagatePrior(nn.Module):
    """
    Compute prior parameters for current step given feature encoding for z's of
    all previous steps.
    """
    
    def __init__(self):
        nn.Module.__init__(self)
        self.fc1 = nn.Linear(arch.rnn_temporal_hidden_size,
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
            torch.split(z, [arch.z_what_size] * 2 + [4] * 2 + [1], dim=-1)
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
        expansion_indices = torch.tensor([0, 4, 2, 5, 1, 3], device=cfg.device)
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
        # self.rnn_prior = nn.LSTMCell(ObjectState.get_size(),
        #                              arch.rnn_prior_hidden_size)
        
        # Predict z_pres, z_where given h^T, h^T, x
        self.propagate_predict = PropagatePredict()
        self.discover_predict = DiscoverPredict()
        
        # Spatial transformers
        self.glimpse_to_image = SpatialTransformer(arch.glimpse_shape, arch.input_shape)
        self.image_to_glimpse = SpatialTransformer(arch.input_shape, arch.glimpse_shape)
        
        # Encode glimpse into z_what parameters
        self.encoder = Encoder()
        self.decoder = Decoder()

        # This is a module, computing current prior parameters from previous
        # priors (encoded in LSTM)
        self.propagate_prior = PropagatePrior()
        
        
        # Embedding layer
        self.embedding = Embedding()
        
        
        # Per-batch current highest id. Shape (B, 1). Initialized later.
        self.highest_id = None
        
        
    def forward(self, x):
        """
        Args:
            x: image sequence of shape (T, B, 1, H, W)

        Returns:
            elbo: (B,) ELBO
            z_pres_likelihood_total: (B,), likelihood sum for z_pres
        """
        
        B = x.size(1)
        
        # (B,)
        kl_total = 0.0
        z_pres_likelihood_total = 0.0
        self.highest_id = torch.zeros(B, 1, device=x.device)
        # (B,)
        likelihood_total = 0.0

        # Finally this will be a list of list of caanvases
        vis_logger['canvas'] = []
        vis_logger['z_pres'] = []
        vis_logger['z_pres_prob'] = []
        vis_logger['z_where'] = []
        vis_logger['id'] = []
        vis_logger['imgs'] = []
        
        # Append T*K items
        vis_logger['kl_pres_list'] = []
        vis_logger['kl_what_list'] = []
        vis_logger['kl_where_list'] = []

        for t in range(arch.T):
            # Enter time step t
            # (B, 1, H, W)
            image = x[t]
            vis_logger['imgs'].append(image[0][0])
            # Compute embedding for this image (B, N)
            image_embed = self.embedding(image)
            
            
            # Initialization
            state_rel = RelationState.get_initial_state(B)
            state_tem_list = []
            
            # Propagate.
            # Note we do not  propagate for the first time step.
            if t != 0:
                # Do propagate
                for k in range(arch.K):
                    state_tem = state_tem_list_prev[k]
                    state_tem, state_rel, kl, z_pres_likelihood = (
                        self.propagate(image, image_embed, state_rel, state_tem))
                    
                    state_tem_list.append(state_tem)
                    kl_total += kl
                    z_pres_likelihood_total += z_pres_likelihood

            # Discover
            for k in range(arch.K):
                # One step of discover
                state_tem, state_rel, kl, z_pres_likelihood = (
                    self.discover(image, image_embed, state_rel))
                
                # Record temporal states
                state_tem_list.append(state_tem)
                # Record losses
                kl_total += kl
                z_pres_likelihood_total += z_pres_likelihood
                
            # Temporal state
            # This will be of length K (for first step) or 2K. Each item is a
            # temporal state.
            # Note that though not so elegant, objects will also be kept in state_tem_list.
            state_tem_list_prev = self.rearrange(state_tem_list)
            
            # Extract object list
            object_list = [state.object for state in state_tem_list_prev]
            
            
            # Reconstruct and compute likelihood
            recons, ll = self.reconstruct(object_list, image)
            likelihood_total += ll
            
            
        elbo = likelihood_total - kl_total
        
        vis_logger['kl'] = kl_total.mean()
        vis_logger['likelihood'] = likelihood_total.mean()
        vis_logger['elbo'] = elbo.mean()
        
        return elbo, z_pres_likelihood_total
            
            
            
        
    
    def propagate(self, x, x_embed, prev_relation, prev_temporal):
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
            
        Returns:
            temporal_state: TemporalState
            relation_state:  RelationState
            kl: kl divergence for all z's. (B, 1)
            z_pres_likelihood: q(z_pres|x). (B, 1)
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
        z_what = z_what_post.rsample()
        # Mask
        z_what = z_what * z_pres
        
        # Now we compute KL divergence and discrete likelihood. Before that, we
        # will need to compute the recursive prior. This is parametrized by
        # previous object state (z) and hidden states from LSTM.
        
        
        # Compute prior for current step
        (z_what_loc_prior, z_what_scale_prior, z_where_loc_prior,
            z_where_scale_prior, z_pres_prob_prior) = (
            self.propagate_prior(h_tem))
        
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
        
        vis_logger['kl_pres_list'].append(kl_z_pres.mean())
        vis_logger['kl_what_list'].append(kl_z_what.mean())
        vis_logger['kl_where_list'].append(kl_z_where.mean())

        # (B,) here, after reduction
        kl = kl_z_what.sum(dim=-1) + kl_z_where.sum(dim=-1) + kl_z_pres.sum(dim=-1)
        
        # Finally, we compute the discrete likelihoods.
        z_pres_likelihood = z_pres_post.log_prob(z_pres)
        
        # Note we also need to mask some of this terms since they do not depend
        # on model parameter. (B, 1) here
        z_pres_likelihood = z_pres_likelihood * prev_temporal.object.z_pres
        z_pres_likelihood = z_pres_likelihood.squeeze()
        
        # Compute id. If z_pres is 1, then inherit that id. Otherwise set it to
        # zero
        id = prev_temporal.object.id * z_pres
        
        # Collect terms into new states.
        object_state = ObjectState(z_pres, z_where, z_what, id, z_pres_prob=z_pres_prob)
        temporal_state = TemporalState(object_state, h_tem, c_tem)
        relation_state = RelationState(object_state, h_rel, c_rel)
        
        return temporal_state, relation_state, kl, z_pres_likelihood

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
            kl: kl divergence for all z's. (B,)
            z_pres_likelihood: q(z_pres|x). (B,)
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
        z_what = z_what_post.rsample()
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
        
        vis_logger['kl_pres_list'].append(kl_z_pres.mean())
        vis_logger['kl_what_list'].append(kl_z_what.mean())
        vis_logger['kl_where_list'].append(kl_z_where.mean())
        
        # (B,) here, after reduction
        kl = kl_z_what.sum(dim=-1) + kl_z_where.sum(dim=-1) + kl_z_pres.sum(dim=-1)

        # Finally, we compute the discrete likelihoods.
        z_pres_likelihood = z_pres_post.log_prob(z_pres)

        # Note we also need to mask some of this terms since they do not depend
        # on model parameter. (B, 1) here
        z_pres_likelihood = z_pres_likelihood * prev_relation.object.z_pres
        z_pres_likelihood = z_pres_likelihood.squeeze()
        
        
        # Compute id. If z_pres = 1, highest_id += 1, and use that id. Otherwise
        # we do not change the highest id and set id to zero
        self.highest_id += z_pres
        id = self.highest_id * z_pres

        # Collect terms into new states.
        object_state = ObjectState(z_pres, z_where, z_what, id=id, z_pres_prob=z_pres_prob)
        
        # For temporal and prior state, we will use the initial state.
        
        B = x.size(0)
        temporal_state = TemporalState.get_initial_state(B, object_state)
        relation_state = RelationState(object_state, h_rel, c_rel)


        return temporal_state, relation_state, kl, z_pres_likelihood
    
    def rearrange(self, temporal_states):
        """
        Some propagation slots are empty. So we move them to the end.
        Args:
            temporal_states: a list of length K or 2K. Each item is a TemporalState

        Returns:
            temporal_states: a new list of states. Length will be K.
        """
        
        K = len(temporal_states)
        # TemporalState has six fields, z_where, z_what, z_pres, id. First,
        # concatenate these into a large tensor
        z = defaultdict(list)
        for key in ['z_where', 'z_what', 'z_pres', 'id', 'h', 'c', 'z_pres_prob']:
            for state in temporal_states:
                z[key].append(state[key])
            # K * (B, N) ->  (B, K, N)
            z[key] = torch.stack(z[key], dim=1)
            
        # (B, K, 1) -> (B, K, 1)
        # This is the indices that will be used to rearrange everything.
        # Note that argsort is not guaranteed to preserve order. But that does
        # not matter.
        
        # This is ugly, but it preserves the order.
        indices = torch.argsort((1.0 - z['z_pres']) * 1000 + z['id'], dim=1)
        
        # Since for a of shape (B, K, N), input should be
        #       output[i, j, k] = in[i, indices[i, j, k], k]
        # we can implement this as torch.gather.
        
        for key in ['z_where', 'z_what', 'z_pres', 'id', 'h', 'c', 'z_pres_prob']:
            # Note we need to expand (B, K, 1) to (B, K, N) for each key
            z[key] = torch.gather(z[key], dim=1, index=indices.expand_as(z[key]))
        
        new_states = []
        # Now reorganized (B, K, N) -> K * (B, N).
        # Note that we only collect the first K states even if there are 2K states.
        # This will not harm.
        for k in range(K):
            new_object = ObjectState(
                z_what=z['z_what'][:, k],
                z_where=z['z_where'][:, k],
                z_pres=z['z_pres'][:, k],
                id=z['id'][:, k],
                z_pres_prob=z['z_pres_prob'][:, k],
            )
            new_h = z['h'][:, k]
            new_c = z['c'][:, k]
            new_state = TemporalState(new_object, new_h, new_c)
            new_states.append(new_state)
            
        return new_states
    
    def reconstruct(self, object_list, img):
        """
        Do reconstruction and compute likelihood.
        
        Args:
            object_list: a list of ObjectState of length K.
            img: original image of shape (B, 1, H, W)

        Returns:
            recons: reconstructed image. (B, 1, H, W)
            likelihood: (B,)
        """
        
        B = img.size(0)
        canvas = torch.zeros_like(img)

        vis_logger['canvas_cur'] = []
        vis_logger['z_pres_cur'] = []
        vis_logger['z_where_cur'] = []
        vis_logger['z_pres_prob_cur'] = []
        vis_logger['id_cur'] = []
        
        for obj in object_list:
            # Decode to (B, 1, H, W)
            glimpse = self.decoder(obj.z_what)
            # Transform to image size
            recons = self.glimpse_to_image(glimpse, obj.z_where, inverse=False)
            
            # (B, 1, 1, 1)
            mask = obj.z_pres[:,:,None,None]
            # (B, 1, H, W)
            canvas = canvas + mask * recons
            
            # (H, W)
            vis_logger['canvas_cur'].append(canvas[0][0])
            # Scalar
            vis_logger['z_pres_cur'].append(obj.z_pres[0][0])
            # Scalar
            vis_logger['z_pres_prob_cur'].append(obj.z_pres_prob[0][0])
            # (4,)
            vis_logger['z_where_cur'].append(obj.z_where[0])
            # Scalar
            vis_logger['id_cur'].append(obj.id[0][0])

        vis_logger['canvas'].append(vis_logger['canvas_cur'])
        vis_logger['z_pres'].append(vis_logger['z_pres_cur'])
        vis_logger['z_pres_prob'].append(vis_logger['z_pres_prob_cur'])
        vis_logger['z_where'].append(vis_logger['z_where_cur'])
        vis_logger['id'].append(vis_logger['id_cur'])
        
        # Construct output distribution
        output_dist = Normal(canvas, arch.x_scale.expand(canvas.size()))
        
        # Likelihood
        likelihood = output_dist.log_prob(img)
        likelihood = likelihood.view(B, -1).sum(1)
        
        return canvas, likelihood
            
            
