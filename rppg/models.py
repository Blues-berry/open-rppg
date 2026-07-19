import os
os.environ["KERAS_BACKEND"] = "jax" 
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false" 
#os.environ["JAX_COMPILATION_CACHE_DIR"] = "./.rppg_jitted" 
import numpy as np

import jax, keras
from keras import ops
from keras import layers
from jax import numpy as jnp

from functools import partial
from itertools import product
from einops import rearrange, repeat
import pickle, functools
from keras import initializers

keras.mixed_precision.set_global_policy("mixed_float16")

class RMSNorm(layers.Layer):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps
        
    def build(self, input_shape):
        super().build(input_shape)
        self.weight = self.add_weight(shape=(input_shape[-1],),
                                        initializer=keras.initializers.Constant(1.),
                                        dtype='float32',
                                        trainable=True)
    
    def call(self, x, z=None):
        x = ops.cast(x, 'float32')
        if z is not None:
            x = x*layers.Activation('silu')(z)
        return x*ops.rsqrt(ops.mean((x**2), -1, True)+self.eps)*self.weight

def segsum(x):
    T = x.shape[-1]
    x = repeat(x, '... d -> ... d e', e=T)
    mask = np.tril(np.ones((T, T), dtype='float32'), -1)
    x *= mask
    x_segsum = ops.cumsum(x, axis=-2)
    mask = 1-np.tril(np.ones((T, T), dtype='float32'))
    x_segsum -= ops.numpy.nan_to_num(ops.array(np.inf)*mask)
    return x_segsum


def ssd(x, A, B, C, chunk=64, init_stat=None):
    x, A, B, C = [rearrange(i, "b (c l) ... -> b c l ...", l=chunk) for i in (x, A, B, C)]
    A = rearrange(A, "b c l h -> b h c l")
    A_cum = ops.cumsum(A, axis=-1)
    
    L = ops.exp(segsum(A))
    # print(C.dtype, B.dtype, L.dtype, x.dtype)
    Y_diag = ops.einsum("bclhn, bcshn, bhcls, bcshp -> bclhp", ops.tile(C, (1, 1, 1, L.shape[1], 1)), ops.tile(B, (1, 1, 1, L.shape[1], 1)), L, x)
    
    decay_states = ops.exp(A_cum[...,-1:]-A_cum)
    states = ops.einsum("bclhn, bhcl, bclhp -> bchpn", ops.tile(B, (1, 1, 1, L.shape[1], 1)), decay_states, x)
    
    if init_stat is None:
        init_stat = ops.zeros_like(states[:, :1])
    states = ops.concatenate([init_stat, states], axis=1)
    decay_chunk = ops.exp(segsum(ops.pad(A_cum[...,-1], ((0, 0), (0, 0), (1, 0)))))
    new_states = ops.einsum('bhzc, bchpn -> bzhpn', decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1:]
    
    state_decay_out = ops.exp(A_cum)
    Y_off = ops.einsum("bclhn, bchpn, bhcl -> bclhp", ops.tile(C, (1, 1, 1, L.shape[1], 1)), states, state_decay_out)
    
    Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")
    
    return Y, final_state

class Mamba2(layers.Layer):
    def __init__(self, expand=2, d_state=128, k_conv=4, chunk_size=64, headdim=64, dtype='float32', **kw):
        super().__init__(**kw)
        self._dtype = dtype
        self.expand = expand
        self.d_state = d_state
        self.k_conv = k_conv
        self.chunk_size = chunk_size
        self.headdim = headdim
        
    def build(self, input_shape):
        """
        input_shape: (Batch, len, n_heads*headdim//expand)
        """
        super().build(input_shape)
        self.d_inner = self.expand*input_shape[-1]
        self.nheads = self.d_inner // self.headdim
        d_model = input_shape[-1]
        d_in_proj = 2*self.d_inner+2*self.d_state+self.nheads
        self.in_proj = layers.Dense(d_in_proj, use_bias=False, dtype=self._dtype)
        conv_dim = self.d_inner + 2*self.d_state
        self.conv1d = layers.Conv1D(conv_dim, self.k_conv, groups=conv_dim, padding='valid', activation='silu', dtype=self._dtype)
        self.dt_bias = self.add_weight(shape=(self.nheads,), dtype='float32', trainable=True)
        self.A_log = self.add_weight(shape=(self.nheads,), dtype='float32', trainable=True)
        self.D = self.add_weight(shape=(self.nheads, 1), dtype='float32', trainable=True)
        self.norm = RMSNorm()
        self.out_proj = layers.Dense(d_model, use_bias=False, dtype=self._dtype)
        self.call(ops.zeros(input_shape))
    
    def step_chunk(self, x, state):
        if state is None:
            conv_state = ops.zeros((x.shape[0], self.k_conv, self.d_inner+2*self.d_state))
            x, _ = self.step_chunk(x, [None, conv_state])
            return x, _
        x = ops.cast(x, self._dtype)
        A = -ops.exp(self.A_log)
        zxbcdt = self.in_proj(x)
        z, xBC, dt = ops.split(zxbcdt, [self.d_inner, 2*self.d_inner+2*self.d_state], axis=-1)
        dt = ops.softplus(dt+self.dt_bias)
        conv_state = ops.concatenate([state[1], xBC], axis=1)
        xBC = self.conv1d(conv_state)[:,-xBC.shape[1]:]
        x, B, C = ops.split(xBC, [self.d_inner, self.d_state+self.d_inner], axis=-1)
        x = rearrange(x, "b l (h p) -> b l h p", p=self.headdim)
        y, new_state = ssd(x*dt[...,None], A*dt, B[...,None,:], C[...,None,:], self.chunk_size, init_stat=state[0])
        y += x*self.D
        y = rearrange(y, "b l h p -> b l (h p)")
        y = self.norm(y, z)
        y = self.out_proj(y)
        return y, [new_state, conv_state[:,-self.k_conv:]]

    def call(self, x, **kw):
        x, _ = self.step_chunk(x, None)
        return x
    
    def init_state(self, input_shape):
        if not self.built:
            self.build((input_shape[0], self.chunk_size, input_shape[-1]))
        conv_state = ops.zeros((input_shape[0], self.k_conv, self.d_inner+2*self.d_state))
        ssd_state = ops.zeros((input_shape[0], 1, self.nheads, self.headdim, self.d_state))
        return [ssd_state, conv_state]
    
    def step(self, x, state, **kw):
        zxbcdt = self.in_proj(x[:,0])
        z, xBC, dt = ops.split(zxbcdt, [self.d_inner, 2*self.d_inner+2*self.d_state], axis=-1)
        
        conv_state = ops.concatenate([state[1][:,1:], xBC[:,None]], axis=1)
        xBC = self.conv1d(conv_state)[:,-1]
        
        x, B, C = ops.split(xBC, [self.d_inner, self.d_inner+self.d_state], axis=-1)
        A = -ops.exp(self.A_log)
        
        dt = ops.softplus(dt+self.dt_bias)
        dA = ops.exp(dt*A)
        x = rearrange(x, 'b (h p) -> b h p', p=self.headdim)
        #dBx = ops.einsum('bh, bn, bhp -> bhpn', dt, B, x)

        dt_exp = dt[...,None,None]
        B_exp = B[:,None,None]
        x_exp = x[...,None]
        dBx = dt_exp * B_exp * x_exp
        
        ssd_state = state[0][:,0]*dA[...,None,None] + dBx
        y = ops.einsum('bhpn, bn -> bhp', ssd_state, C)
        y += self.D*x
        y = rearrange(y, 'b h p -> b (h p)')
        y = self.norm(y, z)
        y = self.out_proj(y)
        
        return y[:,None], [ssd_state[:,None], conv_state]


#i = np.zeros((1, 160, 128))
#mamba = Mamba2(chunk_size=160,)
#mamba = jax.jit(mamba)
#mamba(i);

class TNM(layers.Layer):

    def __init__(self, frames=0, axis=0, eps=2e-6):
        super().__init__()
        self.axis = axis
        self.eps = eps
        def norm(x, training=None):
            if not frames:
                _frames = x.shape[axis]
            else:
                _frames = frames
            dtype = x.dtype
            x_ = ops.cast(x, 'float32')
            x_ = ops.reshape(x_, (*x.shape[:axis], -1, _frames, *x.shape[axis+1:]))
            xmean = mean = ops.mean(x_, axis=axis+1, keepdims=True)
            tshape = [1]*len(x_.shape)
            tshape[axis+1] = _frames
            t = ops.reshape(ops.linspace(0, 1, _frames), tshape)
            n = ops.sum((t-0.5)*(x_-mean), axis=axis+1, keepdims=True)
            d = ops.sum((t-0.5)**2, axis=axis+1, keepdims=True)
            i = mean - n/d * 0.5
            trend = n/d * t + i
            x_ = x_ - trend
            # mean = ops.mean(x_, axis=axis+1, keepdims=True)
            mean = 0
            std = (ops.mean(x2mean:=(x_-mean)**2, axis=axis+1, keepdims=True)+eps)**0.5
            x_ = (x_-mean)/std
            # x_ = layers.GaussianDropout(0.02)(x_, training=training)
            r = ops.reshape(x_, (*x.shape[:axis], -1, *x.shape[axis+1:])) 
            return ops.cast(r, dtype), {'xmean':ops.take(xmean, -1, axis=axis), 'x2mean':ops.take(ops.take(x2mean, -1, axis=axis), [0], axis=axis)}
        self.n = norm
    
    def step_chunk(self, x, *v):
        return self.n(x)

    def call(self, x, training=None):
        return self.n(x, training=training)[0]
    
    def init_state(self, input_shape):
        xmean = ops.mean(ops.zeros(input_shape), self.axis, True)
        return {'xmean':xmean, 'x2mean':xmean+1e-5}
    
    def step(self, x, state, dt=1/30, lbd=160/30/2):
        decay = ops.exp(-ops.log(2)/lbd*dt)
        xmean = (1-decay)*x + decay*state['xmean']
        x2mean = (1-decay)*(x-state['xmean'])**2 + decay*state['x2mean']
        std = (x2mean+self.eps)**0.5
        return ops.tanh((x-xmean)/std/3)*3, {'xmean':xmean, 'x2mean':x2mean}

class SSConv(layers.Layer):
    def __init__(self, filters=64, kernel=(3, 3), chunk=160, **kw):
        super().__init__(**kw) 
        self.k = filters
        self.chunk = chunk
        self.s = kernel
    
    def build(self, input_shape):
        super().build(input_shape)
        self.conv1 = keras.Sequential([layers.Conv3D(self.k, (1, *self.s), padding='same'), 
                                       layers.Dense(self.k*2, activation='relu')])
        self.proj = layers.Dense(self.k, use_bias=False)
        self.reduce = lambda x:ops.mean(x, (2, 3))
        self.conv2 = layers.Conv3D(self.k, (1, *self.s), padding='same', activation='relu')
        self.mamba = Mamba2(chunk_size=self.chunk, d_state=128, headdim=self.k, k_conv=12)
        self.drop = layers.GaussianDropout(0.5)
        #self.drop = lambda x,**y:x
        #self.fuse = layers.Dense(self.k, use_bias=False)
        self.tn = TNM(frames=self.chunk, axis=1)
        self.call(ops.zeros(input_shape))
    
    def step_chunk(self, x0, state):
        x0, tn_state = self.tn.step_chunk(x0)
        x1 = self.conv1(x0)
        x2 = self.reduce(x1)
        x2, mamba_state = self.mamba.step_chunk(x2, state=state['mamba'])
        x2 = self.proj(x2)
        x0 += x2[:,:,None,None]
        x1 = self.conv2(x0)
        return x1, {'tn':tn_state, 'mamba':mamba_state}

    def call(self, x0, training=None):
        x0 = self.tn(x0, training=training)
        x1 = self.conv1(x0)
        x2 = self.reduce(x1)
        x2 = self.drop(x2, training=training)
        x2 = self.mamba(x2)
        self.ssd_features = x2
        x2 = self.proj(x2)
        x0 += x2[:,:,None,None]
        x1 = self.conv2(x0)
        self.out_features = x1
        return x1
    
    def init_state(self, input_shape):
        return {'mamba':self.mamba.init_state((input_shape[0], self.chunk, self.k)),
                'tn':self.tn.init_state(input_shape)}
    
    def step(self, x0, state, **kw):
        x0, tn_state = self.tn.step(x0, state['tn'], **kw)
        x1 = self.conv1(x0)
        x2 = self.reduce(x1)
        x2, mamba_state = self.mamba.step(x2, state['mamba'])
        x2 = self.proj(x2)
        x0 += x2[:,:,None,None]
        x1 = self.conv2(x0)
        return x1, {'tn':tn_state, 'mamba':mamba_state}

class SSCBlock(layers.Layer):
    def __init__(self, n, filters=32, chunk=160, kernel=(3, 3), downsample=True, **kw):
        super().__init__(**kw)
        self.k = filters
        self.ssc = keras.Sequential([SSConv(filters, chunk=chunk, kernel=kernel) for _ in range(n)])
        self.in_proj = layers.Dense(filters, use_bias=False)
        self.ds = layers.Conv3D(filters, (1, 2, 2), (1, 2, 2), padding='same') if downsample else lambda x:x

    def build(self, input_shape):
        super().build(input_shape)
        self.call(ops.zeros(input_shape))

    def call(self, x, training=None):
        x1 = self.in_proj(x)
        x = self.ssc(x1, training=training)
        x += x1
        x = self.ds(x)
        return x

    def step_chunk(self, x, state):
        x = x1 = self.in_proj(x)
        new_state = []
        for n, layer in enumerate(self.ssc.layers):
            x, s = layer.step_chunk(x, state[n])
            new_state.append(s)
        x += x1
        x = self.ds(x)
        return x, new_state
    
    def init_state(self, input_shape):
        #print(input_shape)
        return [i.init_state((*input_shape[:-1], self.k)) for i in self.ssc.layers]
    
    def step(self, x, state, **kw):
        x = x1 = self.in_proj(x)
        new_state = []
        for n, layer in enumerate(self.ssc.layers):
            x, s = layer.step(x, state[n], **kw)
            new_state.append(s)
        x += x1
        x = self.ds(x)
        return x, new_state

class InfinitePulse(keras.Model):

    def __init__(self, n_layers=[2, 2, 2, 2], filters=[32]*4, chunk_size=160, **kw):
        super().__init__(**kw)
        self.network = keras.Sequential([SSCBlock(l, f, chunk_size) for l, f in zip(n_layers, filters)])
        self.out_proj = keras.Sequential([layers.Dense(32, use_bias=False), layers.GaussianDropout(0.5)])
        self.head = keras.Sequential([
            Mamba2(chunk_size=chunk_size, headdim=32),
            Mamba2(chunk_size=chunk_size, headdim=32),
            layers.Dense(1),
        ])
        self.n_layers = n_layers
        self.filters = filters

    def build(self, input_shape):
        super().build(input_shape)
        self.call(ops.zeros(input_shape))

    def init_state(self, input_shape):
        b, l, h, w, c = input_shape
        r = []
        for ly, f, layer in zip(self.n_layers, self.filters, self.network.layers):
            r.append(layer.init_state((b, l, h, w, f)))
            h, w = (h+1)//2, (w+1)//2
        for layer in self.head.layers:
            if isinstance(layer, (Mamba2, TNM)):
                r.append(layer.init_state((b, l, 32)))
        return r
    
    def call(self, x, training=None,**kw):
        x = self.network(x, training=training)
        #x = rearrange(x, 'b l h w c -> b l (h w c)')
        x = ops.mean(x, (2, 3))
        x = self.out_proj(x, training=training)
        # x = (x-ops.mean(x, -1, True))/(ops.std(x, -1, True)+1e-6)
        x = self.head(x, training=training)
        return x[...,0]

    def step_chunk(self, x, state):
        new_state = []
        for n, layer in enumerate(self.network.layers):
            x, s = layer.step_chunk(x, state[n])
            new_state.append(s)
        #x = rearrange(x, 'b l h w c -> b l (h w c)')
        x = ops.mean(x, (2, 3))
        x = self.out_proj(x)
        #x = (x-ops.mean(x, -1, True))/(ops.std(x, -1, True)+1e-6)
        for m, layer in enumerate(self.head.layers):
            if isinstance(layer, (Mamba2, TNM)):
                x, s = layer.step_chunk(x, state[n+m+1])
                new_state.append(s)
            else:
                x = layer(x)
        return x[...,0], new_state
        
    def step(self, x, state, **kw):
        new_state = []
        for n, layer in enumerate(self.network.layers):
            x, s = layer.step(x, state[n], **kw)
            new_state.append(s)
        #x = rearrange(x, 'b l h w c -> b l (h w c)')
        x = ops.mean(x, (2, 3))
        x = self.out_proj(x)
        #x = (x-ops.mean(x, -1, True))/(ops.std(x, -1, True)+1e-6)
        for m, layer in enumerate(self.head.layers):
            if isinstance(layer, (Mamba2, TNM)):
                x, s = layer.step(x, state[n+m+1], **kw)
                new_state.append(s)
            else:
                x = layer(x)
        return x[...,0], new_state

from functools import lru_cache
import pkg_resources

def load_ME(weight):
    model = InfinitePulse([2]*4)
    model.build((1, 160, 36, 36, 3))
    model.load_weights(weight)
    state_path = pkg_resources.resource_filename(
        'rppg', 
        'weights/state.pkl'
    )
    with open(state_path, 'rb') as f:
        state = pickle.load(f)
    return model, state

@lru_cache(maxsize=1)
def load_ME_chunk_rlap():
    weights_path = pkg_resources.resource_filename(
        'rppg','weights/ME.rlap.weights.h5')
    model, state = load_ME(weights_path)
    @jax.jit
    def call(x, state):
        y, state = model.step_chunk(x[None]/255., state)
        return {'bvp':y[0]}, state
    call(np.zeros((160,36,36,3), dtype='uint8'), state)
    return call, state, {'fps':30., 'input':(160, 36, 36, 3)}

@lru_cache(maxsize=1)
def load_ME_chunk_pure():
    weights_path = pkg_resources.resource_filename(
        'rppg','weights/ME.pure.weights.h5')
    model, state = load_ME(weights_path)
    @jax.jit
    def call(x, state):
        y, state = model.step_chunk(x[None]/255., state)
        return {'bvp':y[0]}, state
    call(np.zeros((160,36,36,3), dtype='uint8'), state)
    return call, state, {'fps':30., 'input':(160, 36, 36, 3)}

@lru_cache(maxsize=1)
def load_ME_rlap():
    weights_path = pkg_resources.resource_filename(
        'rppg','weights/ME.rlap.weights.h5')
    model, state = load_ME(weights_path)
    @jax.jit
    def call(x, state):
        y, state = model.step(x[None]/255., state)
        return {'bvp':y[0]}, state
    call(np.zeros((1,36,36,3), dtype='uint8'), state)
    return call, state, {'fps':30., 'input':(1, 36, 36, 3)}

@lru_cache(maxsize=1)
def load_ME_pure():
    weights_path = pkg_resources.resource_filename(
        'rppg','weights/ME.pure.weights.h5')
    model, state = load_ME(weights_path)
    @jax.jit
    def call(x, state):
        y, state = model.step(x[None]/255., state)
        return {'bvp':y[0]}, state
    call(np.zeros((1,36,36,3), dtype='uint8'), state)
    return call, state, {'fps':30., 'input':(1, 36, 36, 3)}


def bidirectional_selective_scan(u, delta, A, B, C, D):
    forward_out = selective_scan(u, delta, A, B[:, :, :A.shape[1]], C[:, :, :A.shape[1]], D)
    
    reversed_u = ops.flip(u, axis=1)
    reversed_delta = ops.flip(delta, axis=1)
    backward_out = selective_scan(reversed_u, reversed_delta, A, 
                                B[:, :, A.shape[1]:], C[:, :, A.shape[1]:], D)
    backward_out = ops.flip(backward_out, axis=1)
    
    return ops.concatenate([forward_out, backward_out], axis=-1)

def selective_scan(u, delta, A, B, C, D):
    dA = ops.einsum('bld,dn->bldn', delta, A)
    dB_u = ops.einsum('bld,bld,bln->bldn', delta, u, B)

    dA_cumsum = ops.pad(dA[:, 1:], [[0,0], [1,1], [0,0], [0,0]])[:, 1:, :, :]
    dA_cumsum = dA_cumsum[:,::-1]
    dA_cumsum = ops.cumsum(dA_cumsum, axis=1)
    dA_cumsum = ops.exp(dA_cumsum)
    dA_cumsum = dA_cumsum[:,::-1]

    x = dB_u * dA_cumsum
    x = ops.cumsum(x, axis=1)/(dA_cumsum + 1e-12)
    y = ops.einsum('bldn,bln->bld', x, C)
    return y + u * D

class BiMamba(layers.Layer):
    def __init__(self, d_states=16, d_conv=4, expand=2, **kwargs):
        super().__init__(**kwargs)
        self.model_states = d_states  
        self.d_conv = d_conv          
        self.expand = expand          
        self.internal_dim = None      
        self.delta_t_rank = None      

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.internal_dim = self.expand * input_dim
        self.delta_t_rank = input_dim // 16

        self.in_projection = layers.Dense(
            self.internal_dim * self.expand, 
            use_bias=False,
            dtype='float32'
        )
        
        self.conv1d = layers.Conv1D(
            filters=self.internal_dim,
            kernel_size=self.d_conv,
            groups=self.internal_dim,
            padding='same', 
            use_bias=True,
            dtype='float32'
        )

        self.x_projection = layers.Dense(
            self.delta_t_rank + 2 * self.model_states * self.expand,
            use_bias=False,
            dtype='float32'
        )

        self.delta_t_projection = layers.Dense(
            self.internal_dim,
            use_bias=True,
            dtype='float32'
        )

        self.A = self.add_weight(
            name='A_log',
            shape=(self.internal_dim, self.model_states),
            initializer=keras.initializers.RandomNormal(mean=0, stddev=0.02),
            trainable=True,
            dtype='float32'
        )
        self.D = self.add_weight(
            name='D',
            shape=(self.internal_dim,),
            initializer='ones',
            trainable=True,
            dtype='float32'
        )

        self.out_projection = layers.Dense(
            input_dim,
            use_bias=False,
            dtype='float32'
        )

    def call(self, inputs):
        batch_size, seq_len, _ = ops.shape(inputs)

        x_and_res = self.in_projection(inputs)
        x, res = ops.split(x_and_res, [self.internal_dim], axis=-1)
        res = ops.concatenate([res, res], axis=-1)

        x = self.conv1d(x)[:, :seq_len] 
        x = ops.swish(x) 
        ssm_out = self._ssm(x)
        y = ssm_out * ops.swish(res) 

        return self.out_projection(y)

    def _ssm(self, x):
        A = -ops.exp(self.A) 
        D = ops.cast(self.D, 'float32')

        x_dbl = self.x_projection(x)
        delta, BC = ops.split(x_dbl, [self.delta_t_rank], axis=-1)
        delta = ops.softplus(self.delta_t_projection(delta))

        B, C = ops.split(
            BC, 
            [self.model_states * self.expand], 
            axis=-1
        )

        return bidirectional_selective_scan(x, delta, A, B, C, D)

import keras
from keras import layers, ops
import math

class ChannelAttention3D(layers.Layer):
    def __init__(self, in_channels, reduction=2, **kwargs):
        super().__init__(**kwargs)
        self.reduction = reduction
        
        self.avg_pool = layers.GlobalAveragePooling3D(keepdims=True)
        self.max_pool = layers.GlobalMaxPooling3D(keepdims=True)
        
        self.fc = keras.Sequential([
            layers.Conv3D(in_channels // reduction, 1, use_bias=False),
            layers.ReLU(),
            layers.Conv3D(in_channels, 1, use_bias=False)
        ])
        self.sigmoid = layers.Activation("sigmoid")

    def call(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        attention = self.sigmoid(out)
        return x * attention

class LateralConnection(layers.Layer):
    def __init__(self, fast_channels=32, slow_channels=64, **kwargs):
        super().__init__(**kwargs)
        self.conv = keras.Sequential([
            layers.Conv3D(slow_channels, (3, 1, 1), strides=(2, 1, 1), padding="same"),
            layers.BatchNormalization(axis=-1),
            layers.ReLU()
        ])

    def call(self, slow_path, fast_path):
        fast_path = self.conv(fast_path)
        return fast_path + slow_path

class CDC(layers.Layer):
    def __init__(self, filters, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', theta=0.2, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.theta = theta

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=self.kernel_size + (input_shape[-1], self.filters)) # (D, H, W, I, O)
        super().build(input_shape)

    def call(self, inputs):
        if self.kernel_size[0] == 3:
            conv_out = ops.nn.conv(inputs, self.kernel, strides=self.strides, padding=self.padding.upper())
            tdc_kernel = ops.sum(self.kernel[ops.array([0, -1])], axis=(0, 1, 2), keepdims=True)
            diff_out = ops.nn.conv(inputs, tdc_kernel, strides=self.strides, padding=self.padding.upper())
            return conv_out - self.theta*diff_out
        else:
            return ops.nn.conv(inputs, self.kernel, strides=self.strides, padding=self.padding.upper())

class MambaLayer(layers.Layer):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.norm1 = layers.LayerNormalization(epsilon=1e-5)
        self.norm2 = layers.LayerNormalization(epsilon=1e-5)
        self.mamba = BiMamba(d_states=d_state, d_conv=d_conv, expand=expand)
        #self.mamba = lambda x:x
        self.drop_path = layers.Dropout(0.0)  # No drop path in inference

    def build(self, input_shape):
        #self.bn = layers.BatchNormalization(axis=-1)
        self.channel_att = ChannelAttention3D(input_shape[-1], reduction=2)
        #self.channel_att = lambda x:x

    def call(self, x, training=False):
        B = ops.shape(x)[0]
        input_shape = ops.shape(x)
        
        # Convert to channels_last format
        #x = ops.transpose(x, [0, 2, 3, 4, 1])  # [B, T, H, W, C]
        x_flat = ops.reshape(x, [B, -1, self.dim])
        
        # Mamba processing
        residual = x_flat
        x_norm = self.norm1(x_flat)
        x_mamba = self.mamba(x_norm)
        x_out = self.norm2(residual + self.drop_path(x_mamba))
        
        # Reshape back
        x_out = ops.reshape(x_out, input_shape)
        #x_out = ops.transpose(x_out, [0, 4, 1, 2, 3])  # Back to channels_first
        
        # Additional processing
        #x_out = self.bn(x_out)
        x_out = self.channel_att(x_out)
        return x_out

def conv_block(in_channels, out_channels, kernel_size, strides, padding, 
               bn=True, activation='relu'):
    layers_list = [
        layers.Conv3D(out_channels, kernel_size, strides=strides, padding=padding)
    ]
    if bn:
        layers_list.append(layers.BatchNormalization(axis=-1))
    if activation == 'relu':
        layers_list.append(layers.ReLU())
    elif activation == 'elu':
        layers_list.append(layers.ELU())
    return keras.Sequential(layers_list)

class PhysMamba(keras.Model):
    def __init__(self, theta=0.5, drop_rate1=0.25, drop_rate2=0.5, frames=128):
        super().__init__()
        
        # Convolutional Blocks
        self.conv_block1 = conv_block(3, 16, (1,5,5), (1,1,1), "same")
        self.conv_block2 = conv_block(16, 32, (3,3,3), (1,1,1), "same")
        self.conv_block3 = conv_block(32, 64, (3,3,3), (1,1,1), "same")
        self.conv_block4 = conv_block(64, 64, (4,1,1), (4,1,1), "valid")
        self.conv_block5 = conv_block(64, 32, (2,1,1), (2,1,1), "valid")
        self.conv_block6 = conv_block(32, 32, (3,1,1), (1,1,1), "same", activation='elu')

        # Stream Blocks
        self.block1 = self._build_block(64, theta)
        self.block2 = self._build_block(64, theta)
        self.block3 = self._build_block(64, theta)
        self.block4 = self._build_block(32, theta)
        self.block5 = self._build_block(32, theta)
        self.block6 = self._build_block(32, theta)

        # Fusion and Upsampling
        self.fuse_1 = LateralConnection()
        self.fuse_2 = LateralConnection()
        
        self.upsample1 = keras.Sequential([
            layers.UpSampling3D(size=(2,1,1)),
            layers.Conv3D(64, (3,1,1), padding="same"),
            layers.BatchNormalization(axis=-1),
            layers.ELU()
        ])
        
        self.upsample2 = keras.Sequential([
            layers.UpSampling3D(size=(2,1,1)),
            layers.Conv3D(48, (3,1,1), padding="same"),
            layers.BatchNormalization(axis=-1),
            layers.ELU()
        ])

        self.conv_last = layers.Conv3D(1, (1,1,1), padding="valid")
        self.poolspa = lambda x:ops.mean(x, axis=(-2, -3), keepdims=True)

    def _build_block(self, channels, theta):
        return keras.Sequential([
            CDC(channels, theta=theta),
            layers.BatchNormalization(axis=-1),
            layers.ReLU(),
            MambaLayer(dim=channels),
            ChannelAttention3D(channels, reduction=2)
        ])

    def call(self, inputs, training=False):
        # Input shape: [batch, channels, time, height, width]
        # x = ops.transpose(inputs, [0, 2, 3, 4, 1])  # Convert to channels_last
        x = inputs 
        x = self.conv_block1(x)
        x = layers.MaxPool3D((1,2,2), strides=(1,2,2))(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = layers.MaxPool3D((1,2,2), strides=(1,2,2))(x)

        # Process streams
        s_x = self.conv_block4(x)  # Slow stream
        f_x = self.conv_block5(x)  # Fast stream
        
        # Processing blocks
        s_x1 = self.block1(s_x)
        s_x1 = layers.MaxPool3D((1,2,2), strides=(1,2,2))(s_x1)
        
        f_x1 = self.block4(f_x)
        f_x1 = layers.MaxPool3D((1,2,2), strides=(1,2,2))(f_x1)

        s_x1 = self.fuse_1(s_x1, f_x1)

        s_x2 = self.block2(s_x1)
        s_x2 = layers.MaxPool3D((1,2,2), strides=(1,2,2))(s_x2)
        
        f_x2 = self.block5(f_x1)
        f_x2 = layers.MaxPool3D((1,2,2), strides=(1,2,2))(f_x2)
        
        s_x2 = self.fuse_2(s_x2, f_x2)

        s_x3 = self.block3(s_x2)
        s_x3 = self.upsample1(s_x3)
        
        f_x3 = self.block6(f_x2)
        f_x3 = self.conv_block6(f_x3)

        # Final processing
        x_fusion = ops.concatenate([f_x3, s_x3], axis=-1)
        x_final = self.upsample2(x_fusion)
        x_final = self.poolspa(x_final)
        x_final = self.conv_last(x_final)

        x = ops.reshape(x_final, (-1, ops.shape(inputs)[2]))
        x = (x-ops.mean(x, axis=1, keepdims=True))/(ops.std(x, axis=1, keepdims=True)+1e-5)
        return x

def load_PhysMamba(weight):
    model = PhysMamba()
    model(np.zeros((1, 128, 128, 128, 3)))
    model.load_weights(weight)
    return model, {}

@lru_cache(maxsize=1)
def load_PhysMamba_rlap():
    weights_path = pkg_resources.resource_filename(
        'rppg','weights/PhysMamba.rlap.weights.h5')
    model, state = load_PhysMamba(weights_path)
    @jax.jit
    def call(x, state):
        x = x[None]/255.
        x = x[:,1:]-x[:,:-1]
        x = (x-ops.mean(x, axis=(2, 3), keepdims=True))/(ops.std(x, axis=(2, 3), keepdims=True)+1e-6)
        x = jnp.concatenate([x, x[:,-1:]], axis=1)
        y = model(x)
        return {'bvp':y[0]}, state
    call(np.zeros((128,128,128,3), dtype='uint8'), state)
    return call, state, {'fps':30., 'input':(128, 128, 128, 3)}

@lru_cache(maxsize=1)
def load_PhysMamba_pure():
    weights_path = pkg_resources.resource_filename(
        'rppg','weights/PhysMamba.pure.weights.h5')
    model, state = load_PhysMamba(weights_path)
    @jax.jit
    def call(x, state):
        x = x[None]/255.
        x = x[:,1:]-x[:,:-1]
        x = (x-ops.mean(x, axis=(2, 3), keepdims=True))/(ops.std(x, axis=(2, 3), keepdims=True)+1e-6)
        x = jnp.concatenate([x, x[:,-1:]], axis=1)
        y = model(x)
        return {'bvp':y[0]}, state
    call(np.zeros((128,128,128,3), dtype='uint8'), state)
    return call, state, {'fps':30., 'input':(128, 128, 128, 3)}

def standardization_input(g):
    def f(cls, x, *a, **b):
        x = (x-ops.mean(x, (-4, -3, -2), True))/(ops.std(x, (-4,-3,-2), True)+1e-6)
        return g(cls, x, *a, **b)
    return f

def standardization_output(g):
    def f(*args, **kwargs):
        a = g(*args, **kwargs)
        if isinstance(a, tuple):
            a, b = a[0], a[1:]
            a = (a-ops.mean(a, 1, True))/(ops.std(a, 1, True)+1e-6)
            return (a,) + b
        else:
            return (a-ops.mean(a, 1, True))/(ops.std(a, 1, True)+1e-6)
    return f

def selective_scan(u, delta, A, B, C, D):
    dA = ops.einsum('bld,dn->bldn', delta, A) 
    dB_u = ops.einsum('bld,bld,bln->bldn', delta, u, B)

    dA_cumsum = ops.pad(
        dA[:, 1:], [[0, 0], [1, 1], [0, 0], [0, 0]])[:, 1:, :, :]

    dA_cumsum = dA_cumsum[:,::-1]

    dA_cumsum = ops.cumsum(dA_cumsum, axis=1)  

    dA_cumsum = ops.exp(dA_cumsum)  
    dA_cumsum = dA_cumsum[:,::-1]

    x = dB_u * dA_cumsum
    x = ops.cumsum(x, axis=1)/(dA_cumsum + 1e-12) 

    y = ops.einsum('bldn,bln->bld', x, C)

    return y + u * D 

class Mamba(layers.Layer):

    def build(self, shape):
        self.input_dims = shape[-1]
        self.internal_dim  = self.expand * shape[-1]
        self.delta_t_rank = shape[-1]//16
        self.in_projection = layers.Dense(
            self.internal_dim * self.expand, use_bias=False)
        self.conv1d = layers.Conv1D(
            filters=self.internal_dim,
            use_bias=True,
            kernel_size=self.d_conv,
            groups=self.internal_dim,
            padding='causal',
        )
        self.x_projection = layers.Dense(self.delta_t_rank + self.model_states * self.expand, use_bias=False,)
        self.delta_t_projection = layers.Dense(self.internal_dim, use_bias=True)

        self.A = repeat(
                ops.arange(1, self.model_states+1), 
                'n -> d n', d=self.internal_dim)

        self.A_log = keras.Variable(
                ops.log(self.A), 
                trainable=True,)

        self.D = keras.Variable(
                np.ones(self.internal_dim), 
                trainable=True,)

        self.out_projection = layers.Dense(
                shape[-1], 
                use_bias=False)
        
    def __init__(self, d_states=48, d_conv=4, expand=2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_states = d_states
        self.d_conv = d_conv
        self.expand = expand

    def call(self, x):
        (batch_size, seq_len, dimension) = x.shape

        x_and_res = self.in_projection(x)

        (x, res) = ops.split(x_and_res, 
                            [self.internal_dim], axis=-1)

        x = self.conv1d(x)[:, :seq_len]

        x = ops.swish(x)
        y = self.ssm(x)
        y = y * ops.swish(res)
        return self.out_projection(y)

    def ssm(self, x):
        (d_in, n) = self.A_log.shape

        A = -ops.exp(ops.cast(self.A_log, 'float32')) # shape -> (d_in, n)
        D = ops.cast(self.D, 'float32')

        x_dbl = self.x_projection(x) # shape -> (batch, seq_len, delta_t_rank + 2*n)

        (delta, B, C) = ops.split(
                x_dbl, 
                [self.delta_t_rank, self.delta_t_rank+n],
                axis=-1)

        delta = ops.softplus(self.delta_t_projection(delta)) # shape -> (batch, seq_len, model_input_dim)

        return selective_scan(x, delta, A, B, C, D)

class Fusion_Stem(layers.Layer):
    def __init__(self, apha=0.5, belta=0.5, dim=24):
        super().__init__()
        
        # Stem11 分支
        self.stem11 = keras.Sequential([
            layers.Conv2D(dim//2, 7, strides=2, padding='same'),
            layers.BatchNormalization(epsilon=1e-5, momentum=0.1),
            layers.ReLU(),
            layers.MaxPooling2D(2, strides=2)
        ])
        
        # Stem12 分支
        self.stem12 = keras.Sequential([
            layers.Conv2D(dim//2, 7, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D(2, strides=2)
        ])
        
        # Stem21 分支
        self.stem21 = keras.Sequential([
            layers.Conv2D(dim, 7, strides=1, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D(2, strides=2)
        ])
        
        # Stem22 分支
        self.stem22 = keras.Sequential([
            layers.Conv2D(dim, 7, strides=1, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D(2, strides=2)
        ])
        
        self.apha = apha
        self.belta = belta

    def call(self, x):

        N, D, H, W, C = x.shape
        
        x1 = ops.concatenate([x[:, :1], x[:, :1], x[:, :D-2]], axis=1)
        x2 = ops.concatenate([x[:, :1], x[:, :D-1]], axis=1)
        x3 = x
        x4 = ops.concatenate([x[:, 1:], x[:, D-1:]], axis=1)
        x5 = ops.concatenate([x[:, 2:], x[:, D-1:], x[:, D-1:]], axis=1)
        
        x_diff = ops.concatenate([x2-x1, x3-x2, x4-x3, x5-x4], axis=2)
        x_diff = ops.reshape(x_diff, (N*D, H, W, 12))
        
        x_diff = self.stem12(x_diff)
            
        x3 = ops.reshape(x3, (N*D, H, W, C))
        x = self.stem11(x3)
        
        x_path1 = self.apha * x + self.belta * x_diff
        x_path1 = self.stem21(x_path1)
        
        x_path2 = self.stem22(x_diff)
        
        x = self.apha * x_path1 + self.belta * x_path2
        return x

class Attention_mask(layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, x):
        xsum = ops.sum(x, (2, 3), True)
        B, L, H, W, C = x.shape
        return x / xsum * H * W * 0.5

class Frequencydomain_FFN(layers.Layer):

    def build(self, shape):
        self.dim = shape[-1] * self.mlp_ratio

        self.r = self.add_weight(shape=(self.dim, self.dim), initializer=initializers.RandomNormal(stddev=self.scale))
        self.i = self.add_weight(shape=(self.dim, self.dim), initializer=initializers.RandomNormal(stddev=self.scale))
        self.rb = self.add_weight(shape=(self.dim,), initializer=initializers.RandomNormal(stddev=self.scale))
        self.ib = self.add_weight(shape=(self.dim,), initializer=initializers.RandomNormal(stddev=self.scale))

        self.fc1 = keras.Sequential([
            layers.Conv1D(self.dim, 1, strides=1, padding='same', use_bias=False),
            layers.BatchNormalization(epsilon=1e-5, momentum=0.1),
            layers.ReLU()
        ])
        
        self.fc2 = keras.Sequential([
            layers.Conv1D(shape[-1], 1, strides=1, padding='same', use_bias=False),
            layers.BatchNormalization(epsilon=1e-5, momentum=0.1)
        ])
    
    def __init__(self, mlp_ratio=2):
        super().__init__()
        self.scale = 0.02
        self.mlp_ratio = mlp_ratio

    def call(self, x):

        x = self.fc1(x)

        B, L, C = x.shape
        x = ops.cast(x, 'float32')
        real, imag = ops.rfft(rearrange(x, "b l c -> b c l"))
        real, imag = [rearrange(x, "b c l -> b l c")/L**0.5 for x in (real, imag)]
        
        x_real = ops.relu(
            ops.einsum("bnc,cd->bnd", real, self.r) - 
            ops.einsum("bnc,cd->bnd", imag, self.i) + 
            self.rb
        )
        
        x_imag = ops.relu(
            ops.einsum("bnc,cd->bnd", imag, self.r) + 
            ops.einsum("bnc,cd->bnd", real, self.i) + 
            self.ib
        )
        
        x_real, x_imag = [rearrange(x, "b l c -> b c l")*L**0.5 for x in (x_real, x_imag)]

        x = ops.irfft((x_real, x_imag))
        x = ops.cast(x, "float32")
        x = rearrange(x, "b c l -> b l c")
        
        x = self.fc2(x)
        return x

def DropPath(drop_prob=0.):
    drop = layers.Dropout(drop_prob)
    def call(x, training=None):
        return x*drop(ops.full((x.shape[0], *[1]*(len(x.shape)-1)), 1.-drop_prob), training=training)
    return call

@partial(jax.jit, static_argnums=(1,2,3))
def scale_seg(x_o, s, tt, B):
    for i in range(1, s):
        for j in range(i):
            x_o = x_o.at[0:B, tt*i: tt*(i+1) , :].set(x_o[0:B, tt*i: tt*(i+1) , :] + x_o[B*(j+1):B*(j+2), tt*(i-j-1): tt*(i-j),:])
        x_o = x_o.at[0:B, tt*i: tt*(i+1) , :].set(x_o[0:B, tt*i: tt*(i+1) , :] / (i+1))
    return x_o

class Block_mamba(keras.Model):
    def __init__(self, mlp_ratio, drop_path=0.1, norm_layer=layers.LayerNormalization):
        super().__init__()
        self.mlp_ratio = mlp_ratio
        self.drop_path = drop_path
        self.norm = norm_layer

    def build(self, shape):
        
        class MambaLayer(keras.Layer):
            def __init__(self, d_state=48, d_conv=4, expand=2):
                super().__init__()
                self.norm = layers.LayerNormalization()
                self.mamba = Mamba(d_states=d_state, d_conv=d_conv, expand=expand)

            def call(self, x):
                B, L, C = x.shape
                x_norm = self.norm(x)
                x_mamba = self.mamba(x_norm)
                return x_mamba
            
        self.norm1 = self.norm()
        self.norm2 = self.norm()
        self.attn = MambaLayer()
        self.mlp = Frequencydomain_FFN(self.mlp_ratio)
        self.drop = DropPath(self.drop_path)
        return self.call(ops.zeros(shape))
        
    def call(self, x, training=None):
        B, D, C = x.shape
        path = 3
        segment = 2**(path-1)
        tt = D // segment
        x_r = ops.tile(x, [segment, 1, 1])
        x_o = x_r
        for i in range(1, segment):
            x_o = x_o.at[i*B:(i+1)*B,:D-i*tt,:].set(x_r[i*B:(i+1)*B,i*tt:,:])
        x_o = self.attn(x_o)
        #x_o = np.zeros(x_o.shape)
        for i in range(1, segment):
            for j in range(i):
                x_o = x_o.at[0:B, tt*i: tt*(i+1) , :].set(x_o[0:B, tt*i: tt*(i+1) , :] + x_o[B*(j+1):B*(j+2), tt*(i-j-1): tt*(i-j),:])
            x_o = x_o.at[0:B, tt*i: tt*(i+1) , :].set(x_o[0:B, tt*i: tt*(i+1) , :] / (i+1))
        #x_o = scale_seg(x_o, segment, tt, B)
        x = x + self.drop(self.norm1(x_o[0:B]), training=training)
        x = x + self.drop(self.mlp(self.norm2(x)), training=training)
        return x

class RhythmMamba(keras.Model):
    def __init__(self, depth=24, embed_dim=96, mlp_ratio=2, drop_path_rate=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.Fusion_Stem = Fusion_Stem(dim=embed_dim//4)
        self.attn_mask = Attention_mask()
        self.stem3 = keras.Sequential([
            layers.Conv3D(embed_dim, kernel_size=(2, 5, 5), strides=(2, 1, 1), padding='same'),
            layers.BatchNormalization(),
        ])
        dpr = [0.]+[float(x) for x in np.linspace(0, drop_path_rate, depth)]
        self.blocks = keras.Sequential([Block_mamba(mlp_ratio=mlp_ratio, drop_path=dpr[i], norm_layer=layers.LayerNormalization) for i in range(depth)])
        self.upsample = layers.UpSampling1D(2)
        self.ConvBlockLast = layers.Conv1D(1, 1, 1)
        
    @standardization_input
    @standardization_output
    def call(self, x, training=None):
        B, D, H, W, C = x.shape
        x = self.Fusion_Stem(x)
        x = rearrange(x, "(b l) h w c -> b l h w c", b=B)
        x = self.stem3(x)

        mask = ops.sigmoid(x)
        mask = self.attn_mask(mask)

        x = x*mask

        x = ops.mean(x, axis=(2, 3))

        x = self.blocks(x, training=training)
        #for l in self.blocks:
        #    x = l(x, training=training)

        rppg = self.upsample(x)
        rppg = self.ConvBlockLast(rppg)
        return rppg[...,0]
    
def load_RhythmMamba(weight):
    model = RhythmMamba()
    model(np.zeros((1, 160, 128, 128, 3)))
    model.load_weights(weight)
    return model, {}

@lru_cache(maxsize=1)
def load_RhythmMamba_rlap():
    weights_path = pkg_resources.resource_filename(
        'rppg','weights/RhythmMamba.rlap.weights.h5')
    model, state = load_RhythmMamba(weights_path)
    @jax.jit
    def call(x, state):
        x = x[None]/255.
        y = model(x)
        return {'bvp':y[0]}, state
    call(np.zeros((160,128,128,3), dtype='uint8'), state)
    return call, state, {'fps':30., 'input':(160, 128, 128, 3)}

@lru_cache(maxsize=1)
def load_RhythmMamba_pure():
    weights_path = pkg_resources.resource_filename(
        'rppg','weights/RhythmMamba.pure.weights.h5')
    model, state = load_RhythmMamba(weights_path)
    @jax.jit
    def call(x, state):
        x = x[None]/255.
        y = model(x)
        return {'bvp':y[0]}, state
    call(np.zeros((160,128,128,3), dtype='uint8'), state)
    return call, state, {'fps':30., 'input':(160, 128, 128, 3)}

class CDC(layers.Layer):
    def __init__(self, filters, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', theta=0.6, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.theta = theta

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=self.kernel_size + (input_shape[-1], self.filters)) # (D, H, W, I, O)
        super().build(input_shape)

    def call(self, inputs):
        if self.kernel_size[0] == 3:
            conv_out = ops.nn.conv(inputs, self.kernel, strides=self.strides, padding=self.padding.upper())
            tdc_kernel = ops.sum(self.kernel[ops.array([0, -1])], axis=(0, 1, 2), keepdims=True)
            diff_out = ops.nn.conv(inputs, tdc_kernel, strides=self.strides, padding=self.padding.upper())
            return conv_out - self.theta*diff_out
        else:
            return ops.nn.conv(inputs, self.kernel, strides=self.strides, padding=self.padding.upper())
        
class MultiHeadedSelfAttention_TDC_gra_sharp(keras.Layer):
    def __init__(self, ch, num_heads, dropout=0.1, theta=0.7):
        super().__init__()
        
        self.proj_q = keras.Sequential([
            CDC(ch, (3, 3, 3), theta=theta, padding='same'),  
            layers.BatchNormalization(),
        ])
        self.proj_k = keras.Sequential([
            CDC(ch, (3, 3, 3), theta=theta, padding='same'),  
            layers.BatchNormalization(),
        ])
        self.proj_v = keras.Sequential([
            layers.Conv3D(ch, (1, 1, 1), use_bias=False),
        ])
        
        self.drop = layers.Dropout(dropout)
        self.n_heads = num_heads

    def call(self, x, gra_sharp=2.0, training=None):
        B, P, C = x.shape # Batch, 640, 96
        x = ops.reshape(x, (B, P//16, 4, 4, C))
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q, k, v = [ops.transpose(layers.Reshape((P, self.n_heads, -1))(i), (0, 2, 1, 3)) for i in (q, k, v)] # Batch, Heads, 640, Channel//Heads
        scores = q @ ops.transpose(k, (0, 1, 3, 2)) / gra_sharp
        scores = ops.cast(scores, 'float32')
        scores = self.drop(ops.softmax(scores, axis=-1), training=training)
        h = layers.Reshape((P, -1))(ops.transpose(scores @ v, (0, 2, 1, 3)))
        
        return h, scores
    
class PositionWiseFeedForward_ST(keras.Layer):
    def __init__(self, ich, och):
        super().__init__()
        
        self.fc1 = keras.Sequential([
            layers.Conv3D(och, 1, use_bias=False),
            layers.BatchNormalization(),
            layers.Activation('elu')
        ])
        
        self.STConv = keras.Sequential([
            layers.Conv3D(och, 3, padding='same', groups=och, use_bias=False),
            layers.BatchNormalization(),
            layers.Activation('elu')
        ])
        
        self.fc2 = keras.Sequential([
            layers.Conv3D(ich, 1, use_bias=False),
            layers.BatchNormalization(),
        ])
        
    def call(self, x):
        B, P, C = x.shape
        x = ops.reshape(x, (B, P//16, 4, 4, C))
        x = self.fc1(x)
        x = self.STConv(x)
        x = self.fc2(x)
        x = ops.reshape(x, (B, P, C))
        
        return x

class Block_ST_TDC_gra_sharp(keras.Layer):
    def __init__(self, num_heads, ich, och, dropout=0.1, theta=0.7):
        super().__init__()
        self.attn = MultiHeadedSelfAttention_TDC_gra_sharp(ich, num_heads, dropout, theta)
        self.proj = layers.Dense(ich)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.pwff = PositionWiseFeedForward_ST(ich, och)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.drop = layers.Dropout(dropout)
        
    def call(self, x, gra_sharp=2., training=None):
        attn, score = self.attn(self.norm1(x), gra_sharp=gra_sharp, training=training)
        h = self.drop(self.proj(attn), training=training)
        x = x + h
        h = self.drop(self.pwff(self.norm2(x)), training=training)
        x = x + h
        
        return x
    
class Transformer_ST_TDC_gra_sharp(keras.Layer):
    def __init__(self, num_layers, num_heads, ich, och, dropout=0.1, theta=0.7):
        super().__init__()
        self.blocks = [Block_ST_TDC_gra_sharp(num_heads, ich, och, dropout, theta) for _ in range(num_layers)]
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def call(self, x, training=None, gra_sharp=2.):
        for i in self.blocks:
            x = i(x, gra_sharp=gra_sharp, training=training)
        return x
    

class PhysFormer(keras.Model):
    def __init__(self, patches=(4, 4, 4), conv_ch=96, ff_ch=144, num_heads=12, num_layers=12, dropout_rate=0.1, theta=0.7, TN=False):
        super().__init__()
        ft, fh, fw = patches
        
        self.patch_embedding = layers.Conv3D(conv_ch, kernel_size=(ft, fh, fw), strides=(ft, fh, fw))
        self.transformers = [Transformer_ST_TDC_gra_sharp(num_layers=num_layers//3, num_heads=num_heads, ich=conv_ch, och=ff_ch, dropout=dropout_rate, theta=theta) for _ in range(3)]
        self.stem0 = keras.Sequential([
            layers.Conv3D(conv_ch//4, (1, 5, 5), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPool3D((1, 2, 2), (1, 2, 2))
        ])
        
        self.stem1 = keras.Sequential([
            layers.Conv3D(conv_ch//2, (3, 3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPool3D((1, 2, 2), (1, 2, 2))
        ])
        
        self.stem2 = keras.Sequential([
            layers.Conv3D(conv_ch, (3, 3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPool3D((1, 2, 2), (1, 2, 2))
        ])
        self.unsample1 = keras.Sequential([
            layers.UpSampling3D((2, 1, 1)),
            layers.Conv3D(conv_ch, (3, 1, 1), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('elu')
        ])
        self.unsample2 = keras.Sequential([
            layers.UpSampling3D((2, 1, 1)),
            layers.Conv3D(conv_ch//2, (3, 1, 1), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('elu')
        ])
        self.ConvBlockLast = layers.Conv1D(1, 1, dtype='float32')
        
        self.tn = TN

    @standardization_input
    @standardization_output
    def call(self, x, gra_sharp=2., training=None):
        b, t, fh, fw, c = x.shape
        
        x = self.stem0(x, training=training)
        x = self.stem1(x, training=training)
        x = self.stem2(x, training=training) 
        
        x = self.patch_embedding(x) 
        
        x = layers.Reshape((-1, x.shape[-1]))(x) 
        
        x = self.transformers[0](x, gra_sharp=gra_sharp, training=training)
        
        x = layers.Reshape((t//4, 4, 4, -1))(x)
        x = layers.Reshape((-1, x.shape[-1]))(x)
        x = self.transformers[1](x, gra_sharp=gra_sharp, training=training)
        
        x = layers.Reshape((t//4, 4, 4, -1))(x)
        x = layers.Reshape((-1, x.shape[-1]))(x)
        x = self.transformers[2](x, gra_sharp=gra_sharp, training=training) 
        
        x = layers.Reshape((t//4, 4, 4, -1))(x) # (B, 40, 4, 4, 96)
        
        x = self.unsample1(x, training=training)
        x = self.unsample2(x, training=training) # (B, 160, 4, 4, 48)
        
        x = ops.mean(x, axis=(2, 3))
        
        x = self.ConvBlockLast(x)[...,0]
        
        return x
    
def load_PhysFormer(weight):
    model = PhysFormer()
    model(np.zeros((1, 160, 128, 128, 3)))
    model.load_weights(weight)
    return model, {}

@lru_cache(maxsize=1)
def load_PhysFormer_rlap():
    weights_path = pkg_resources.resource_filename(
        'rppg','weights/PhysFormer.rlap.weights.h5')
    model, state = load_PhysFormer(weights_path)
    @jax.jit
    def call(x, state):
        x = x[None]/255.
        y = model(x)
        return {'bvp':y[0]}, state
    call(np.zeros((160,128,128,3), dtype='uint8'), state)
    return call, state, {'fps':30., 'input':(160, 128, 128, 3)}

@lru_cache(maxsize=1)
def load_PhysFormer_pure():
    weights_path = pkg_resources.resource_filename(
        'rppg','weights/PhysFormer.pure.weights.h5')
    model, state = load_PhysFormer(weights_path)
    @jax.jit
    def call(x, state):
        x = x[None]/255.
        y = model(x)
        return {'bvp':y[0]}, state
    call(np.zeros((160,128,128,3), dtype='uint8'), state)
    return call, state, {'fps':30., 'input':(160, 128, 128, 3)}

class TSM(layers.Layer):
    def __init__(self, n, frames=160):
        super().__init__()
        self.n = n
        self.frames = frames

    def call(self, x, fold_div=3):
        if self.n==0:
            return x
        shape = x.shape
        x = ops.reshape(x, (-1, self.frames, *x.shape[1:]))
        b, nt, *c = x.shape
        x = ops.reshape(x, (b, -1, self.n, *c))
        fold = c[-1] // fold_div
        out = ops.concatenate([ops.concatenate([x[:, :, 1:, ..., :fold], x[:, :, -1:, ..., :fold]], axis=2),
                        ops.concatenate([x[:, :, :1, ..., fold:fold*2], x[:, :, :-1, ..., fold:fold*2]], axis=2),
                        x[:, :, :, ..., fold*2:]], axis=-1)
        return ops.reshape(out, shape)

class Attention_mask(layers.Layer):
    
    def call(self, x):
        return 0.5*x/ops.mean(x, axis=(1, 2), keepdims=True)

class TSCAN(keras.Model):

    def __init__(self, input_frames=160):
        super().__init__()
        n = input_frames//8
        self.d1 = keras.Sequential([
            TSM(n),
            layers.Conv2D(32, (3, 3), padding='same', activation='tanh'),
            TSM(n),
            layers.Conv2D(32, (3, 3), activation='tanh'),
        ])
        self.d2 = keras.Sequential([
            TSM(n),
            layers.Conv2D(64, (3, 3), padding='same', activation='tanh'),
            TSM(n),
            layers.Conv2D(64, (3, 3), activation='tanh'),
        ])
        self.g1 = keras.Sequential([
            layers.Conv2D(32, (3, 3), padding='same', activation='tanh'),
            layers.Conv2D(32, (3, 3), activation='tanh'),
        ])
        self.g2 = keras.Sequential([
            layers.Conv2D(64, (3, 3), padding='same', activation='tanh'),
            layers.Conv2D(64, (3, 3), activation='tanh'),
        ])
        self.attn1 = keras.Sequential([
            layers.Conv2D(1, (1, 1), activation='sigmoid'),
            Attention_mask()
        ])
        self.attn2 = keras.Sequential([
            layers.Conv2D(1, (1, 1), activation='sigmoid'),
            Attention_mask()
        ])
        self.pd = keras.Sequential([
            layers.AveragePooling2D((2, 2)),
            layers.Dropout(0.25)
        ])
        self.d = keras.Sequential([
            layers.Flatten(),
            layers.Dense(128, activation='tanh'),
            layers.Dropout(0.5),
            layers.Dense(1)
        ])

    def call(self, x, training=None, return_attn=False):
        d, g = x 
        b = d.shape[0]
        d = ops.reshape(d, (-1, *d.shape[2:]))
        g = ops.reshape(g, (-1, *g.shape[2:]))
        d = self.d1(d)
        g = self.g1(g)
        attn1 = self.attn1(g)
        d = ops.stack(ops.split(d, b))*ops.stack(ops.split(self.attn1(g), b))
        d = ops.reshape(d, (-1, *d.shape[2:]))
        d = self.pd(d, training=training)
        d = self.d2(d)
        g = self.pd(g, training=training)
        g = self.g2(g)
        attn2 = self.attn2(g)
        d = ops.stack(ops.split(d, b))*ops.stack(ops.split(self.attn2(g), b))
        d = ops.reshape(d, (-1, *d.shape[2:]))
        d = self.pd(d, training=training)
        d = self.d(d, training=training)
        if return_attn:
            return ops.reshape(d, (-1, x[0].shape[1])), attn1, attn2
        return ops.reshape(d, (-1, x[0].shape[1]))
    
class TSCANToEnd(keras.Model):
    
    def __init__(self, model):
        super().__init__()
        self.inner = model
    
    def call(self, x, training=None):
        x_ = x[:, 1:] - x[:, :-1]
        x_ = (x_ - ops.mean(x_, axis=(2,3 ), keepdims=True))/(ops.std(x_, axis=(2, 3), keepdims=True)+1e-6)
        return self.inner((ops.concatenate([x_, x_[:, -1:]], axis=1), ops.mean(x, axis=(1, ), keepdims=True)), training=training)

def load_TSCAN(weight):
    model = TSCANToEnd(TSCAN())
    model(np.zeros((1, 160, 36, 36, 3)))
    model.load_weights(weight)
    return model, {}

@lru_cache(maxsize=1)
def load_TSCAN_rlap():
    weights_path = pkg_resources.resource_filename(
        'rppg','weights/TSCAN.rlap.weights.h5')
    model, state = load_TSCAN(weights_path)
    @jax.jit
    def call(x, state):
        x = x[None]/255.
        y = model(x)[0]
        y = jnp.cumsum(y)
        y = (y-jnp.mean(y))/(jnp.std(y)+1e-6)
        y = jnp.diff(jnp.concat([jnp.array([0]), y], axis=0))
        return {'bvp':y}, state
    return call, state, {'fps':30., 'input':(160, 36, 36, 3), 'cumsum_output':True}

@lru_cache(maxsize=1)
def load_TSCAN_pure():
    weights_path = pkg_resources.resource_filename(
        'rppg','weights/TSCAN.pure.weights.h5')
    model, state = load_TSCAN(weights_path)
    @jax.jit
    def call(x, state):
        x = x[None]/255.
        y = model(x)[0]
        y = jnp.cumsum(y)
        y = (y-jnp.mean(y))/(jnp.std(y)+1e-6)
        y = jnp.diff(jnp.concat([jnp.array([0]), y], axis=0))
        return {'bvp':y}, state
    return call, state, {'fps':30., 'input':(160, 36, 36, 3), 'cumsum_output':True}

class PhysNet(keras.Model):

    def __init__(self):
        super().__init__()
        self.ConvBlock1 = keras.Sequential([
            layers.Conv3D(16, kernel_size=(1, 5, 5), strides=1, padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu')
        ])
        self.ConvBlock2 = keras.Sequential([
            layers.Conv3D(32, kernel_size=(3, 3, 3), strides=1, padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu')
        ])
        self.ConvBlock3 = keras.Sequential([
            layers.Conv3D(64, kernel_size=(3, 3, 3), strides=1, padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu')
        ])
        self.ConvBlock4 = keras.Sequential([
            layers.Conv3D(64, kernel_size=(3, 3, 3), strides=1, padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu')
        ])
        self.ConvBlock5 = keras.Sequential([
            layers.Conv3D(64, kernel_size=(3, 3, 3), strides=1, padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu')
        ])
        self.ConvBlock6 = keras.Sequential([
            layers.Conv3D(64, kernel_size=(3, 3, 3), strides=1, padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu')
        ])
        self.ConvBlock7 = keras.Sequential([
            layers.Conv3D(64, kernel_size=(3, 3, 3), strides=1, padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu')
        ])
        self.ConvBlock8 = keras.Sequential([
            layers.Conv3D(64, kernel_size=(3, 3, 3), strides=1, padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu')
        ])
        self.ConvBlock9 = keras.Sequential([
            layers.Conv3D(64, kernel_size=(3, 3, 3), strides=1, padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu')
        ])
        self.upsample = keras.Sequential([
            layers.Conv3DTranspose(64, kernel_size=(4, 1, 1), strides=(2, 1, 1), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('elu')
        ])
        self.upsample2 = keras.Sequential([
            layers.Conv3DTranspose(64, kernel_size=(4, 1, 1), strides=(2, 1, 1), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('elu')
        ])
        self.convBlock10 = layers.Conv3D(1, kernel_size=(1, 1, 1), strides=1)
        self.MaxpoolSpa = layers.MaxPool3D((1, 2, 2), strides=(1, 2, 2))
        self.MaxpoolSpaTem = layers.MaxPool3D((2, 2, 2), strides=2)
        self.poolspa = layers.AvgPool3D((1, 2, 2))
        self.flatten = layers.Reshape((-1,))

    @standardization_input
    @standardization_output
    def call(self, x, training=None):

        x = self.ConvBlock1(x, training=training)
        x = self.MaxpoolSpa(x)

        x = self.ConvBlock2(x, training=training)

        x = self.ConvBlock3(x, training=training)
        x = self.MaxpoolSpaTem(x)

        x = self.ConvBlock4(x, training=training)

        x = self.ConvBlock5(x, training=training)
        x = self.MaxpoolSpaTem(x)

        x = self.ConvBlock6(x, training=training)

        x = self.ConvBlock7(x, training=training)
        x = self.MaxpoolSpa(x)

        x = self.ConvBlock8(x, training=training)

        x = self.ConvBlock9(x, training=training)
        x = self.upsample(x, training=training)
        x = self.upsample2(x, training=training)
        x = self.poolspa(x)
        x = self.convBlock10(x, training=training)
        x = self.flatten(x)
        x = (x-ops.mean(x, axis=1, keepdims=True))/ops.std(x, axis=1, keepdims=True)
        return x
    
def load_PhysNet(weight):
    model = PhysNet()
    model(np.zeros((1, 128, 32, 32, 3)))
    model.load_weights(weight)
    return model, {}

@lru_cache(maxsize=1)
def load_PhysNet_rlap():
    weights_path = pkg_resources.resource_filename(
        'rppg','weights/physnet.rlap.weights.h5')
    model, state = load_PhysNet(weights_path)
    @jax.jit
    def call(x, state):
        x = x[None]/255.
        y = model(x)
        return {'bvp':y[0]}, state
    call(np.zeros((128,32,32,3), dtype='uint8'), state)
    return call, state, {'fps':30., 'input':(128, 32, 32, 3)}

@lru_cache(maxsize=1)
def load_PhysNet_pure():
    weights_path = pkg_resources.resource_filename(
        'rppg','weights/physnet.pure.weights.h5')
    model, state = load_PhysNet(weights_path)
    @jax.jit
    def call(x, state):
        x = x[None]/255.
        y = model(x)
        return {'bvp':y[0]}, state
    call(np.zeros((128,32,32,3), dtype='uint8'), state)
    return call, state, {'fps':30., 'input':(128, 32, 32, 3)}

class EfficientPhys(keras.Model):
    def __init__(self, input_frames=160):
        super().__init__()
        self.TSM = TSM(8, input_frames)
        self.mc1 = layers.Conv2D(32, kernel_size=3, padding='same', activation='tanh')
        self.mc2 = layers.Conv2D(32, kernel_size=3, activation='tanh')
        self.mc3 = layers.Conv2D(64, kernel_size=3, padding='same', activation='tanh')
        self.mc4 = layers.Conv2D(64, kernel_size=3, activation='tanh')
        self.attc1 = layers.Conv2D(32, kernel_size=1, activation='sigmoid')
        self.msk = Attention_mask()
        self.attc2 = layers.Conv2D(64, kernel_size=1, activation='sigmoid')
        self.avgp = layers.AvgPool2D(2)
        self.dp1 = layers.Dropout(0.25)
        self.dp2 = layers.Dropout(0.5)
        self.dense1 = layers.Dense(128, activation='tanh')
        self.dense2 = layers.Dense(1)
        self.bn = layers.BatchNormalization(axis=1)
        self.ft = layers.Flatten()

    def call(self, y, training=None, return_attn=False):
        x = ops.reshape(y, (-1, *y.shape[2:]))
        x = x[1:]-x[:-1]
        x = ops.concatenate([x, ops.zeros((1, *x.shape[-3:]))], axis=0)
        x = ops.reshape(x, (x.shape[0], -1, x.shape[3]))
        x = self.bn(x, training=training)
        x = ops.reshape(x, (-1, *y.shape[2:]))
        x = self.TSM(x)
        
        x = self.mc1(x)
        x = self.TSM(x)
        
        x = self.mc2(x)
        msk1 = self.msk(self.attc1(x))
        x = layers.multiply([x, msk1])
        x = self.avgp(x)
        x = self.dp1(x, training=training)
        x = self.TSM(x)
        
        x = self.mc3(x)
        x = self.TSM(x)
        
        x = self.mc4(x)
        msk2 = self.msk(self.attc2(x))
        x = layers.multiply([x, msk2])
        x = self.avgp(x)
        x = self.dp1(x, training=training)
        #x = ops.reshape(x, (-1, ops.reduce_prod(x.get_shape()[1:])))
        x = self.ft(x)
        x = self.dense1(x)
        x = self.dp2(x, training=training)
        x = self.dense2(x)
        if return_attn:
            return ops.reshape(x, (-1, y.shape[1])), msk1, msk2
        return ops.reshape(x, (-1, y.shape[1]))
    
    
def load_EfficientPhys(weight):
    model = EfficientPhys()
    model(np.zeros((1, 160, 72, 72, 3)))
    model.load_weights(weight)
    return model, {}

@lru_cache(maxsize=1)
def load_EfficientPhys_rlap():
    weights_path = pkg_resources.resource_filename(
        'rppg','weights/EfficientPhys.rlap.weights.h5')
    model, state = load_EfficientPhys(weights_path)
    @jax.jit
    def call(x, state):
        x = x[None]/255.
        y = model(x)[0]
        y = jnp.cumsum(y)
        y = (y-jnp.mean(y))/(jnp.std(y)+1e-6)
        y = jnp.diff(jnp.concat([jnp.array([0]), y], axis=0))
        return {'bvp':y}, state
    call(np.zeros((160,72,72,3), dtype='uint8'), state)
    return call, state, {'fps':30., 'input':(160, 72, 72, 3), 'cumsum_output':True}

@lru_cache(maxsize=1)
def load_EfficientPhys_pure():
    weights_path = pkg_resources.resource_filename(
        'rppg','weights/EfficientPhys.pure.weights.h5')
    model, state = load_EfficientPhys(weights_path)
    @jax.jit
    def call(x, state):
        x = x[None]/255.
        y = model(x)[0]
        y = jnp.cumsum(y)
        y = (y-jnp.mean(y))/(jnp.std(y)+1e-6)
        y = jnp.diff(jnp.concat([jnp.array([0]), y], axis=0))
        return {'bvp':y}, state
    call(np.zeros((160,72,72,3), dtype='uint8'), state)
    return call, state, {'fps':30., 'input':(160, 72, 72, 3), 'cumsum_output':True}
    
from .models_code.FacePhys import FacePhys
    
def load_FacePhys(weight):
    model = FacePhys([2]*4, [32]*4)
    model.build((1, 1, 36, 36, 3))
    model.load_weights(weight)
    state = model.init_state((1, 1, 36, 36, 3))
    return model, state

@lru_cache(maxsize=1)
def load_FacePhys_rlap():
    weights_path = pkg_resources.resource_filename(
        'rppg','weights/FacePhys.rlap.weights.h5')
    model, state = load_FacePhys(weights_path)
    @jax.jit
    def call(x, state, dt=1/30):
        y, state = model.step(x[None]/255., state, dt=dt)
        return {'bvp':y[0]}, state
    _, s = call(np.zeros((1,36,36,3), dtype='uint8'), state)
    _, s = call(np.zeros((1,36,36,3), dtype='uint8'), s)
    return call, state, {'fps':30., 'input':(1, 36, 36, 3)}