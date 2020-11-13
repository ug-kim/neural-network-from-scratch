import numpy as np


def linear(x, w, b):
    N = x.shape[0]
    x_reshape = x.reshape(N, -1)
    out = np.dot(x_reshape, w) + b
    cache = (x, w, b)

    return out, cache


def dlinear(dout, cache):
    x, w, _ = cache
    N = x.shape[0]
    x_reshape = x.reshape(N, -1)
    dx = np.dot(dout, w.T)
    dx = dx.reshape(*x.shape)
    dw = np.dot(x_reshape.T, dout)
    db = np.sum(dout, axis=0)

    return dx, dw, db


def relu(x):
    out = np.maximum(x, 0)
    return out, x


def drelu(dout, cache):
    dx = dout * (cache > 0)
    return dx


def softmax(x, y):
    softmax = np.exp(x)
    softmax = (softmax.T) / (np.sum(softmax, axis=1))
    softmax = softmax.T
    N = x.shape[0]
    loss = 0.0
    temp = softmax[np.arange(N), y]
    loss -= np.sum(np.log(temp))
    loss /= N

    dscores = np.copy(softmax)
    dscores[np.arange(N), y] -= 1
    dscores /= N

    return loss, dscores


def conv(x, w, b, conv_param):
    N, C, H, W = x.shape  # N: batch_size, C: channel_size, H: height, W: width
    # K: num_of_filters, F: filter height (same with width)
    K, _, F, _ = w.shape
    stride = conv_param['stride']
    pad = conv_param['pad']

    H_out = int((H - F + 2*pad) / stride + 1)
    W_out = int((W - F + 2*pad) / stride + 1)

    out = np.zeros((N, K, H_out, W_out))
    x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad),
                       (pad, pad)), mode='constant')

    for i in range(H_out):
        for j in range(W_out):
            patch = x_pad[:, :, (i*stride):(i*stride + F), (j*stride):(j*stride + F)]
            for k in range(K):
                out[:, k , i, j] = np.sum(patch * w[k, :, :, :], axis=(1,2,3))
    

    out = out + (b)[None, :, None, None]

    cache = (x, w, b, conv_param)
    return out, cache


def dconv(dout, cache):
    x, w, b, conv_param = cache

    _, _, H_out, W_out = dout.shape
    N, C, H, W = x.shape
    K, _, F, _ = w.shape
    stride = conv_param['stride']
    pad = conv_param['pad']

    x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad),
                    (pad, pad)), mode='constant')


    dw, db, dx_pad, dx = np.zeros_like(w), np.zeros_like(b), np.zeros_like(x_pad), np.zeros_like(x)

    db = np.sum(dout, axis=(0, 2, 3))

    for i in range(H_out):
        for j in range(W_out):
            patch = x_pad[:, :, (i*stride):(i*stride + F), (j*stride):(j*stride + F)]
            for k in range(K):
                dw[k, :, :, :] += np.sum(patch * (dout[:, k, i, j])[:, None, None, None], axis=0)
            for n in range(N):
                dx_pad[n, :, (i*stride):(i*stride + F), (j*stride):(j*stride + F)] += np.sum((w[:, :, :, :] * (dout[n, :, i, j])[:, None, None, None]), axis=0)

    dx = dx_pad[:, :, pad:-pad, pad:-pad]

    return dx, dw, db


def max_pool(x, pool_param):
    N, C, H, W = x.shape
    H_p = pool_param['pool_height']
    W_p = pool_param['pool_width']
    stride = pool_param['stride']

    H_out = (H - H_p) // stride + 1
    W_out = (W - W_p) // stride + 1
    out = np.zeros((N, C, H_out, W_out))

    for i in range(H_out):
        for j in range(W_out):
            patch = x[:, :, (i*stride):(i*stride + H_p),
                      (j*stride):(j*stride + W_p)]
            out[:, :, i, j] = np.max(patch, axis=(2, 3))

    cache = (x, pool_param)
    return out, cache


def dmax_pool(dout, cache):
    x, pool_param = cache
    N, C, H, W = x.shape
    H_p = pool_param['pool_height']
    W_p = pool_param['pool_width']
    stride = pool_param['stride']
    H_out = (H - H_p) // stride + 1
    W_out = (W - W_p) // stride + 1
    dx = np.zeros_like(x)

    for i in range(H_out):
        for j in range(W_out):
            patch = x[:, :, (i*stride):(i*stride + H_p),
                      (j*stride):(j*stride + W_p)]
            max_patch = np.max(patch, axis=(2, 3))
            temp = (patch == (max_patch)[:, :, None, None])
            dx[:, :, (i*stride):(i*stride + H_p), (j*stride):(j*stride + W_p)
               ] += temp * (dout[:, :, i, j])[:, :, None, None]

    return dx


def conv_relu_pool(x, w, b, conv_param, pool_param):
    a, conv_cache = conv(x, w, b, conv_param)
    s, relu_cache = relu(a)
    out, pool_cache = max_pool(s, pool_param)
    return out, (conv_cache, relu_cache, pool_cache)


def dconv_relu_pool(dout, cache):
    conv_cache, relu_cache, pool_cache = cache
    ds = dmax_pool(dout, pool_cache)
    da = drelu(ds, relu_cache)
    dx, dw, db = dconv(da, conv_cache)

    return dx, dw, db
