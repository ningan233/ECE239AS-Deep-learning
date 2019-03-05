import numpy as np
from nndl.layers import *
import pdb

""" 
This code was originally written for CS 231n at Stanford University
(cs231n.stanford.edu).  It has been modified in various areas for use in the
ECE 239AS class at UCLA.  This includes the descriptions of what code to
implement as well as some slight potential changes in variable names to be
consistent with class nomenclature.  We thank Justin Johnson & Serena Yeung for
permission to use this code.  To see the original version, please visit
cs231n.stanford.edu.  
"""

def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  pad = conv_param['pad']
  stride = conv_param['stride']

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the forward pass of a convolutional neural network.
  #   Store the output as 'out'.
  #   Hint: to pad the array, you can use the function np.pad.
  # ================================================================ #
  N, C, H, W = x.shape
  F, C, HH, WW = w.shape
  
  xpad = np.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)), mode='constant')
  
  H_prime = int((H + 2 * pad - HH) / stride + 1)
  W_prime = int((W + 2 * pad - WW) / stride + 1)
  
  out = np.empty([N,F,H_prime,W_prime])
  
  for n in range(N):
      for f in range(F):
          for m in range(H_prime):
              for j in range(W_prime):
                  out[n,f,m,j] = np.sum(w[f,:,:,:] * xpad[n,:,m*stride:m*stride+HH,j*stride:j*stride+WW]) + b[f]

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ #
    
  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None

  N, F, out_height, out_width = dout.shape
  x, w, b, conv_param = cache
  
  stride, pad = [conv_param['stride'], conv_param['pad']]
  xpad = np.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)), mode='constant')
  num_filts, _, f_height, f_width = w.shape

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the backward pass of a convolutional neural network.
  #   Calculate the gradients: dx, dw, and db.
  # ================================================================ #
  dw = np.zeros_like(w)
  db = np.zeros_like(b)
  dx = np.zeros_like(x)
  dxpad = np.zeros_like(xpad)
  
  F, C, HH, WW = w.shape
  N, C, H, W = x.shape
  H_prime = int((H + 2 * pad - HH) / stride + 1)
  W_prime = int((W + 2 * pad - WW) / stride + 1)

  for n in range(N):
      for f in range(F):
          for i in range(H_prime):
              for j in range(W_prime):
                  dw[f,:,:,:] += xpad[n,:,i*stride:i*stride+HH,j*stride:j*stride+WW] * dout[n,f,i,j]
                  db[f] += dout[n,f,i,j]
                  dxpad[n,:,i*stride:i*stride+HH,j*stride:j*stride+WW] += w[f,:,:,:] * dout[n,f,i,j]

  dx[:] = dxpad[:,:,pad:-pad,pad:-pad]

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ #

  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  
  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the max pooling forward pass.
  # ================================================================ #
  pool_h = pool_param['pool_height']
  pool_w = pool_param['pool_width']
  stride = pool_param['stride']
  
  N, C, H, W = np.shape(x)
  H_prime = int((H - pool_h) / stride + 1)
  W_prime = int((W - pool_w) / stride + 1)
  
  out = np.empty([N,C,H_prime,W_prime])
  for n in range(N):
      for c in range(C):
          for i in range(H_prime):
              for j in range(W_prime):
                  out[n,c,i,j] = np.max(x[n,c,i*stride:i*stride+pool_h,j*stride:j*stride+pool_w])
  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 
  cache = (x, pool_param)
  return out, cache

def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  x, pool_param = cache
  pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the max pooling backward pass.
  # ================================================================ #

  N, C, H, W = x.shape
  dx = np.zeros_like(x)
  
  pool_h = pool_param['pool_height']
  pool_w = pool_param['pool_width']
  stride = pool_param['stride']
  
  H_prime = int((H - pool_h) / stride + 1)
  W_prime = int((W - pool_w) / stride + 1)

  for n in range(N):
      for c in range(C):
          for i in range(H_prime):
              for j in range(W_prime):
                  x_filter = x[n,c,i*stride:i*stride+pool_height,j*stride:j*stride+pool_width]
                  max_x = np.max(x_filter)
                  dx[n,c,i*stride:i*stride+pool_height,j*stride:j*stride+pool_width] += (x_filter == max_x) * dout[n,c,i,j]



  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

  return dx

def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Computes the forward pass for spatial batch normalization.
  
  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features
    
  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """
  out, cache = None, None

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the spatial batchnorm forward pass.
  #
  #   You may find it useful to use the batchnorm forward pass you 
  #   implemented in HW #4.
  # ================================================================ #
  N,C,H,W = x.shape
  x_transpose = x.transpose((0,2,3,1))
  x_transpose = x_transpose.reshape(-1,C)
  """
  mode = bn_param['mode']
  eps = bn_param.get('eps', 1e-5)
  momentum = bn_param.get('momentum', 0.9)
  
  running_mean = bn_param.get('running_mean', np.zeros(C, dtype=x.dtype))
  running_var = bn_param.get('running_var', np.zeros(C, dtype=x.dtype))
  
  out, cache = None, None
  
  if mode == 'train':
      # calculate the sample mean and sample variance
      sample_mean = np.mean(x_transpose, axis=0)
      sample_var = np.var(x_transpose, axis=0)
      # calculate and record running_mean and running_var
      running_mean = momentum * running_mean + (1 - momentum) * sample_mean
      running_var = momentum * running_var + (1 - momentum) * sample_var
      # normalize the activations
      x_hat = (x_transpose - sample_mean)/ np.sqrt((sample_var + eps)) 
      # calculate out and cache
      out = gamma * x_hat + beta
      cache = (x, x_hat, sample_mean, sample_var, gamma, beta, eps) 
   
  elif mode == 'test':  
    # use running_mean and running_var to normalized testing data
    x_hat = (x_transpose - running_mean)/ np.sqrt((running_var + eps))
    # calculate out
    out = gamma * x_hat + beta
    
  else:
    raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

  # Store the updated running means back into bn_param
  bn_param['running_mean'] = running_mean
  bn_param['running_var'] = running_var
  
  # reshape
  out = out.reshape(*x_transpose.shape).transpose((0,3,1,2))  
  
  """
  x_transpose = x.transpose((0,2,3,1))
  x_reshape = x_transpose.reshape((-1,x.shape[1]))
  out, cache = batchnorm_forward(x_reshape, gamma, beta, bn_param)
  out = out.reshape(*x_transpose.shape).transpose((0,3,1,2))
  

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

  return out, cache


def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """
  dx, dgamma, dbeta = None, None, None

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the spatial batchnorm backward pass.
  #
  #   You may find it useful to use the batchnorm forward pass you 
  #   implemented in HW #4.
  # ================================================================ #
  dout_transpose = dout.transpose((0,2,3,1))
  dout_reshape = dout_transpose.reshape(-1, dout.shape[1])

  dx, dgamma, dbeta = batchnorm_backward(dout_reshape, cache)

  dx = dx.reshape(*dout_transpose.shape)
  dx = dx.transpose((0,3,1,2))

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

  return dx, dgamma, dbeta