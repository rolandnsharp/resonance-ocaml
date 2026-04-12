## Resonance model — damped rotation neural network.
##
## Architecture:
##   bank(FFT, once) → [rotate → project → mix → SineGate → residual] × L → project → readout
##
## One equation per oscillator per layer:
##   pos' = γ(t)·[pos·cos(ω) + vel·sin(ω)/ω] + (1-γ(t))·β(t)·drive_pos
##   vel' = γ(t)·[vel·cos(ω) - pos·ω·sin(ω)] + (1-γ(t))·β(t)·drive_vel

import std/[math, strformat, random]
import gpu

type
  Param* = object
    w*: GpuBuf      # weights
    g*: GpuBuf      # gradients
    m*: GpuBuf      # Adam momentum
    v*: GpuBuf      # Adam velocity
    n*: int         # number of elements

  Layer* = object
    projGamma*: Param    # dim → n_osc (damping control)
    projBeta*: Param     # dim → n_osc (absorption control)
    projSense*: Param    # dim → n_osc (readout sensitivity)
    # Interference: dense W computes pairwise cross-terms
    spectralRe*: Param   # reused as wMix (dim × dim)
    spectralIm*: Param   # unused placeholder
    gateProj*: Param     # unused placeholder
    # Projection biases
    projGammaBias*: Param
    projBetaBias*: Param
    projSenseBias*: Param
    mixScale*: Param     # dim (residual scaling)

  Model* = object
    nOsc*: int
    dim*: int
    nLayers*: int
    vocabSize*: int
    seqLen*: int
    # Weights
    drive*: Param        # vocabSize × nOsc (token → oscillator excitation)
    layers*: seq[Layer]
    wOut*: Param         # dim × vocabSize (final projection)
    # Oscillator constants (precomputed, not learned)
    cosW*: GpuBuf        # cos(ω) per oscillator
    sinW*: GpuBuf        # sin(ω) per oscillator
    freqs*: GpuBuf       # ω per oscillator
    # Bank kernels (for FFT encoding)
    hPosFft*: GpuBuf     # FFT of position impulse responses
    hVelFft*: GpuBuf     # FFT of velocity impulse responses

proc initParam(n: int, initStd: float32 = 0.0): Param =
  result.n = n
  result.w = gpuCreate(n)
  result.g = gpuCreate(n)
  result.m = gpuCreate(n)
  result.v = gpuCreate(n)
  if initStd > 0:
    var data = newSeq[float32](n)
    for i in 0..<n:
      # Box-Muller
      let u1 = rand(1.0).float32 + 1e-7
      let u2 = rand(1.0).float32
      data[i] = sqrt(-2.0 * ln(u1)).float32 * cos(2.0 * PI * u2).float32 * initStd
    result.w.gpuUpload(data)

proc adamStep*(p: var Param, lr, b1, b2, wd, bc1, bc2: float32) =
  if p.n > 0:
    gpu_adamw(p.w.data, p.g.data, p.m.data, p.v.data,
              lr.cfloat, b1.cfloat, b2.cfloat, wd.cfloat,
              bc1.cfloat, bc2.cfloat, p.n.cint)

proc createModel*(nOsc, nLayers, vocabSize, seqLen: int): Model =
  let dim = nOsc * 2
  let scale = 1.0 / sqrt(dim.float32)
  let projScale = 0.1 / sqrt(nOsc.float32)

  result.nOsc = nOsc
  result.dim = dim
  result.nLayers = nLayers
  result.vocabSize = vocabSize
  result.seqLen = seqLen

  # Drive table
  result.drive = initParam(vocabSize * nOsc, 0.02)

  # Output projection
  result.wOut = initParam(dim * vocabSize, 0.001)

  # Layers
  result.layers = newSeq[Layer](nLayers)
  for l in 0..<nLayers:
    result.layers[l].projGamma = initParam(dim * nOsc, projScale)
    result.layers[l].projBeta = initParam(dim * nOsc, projScale)
    result.layers[l].projSense = initParam(dim * nOsc, projScale)
    # Interference: dense W computes all pairwise cross-terms between oscillators
    result.layers[l].spectralRe = initParam(dim * dim, scale)  # reuse field as wMix
    result.layers[l].spectralIm = initParam(0)  # unused
    result.layers[l].gateProj = initParam(0)   # unused
    # Projection biases (Python had these, we didn't)
    result.layers[l].projGammaBias = initParam(nOsc, 0.0)
    result.layers[l].projBetaBias = initParam(nOsc, 0.0)
    result.layers[l].projSenseBias = initParam(nOsc, 0.0)
    result.layers[l].mixScale = initParam(dim, 0.0)
    # Init mixScale to ones
    var ones = newSeq[float32](dim)
    for i in 0..<dim: ones[i] = 1.0
    result.layers[l].mixScale.w.gpuUpload(ones)

  # Precompute oscillator constants
  var cosData = newSeq[float32](nOsc)
  var sinData = newSeq[float32](nOsc)
  var freqData = newSeq[float32](nOsc)
  for k in 0..<nOsc:
    let f = 0.1 + (PI - 0.1) * k.float32 / max(1, nOsc - 1).float32
    freqData[k] = f
    cosData[k] = cos(f)
    sinData[k] = sin(f)

  result.cosW = gpuCreate(nOsc)
  result.sinW = gpuCreate(nOsc)
  result.freqs = gpuCreate(nOsc)
  result.cosW.gpuUpload(cosData)
  result.sinW.gpuUpload(sinData)
  result.freqs.gpuUpload(freqData)

  echo &"Resonance: {nOsc} osc, {nLayers} layers, dim={dim}, vocab={vocabSize}"
  let nParams = vocabSize * nOsc + dim * vocabSize +
    nLayers * (3 * dim * nOsc + dim * dim + dim)
  echo &"Parameters: {nParams}"

  # Precompute impulse response FFTs for the bank
  let fftLen = 2 * seqLen
  let fftCLen = fftLen div 2 + 1  # complex output length of R2C
  var hPos = newSeq[float32](seqLen)
  var hVel = newSeq[float32](seqLen)
  var padded = newSeq[float32](fftLen)

  # We'll store all kernel FFTs as interleaved (re, im) pairs
  # hPosFft: nOsc * fftCLen * 2 floats
  result.hPosFft = gpuCreate(nOsc * fftCLen * 2)
  result.hVelFft = gpuCreate(nOsc * fftCLen * 2)

  let tmpReal = gpuCreate(fftLen)
  let tmpComplex = gpuCreate(fftCLen * 2)

  for k in 0..<nOsc:
    let w0 = freqData[k]
    let g = 0.05 + 0.15 * k.float32 / max(1, nOsc - 1).float32
    let wd = w0 * sqrt(max(1e-6, 1.0 - g * g))
    let alpha = g * w0
    for i in 0..<seqLen:
      let t = i.float32
      let decay = exp(-alpha * t)
      hPos[i] = decay * sin(wd * t) / max(1e-8, wd)
      hVel[i] = decay * (cos(wd * t) - alpha / max(1e-8, wd) * sin(wd * t))
    # Pad and FFT
    for i in 0..<fftLen: padded[i] = if i < seqLen: hPos[i] else: 0.0
    tmpReal.gpuUpload(padded)
    discard gpu_fft_r2c_batched(tmpReal.data, tmpComplex.data, fftLen.cint, 1.cint)
    # Copy to the k-th slot in hPosFft
    var complexData = gpuDownload(tmpComplex)
    var allData = gpuDownload(result.hPosFft)
    for i in 0..<fftCLen * 2:
      allData[k * fftCLen * 2 + i] = complexData[i]
    result.hPosFft.gpuUpload(allData)

    # Same for velocity
    for i in 0..<fftLen: padded[i] = if i < seqLen: hVel[i] else: 0.0
    tmpReal.gpuUpload(padded)
    discard gpu_fft_r2c_batched(tmpReal.data, tmpComplex.data, fftLen.cint, 1.cint)
    complexData = gpuDownload(tmpComplex)
    allData = gpuDownload(result.hVelFft)
    for i in 0..<fftCLen * 2:
      allData[k * fftCLen * 2 + i] = complexData[i]
    result.hVelFft.gpuUpload(allData)

  echo &"Bank: {nOsc} oscillators, fftLen={fftLen}"

proc countParams*(m: Model): int =
  result = m.drive.n + m.wOut.n
  for l in m.layers:
    result += l.projGamma.n + l.projBeta.n + l.projSense.n +
              l.spectralRe.n + l.projGammaBias.n + l.projBetaBias.n +
              l.projSenseBias.n + l.mixScale.n
