(** Oscillator bank with FFT convolution.

    Given a sequence of drive forces, convolve each oscillator's
    impulse response with the drive to get the causal response
    at every position. This is the key operation — at position t,
    the state encodes ALL previous tokens filtered through each
    oscillator's resonance.

    token_sequence |> drives |> fft_convolve h(t) |> [pos; vel] *)

open Owl

(** Compute impulse response kernel for one oscillator, n frames *)
let kernel osc n =
  Array.init n (fun i -> Oscillator.impulse osc (Float.of_int i))

let kernel_vel osc n =
  Array.init n (fun i -> Oscillator.impulse_vel osc (Float.of_int i))

(** Causal convolution via FFT: convolve signal with kernel.
    Zero-padded to 2n to prevent circular wraparound. *)
let causal_convolve signal kern =
  let n = Array.length signal in
  let fft_len = 2 * n in
  let pad arr =
    let a = Dense.Ndarray.D.zeros [| fft_len |] in
    Array.iteri (fun i v -> Dense.Ndarray.D.set a [| i |] v) arr; a
  in
  let s_fft = Owl_fft.D.rfft (pad signal) in
  let k_fft = Owl_fft.D.rfft (pad kern) in
  let product = Dense.Ndarray.Z.mul s_fft k_fft in
  let result = Owl_fft.D.irfft product in
  Array.init n (fun i -> Dense.Ndarray.D.get result [| i |])

(** Encode a sequence of drive forces through the oscillator bank.

    drives: (seq_len, n_osc) — force per oscillator per position
    Returns: (seq_len, 2 * n_osc) — [position; velocity] states

    Each oscillator independently convolves its drive with h(t).
    The result at position t is the causal response to all
    tokens 0..t, filtered through that oscillator's resonance. *)
let encode oscillators drives =
  let seq_len = Array.length drives in
  let n_osc = Array.length oscillators in
  (* Precompute impulse response kernels *)
  let h_pos = Array.map (fun osc -> kernel osc seq_len) oscillators in
  let h_vel = Array.map (fun osc -> kernel_vel osc seq_len) oscillators in
  (* For each oscillator, extract its drive column and convolve *)
  let pos_responses = Array.init n_osc (fun k ->
    let drive_col = Array.init seq_len (fun t -> drives.(t).(k)) in
    causal_convolve drive_col h_pos.(k)
  ) in
  let vel_responses = Array.init n_osc (fun k ->
    let drive_col = Array.init seq_len (fun t -> drives.(t).(k)) in
    causal_convolve drive_col h_vel.(k)
  ) in
  (* Pack into (seq_len, 2*n_osc): [pos_0..pos_n, vel_0..vel_n] per position *)
  Array.init seq_len (fun t ->
    Array.init (2 * n_osc) (fun k ->
      if k < n_osc then pos_responses.(k).(t)
      else vel_responses.(k - n_osc).(t)
    )
  )
