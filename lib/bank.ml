(** Oscillator bank with FFT convolution.

    Precomputes kernel FFTs — they don't change during training.
    Only the drive FFTs are computed per sequence. *)

open Owl

(** Precomputed kernel FFTs for a given sequence length *)
type kernels = {
  h_pos_fft : Dense.Ndarray.Z.arr array;  (** per-oscillator position kernel FFT *)
  h_vel_fft : Dense.Ndarray.Z.arr array;  (** per-oscillator velocity kernel FFT *)
  fft_len : int;
}

let precompute_kernels oscillators seq_len =
  let fft_len = 2 * seq_len in
  let pad arr =
    let a = Dense.Ndarray.D.zeros [| fft_len |] in
    Array.iteri (fun i v -> Dense.Ndarray.D.set a [| i |] v) arr; a
  in
  let make_kernel_fft osc impulse_fn =
    let k = Array.init seq_len (fun i -> impulse_fn osc (Float.of_int i)) in
    Owl_fft.D.rfft (pad k)
  in
  {
    h_pos_fft = Array.map (fun osc -> make_kernel_fft osc Oscillator.impulse) oscillators;
    h_vel_fft = Array.map (fun osc -> make_kernel_fft osc Oscillator.impulse_vel) oscillators;
    fft_len;
  }

(** Causal convolution: FFT the drive, multiply by precomputed kernel FFT, IFFT *)
let convolve_with_kernel drive_col kernel_fft fft_len seq_len =
  let padded = Dense.Ndarray.D.zeros [| fft_len |] in
  Array.iteri (fun i v -> Dense.Ndarray.D.set padded [| i |] v) drive_col;
  let drive_fft = Owl_fft.D.rfft padded in
  let product = Dense.Ndarray.Z.mul drive_fft kernel_fft in
  let result = Owl_fft.D.irfft product in
  Array.init seq_len (fun i -> Dense.Ndarray.D.get result [| i |])

(** Encode a token sequence through the oscillator bank.
    drives: (seq_len, n_osc)
    Returns: (seq_len, 2 * n_osc) *)
let encode oscillators kernels drives =
  let seq_len = Array.length drives in
  let n_osc = Array.length oscillators in
  (* Extract drive columns and convolve with precomputed kernels *)
  let pos = Array.init n_osc (fun k ->
    let col = Array.init seq_len (fun t -> drives.(t).(k)) in
    convolve_with_kernel col kernels.h_pos_fft.(k) kernels.fft_len seq_len
  ) in
  let vel = Array.init n_osc (fun k ->
    let col = Array.init seq_len (fun t -> drives.(t).(k)) in
    convolve_with_kernel col kernels.h_vel_fft.(k) kernels.fft_len seq_len
  ) in
  (* Pack: (seq_len, 2*n_osc) *)
  Array.init seq_len (fun t ->
    Array.init (2 * n_osc) (fun k ->
      if k < n_osc then pos.(k).(t) else vel.(k - n_osc).(t)))
