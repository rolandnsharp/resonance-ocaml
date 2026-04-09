(** Oscillator bank with FFT convolution.

    Precomputes kernel FFTs. When frequencies change (learning),
    kernels are recomputed. Also computes derivative kernels
    dh/dω for frequency learning. *)

open Owl

type kernels = {
  h_pos_fft : Dense.Ndarray.Z.arr array;
  h_vel_fft : Dense.Ndarray.Z.arr array;
  fft_len : int;
  seq_len : int;
}

let precompute_kernels oscillators seq_len =
  let fft_len = 2 * seq_len in
  let pad arr =
    let a = Dense.Ndarray.D.zeros [| fft_len |] in
    Array.iteri (fun i v -> Dense.Ndarray.D.set a [| i |] v) arr; a
  in
  let make_fft f osc =
    Owl_fft.D.rfft (pad (Array.init seq_len (fun i -> f osc (Float.of_int i))))
  in
  {
    h_pos_fft = Array.map (fun osc -> make_fft Oscillator.impulse osc) oscillators;
    h_vel_fft = Array.map (fun osc -> make_fft Oscillator.impulse_vel osc) oscillators;
    fft_len; seq_len;
  }

(** Convolve one oscillator's drive with a precomputed kernel FFT *)
let convolve_one drive_col kernel_fft fft_len seq_len =
  let padded = Dense.Ndarray.D.zeros [| fft_len |] in
  Array.iteri (fun i v -> Dense.Ndarray.D.set padded [| i |] v) drive_col;
  let d_fft = Owl_fft.D.rfft padded in
  let result = Owl_fft.D.irfft (Dense.Ndarray.Z.mul d_fft kernel_fft) in
  Array.init seq_len (fun i -> Dense.Ndarray.D.get result [| i |])

(** Encode: drives → FFT convolve → states *)
let encode oscillators kernels drives =
  let seq_len = Array.length drives in
  let n_osc = Array.length oscillators in
  let fft_len = kernels.fft_len in
  let convolve_osc k kernel_arr =
    let col = Array.init seq_len (fun t -> drives.(t).(k)) in
    convolve_one col kernel_arr.(k) fft_len seq_len
  in
  let pos = Array.init n_osc (fun k -> convolve_osc k kernels.h_pos_fft) in
  let vel = Array.init n_osc (fun k -> convolve_osc k kernels.h_vel_fft) in
  Array.init seq_len (fun t ->
    Array.init (2 * n_osc) (fun k ->
      if k < n_osc then pos.(k).(t) else vel.(k - n_osc).(t)))

(** Adjoint of convolve_one: convolve with conj(H) in freq domain *)
let convolve_adjoint d_col kernel_fft fft_len seq_len =
  let padded = Dense.Ndarray.D.zeros [| fft_len |] in
  Array.iteri (fun i v -> Dense.Ndarray.D.set padded [| i |] v) d_col;
  let d_fft = Owl_fft.D.rfft padded in
  let result = Owl_fft.D.irfft
    (Dense.Ndarray.Z.mul d_fft (Dense.Ndarray.Z.conj kernel_fft)) in
  Array.init seq_len (fun i -> Dense.Ndarray.D.get result [| i |])

(** Backward: d_bank_states → d_drives via FFT adjoint.
    Each oscillator's pos + vel gradients convolve with conj(H). *)
let encode_backward kernels d_states =
  let seq_len = Array.length d_states in
  let n_osc = Array.length d_states.(0) / 2 in
  let fft_len = kernels.fft_len in
  let adj k h_fft offset =
    convolve_adjoint
      (Array.init seq_len (fun t -> d_states.(t).(k + offset)))
      h_fft.(k) fft_len seq_len
  in
  let d_pos = Array.init n_osc (fun k -> adj k kernels.h_pos_fft 0) in
  let d_vel = Array.init n_osc (fun k -> adj k kernels.h_vel_fft n_osc) in
  Array.init seq_len (fun t ->
    Array.init n_osc (fun k -> d_pos.(k).(t) +. d_vel.(k).(t)))
