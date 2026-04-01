(** Oscillator bank with FFT convolution.

    Precomputes kernel FFTs. When frequencies change (learning),
    kernels are recomputed. Also computes derivative kernels
    dh/dω for frequency learning. *)

open Owl

type kernels = {
  h_pos_fft : Dense.Ndarray.Z.arr array;
  h_vel_fft : Dense.Ndarray.Z.arr array;
  dh_pos_fft : Dense.Ndarray.Z.arr array;  (** dh/dω for frequency learning *)
  dh_vel_fft : Dense.Ndarray.Z.arr array;
  fft_len : int;
  seq_len : int;
}

(** dh/dω: how the impulse response changes when we retune *)
let d_impulse_d_omega osc t =
  let wd = Oscillator.damped_freq osc in
  let alpha = osc.gamma *. osc.omega0 in
  let decay = exp (-. alpha *. t) in
  t *. decay *. cos (wd *. t) /. (Float.max 1e-8 wd)

let d_impulse_vel_d_omega osc t =
  let wd = Oscillator.damped_freq osc in
  let alpha = osc.gamma *. osc.omega0 in
  let decay = exp (-. alpha *. t) in
  -. t *. decay *. sin (wd *. t)

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
    dh_pos_fft = Array.map (fun osc -> make_fft d_impulse_d_omega osc) oscillators;
    dh_vel_fft = Array.map (fun osc -> make_fft d_impulse_vel_d_omega osc) oscillators;
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

(** Encode with derivative kernels: same drives, dh/dω kernels.
    Returns d_state/d_omega for each oscillator at each position. *)
let encode_deriv oscillators kernels drives =
  let seq_len = Array.length drives in
  let n_osc = Array.length oscillators in
  let fft_len = kernels.fft_len in
  let convolve_osc k kernel_arr =
    let col = Array.init seq_len (fun t -> drives.(t).(k)) in
    convolve_one col kernel_arr.(k) fft_len seq_len
  in
  let dpos = Array.init n_osc (fun k -> convolve_osc k kernels.dh_pos_fft) in
  let dvel = Array.init n_osc (fun k -> convolve_osc k kernels.dh_vel_fft) in
  (* Return (seq_len, 2*n_osc) — gradient of state w.r.t. each osc's omega *)
  Array.init n_osc (fun k ->
    Array.init seq_len (fun t -> (dpos.(k).(t), dvel.(k).(t))))
