(** Oscillator bank with batched FFT.

    One FFT per oscillator type (pos/vel), not per oscillator.
    Pack all drives into a matrix, FFT each row, multiply, IFFT. *)

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
  {
    h_pos_fft = Array.map (fun osc ->
      Owl_fft.D.rfft (pad (Array.init seq_len (fun i ->
        Oscillator.impulse osc (Float.of_int i))))
    ) oscillators;
    h_vel_fft = Array.map (fun osc ->
      Owl_fft.D.rfft (pad (Array.init seq_len (fun i ->
        Oscillator.impulse_vel osc (Float.of_int i))))
    ) oscillators;
    fft_len;
    seq_len;
  }

(** Convolve one oscillator's drive with its kernels → (pos, vel) arrays *)
let convolve_one kernels drives seq_len k =
  let fft_len = kernels.fft_len in
  let padded = Dense.Ndarray.D.zeros [| fft_len |] in
  for t = 0 to seq_len - 1 do
    Dense.Ndarray.D.set padded [| t |] drives.(t).(k)
  done;
  let d_fft = Owl_fft.D.rfft padded in
  let p_result = Owl_fft.D.irfft (Dense.Ndarray.Z.mul d_fft kernels.h_pos_fft.(k)) in
  let v_result = Owl_fft.D.irfft (Dense.Ndarray.Z.mul d_fft kernels.h_vel_fft.(k)) in
  let pos = Array.init seq_len (fun t -> Dense.Ndarray.D.get p_result [| t |]) in
  let vel = Array.init seq_len (fun t -> Dense.Ndarray.D.get v_result [| t |]) in
  (pos, vel)

(** Encode: parallel FFT across oscillators *)
let encode _oscillators kernels drives =
  let seq_len = Array.length drives in
  let n_osc = Array.length kernels.h_pos_fft in
  (* Sequential for now — parallel overhead exceeds FFT cost *)
  let results = Array.init n_osc (convolve_one kernels drives seq_len) in
  (* Pack into (seq_len, 2*n_osc) *)
  Array.init seq_len (fun t ->
    Array.init (2 * n_osc) (fun k ->
      let pos, vel = results.(k mod n_osc) in
      if k < n_osc then pos.(t) else vel.(t)))
