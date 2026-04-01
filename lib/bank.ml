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

(** Encode: one FFT per drive column, multiply by kernel, IFFT.
    Reuses a single padded buffer to reduce allocation. *)
let encode _oscillators kernels drives =
  let seq_len = Array.length drives in
  let n_osc = Array.length kernels.h_pos_fft in
  let fft_len = kernels.fft_len in
  let padded = Dense.Ndarray.D.zeros [| fft_len |] in
  let pos = Array.make_matrix n_osc seq_len 0.0 in
  let vel = Array.make_matrix n_osc seq_len 0.0 in
  for k = 0 to n_osc - 1 do
    (* Fill padded buffer with drive column k *)
    for t = 0 to seq_len - 1 do
      Dense.Ndarray.D.set padded [| t |] drives.(t).(k)
    done;
    for t = seq_len to fft_len - 1 do
      Dense.Ndarray.D.set padded [| t |] 0.0
    done;
    let d_fft = Owl_fft.D.rfft padded in
    (* Position *)
    let p_result = Owl_fft.D.irfft (Dense.Ndarray.Z.mul d_fft kernels.h_pos_fft.(k)) in
    for t = 0 to seq_len - 1 do
      pos.(k).(t) <- Dense.Ndarray.D.get p_result [| t |]
    done;
    (* Velocity *)
    let v_result = Owl_fft.D.irfft (Dense.Ndarray.Z.mul d_fft kernels.h_vel_fft.(k)) in
    for t = 0 to seq_len - 1 do
      vel.(k).(t) <- Dense.Ndarray.D.get v_result [| t |]
    done
  done;
  Array.init seq_len (fun t ->
    Array.init (2 * n_osc) (fun k ->
      if k < n_osc then pos.(k).(t) else vel.(k - n_osc).(t)))
