(** A bank of oscillators — the fundamental computational unit.

    Each oscillator in the bank has its own frequency and damping.
    The bank analyzes a signal by convolving it with each oscillator's
    impulse response. The result is a decomposition of the signal
    into resonant modes — position and velocity for each oscillator. *)

type t = {
  oscillators : Oscillator.t array;
  hop : int;           (** samples per frame *)
  sample_rate : float;
}

let create ~n ~sample_rate ~hop =
  let osc = Array.init n (fun i ->
    let f = Float.of_int i /. Float.of_int (n - 1) in
    (* Logarithmic frequency spread from 20 Hz to Nyquist *)
    let freq = 20.0 *. (sample_rate /. 2.0 /. 20.0) ** f in
    let omega0 = freq *. 2.0 *. Float.pi in
    Oscillator.create ~omega0 ~gamma:0.1
  ) in
  { oscillators = osc; hop; sample_rate }

let n_osc bank = Array.length bank.oscillators

(** Convolve a signal frame with one oscillator's impulse response.
    Returns (position, velocity) — the oscillator's state. *)
let convolve_frame bank osc_idx (frame : float array) =
  let osc = bank.oscillators.(osc_idx) in
  let dt = 1.0 /. bank.sample_rate in
  let n = Array.length frame in
  let pos = ref 0.0 in
  let vel = ref 0.0 in
  for i = 0 to n - 1 do
    let t = Float.of_int (n - 1 - i) *. dt in
    let h = Oscillator.impulse_response osc t in
    pos := !pos +. frame.(i) *. h;
    (* velocity response: dh/dt ≈ finite difference *)
    let h2 = Oscillator.impulse_response osc (t +. dt) in
    vel := !vel +. frame.(i) *. (h2 -. h) /. dt
  done;
  (!pos, !vel)

(** Analyze a signal into oscillator states.
    signal -> (n_frames, 2 * n_osc) state matrix *)
let analyze bank (signal : float array) =
  let n_osc = n_osc bank in
  let n_samples = Array.length signal in
  let n_frames = n_samples / bank.hop in
  (* State: position and velocity for each oscillator at each frame *)
  let states = Array.init n_frames (fun frame_idx ->
    let frame_start = frame_idx * bank.hop in
    let frame_end = Int.min n_samples (frame_start + bank.hop * 4) in
    let frame = Array.sub signal frame_start (frame_end - frame_start) in
    Array.init (2 * n_osc) (fun k ->
      let osc_idx = k mod n_osc in
      let pos, vel = convolve_frame bank osc_idx frame in
      if k < n_osc then pos else vel
    )
  ) in
  states
