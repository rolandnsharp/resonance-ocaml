(** A bank of oscillators — each one rings at its own frequency.

    Low frequency, low damping → rings for hundreds of bytes (sentence context)
    High frequency, high damping → rings briefly (character detail)
    The physics gives us multi-scale context for free. *)

type t = {
  oscillators : Oscillator.t array;
  n : int;
  dt : float;
}

let create n =
  let osc = Array.init n (fun i ->
    let f = Float.of_int i /. Float.of_int (n - 1) in
    let omega0 = 0.1 +. (Float.pi -. 0.1) *. f in
    let gamma = 0.05 +. 0.15 *. f in
    Oscillator.create ~omega0 ~gamma
  ) in
  { oscillators = osc; n; dt = 1.0 }

(** Strike: exact ODE step per oscillator.
    No crude decay — each oscillator decays at its own natural rate. *)
let strike bank (state : Vec.t) (force : Vec.t) : Vec.t =
  let n = bank.n in
  Vec.create (2 * n) (fun k ->
    let i = k mod n in
    let osc = bank.oscillators.(i) in
    let pos = state.(i) in
    let vel = state.(n + i) in
    let f = force.(i) in
    let pos', vel' = Oscillator.step osc ~dt:bank.dt ~force:f (pos, vel) in
    if k < n then pos' else vel'
  )

let zero_state bank = Vec.zeros (2 * bank.n)
