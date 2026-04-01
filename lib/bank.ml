(** A bank of oscillators.

    Token arrives → bank rings → state emerges.

    The bank is a function: force vector → state vector.
    Each oscillator independently transforms its component of the force
    into [position, velocity]. The collection of all oscillator states
    IS the representation. *)

type t = {
  oscillators : Oscillator.t array;
  n : int;
}

let create n =
  let osc = Array.init n (fun i ->
    let f = Float.of_int i /. Float.of_int (n - 1) in
    let omega0 = 0.1 +. (Float.pi -. 0.1) *. f in
    let gamma = 0.05 +. 0.15 *. f in
    Oscillator.create ~omega0 ~gamma
  ) in
  { oscillators = osc; n }

(** Strike the bank: apply force vector, return new state.
    Pure function — takes old state, returns new state. No mutation.

    state = [pos_0, pos_1, ..., vel_0, vel_1, ...]
    Each oscillator: pos' = decay * pos + force * h(dt)
                     vel' = decay * vel + force              *)
let strike bank ~decay ~dt (state : Vec.t) (force : Vec.t) : Vec.t =
  let n = bank.n in
  Vec.create (2 * n) (fun k ->
    let i = k mod n in
    let osc = bank.oscillators.(i) in
    let f = force.(i) in
    if k < n then
      (* position *)
      decay *. state.(k) +. f *. Oscillator.impulse osc dt
    else
      (* velocity *)
      decay *. state.(k) +. f *. Oscillator.impulse_vel osc dt
  )

(** Zero state *)
let zero_state bank = Vec.zeros (2 * bank.n)
