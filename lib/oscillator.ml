(** H(ω) = 1 / (ω₀² - ω² + 2iγω)

    One equation. Everything else is composition of this. *)

type t = { omega0 : float; gamma : float }

let create omega0 gamma =
  { omega0; gamma = Float.max 0.01 (Float.min 0.99 gamma) }

let damped_freq t =
  t.omega0 *. sqrt (Float.max 1e-6 (1.0 -. t.gamma *. t.gamma))

(** Impulse response at time t *)
let impulse t time =
  let wd = damped_freq t in
  let alpha = t.gamma *. t.omega0 in
  exp (-. alpha *. time) *. sin (wd *. time) /. (Float.max 1e-8 wd)

(** Velocity impulse response at time t *)
let impulse_vel t time =
  let wd = damped_freq t in
  let alpha = t.gamma *. t.omega0 in
  let decay = exp (-. alpha *. time) in
  decay *. (cos (wd *. time) -. alpha /. (Float.max 1e-8 wd) *. sin (wd *. time))

(** Logarithmic frequency spread: n oscillators from low to high *)
let spread n =
  Array.init n (fun i ->
    let f = Float.of_int i /. Float.max 1.0 (Float.of_int (n - 1)) in
    create (0.1 +. (Float.pi -. 0.1) *. f) (0.05 +. 0.15 *. f))
