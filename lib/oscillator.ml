(** The one equation: H(ω) = 1 / (ω₀² - ω² + 2iγω)

    Everything flows from this. A damped harmonic oscillator is a
    function from driving force to response. Compose them and you
    get a neural network. *)

type complex = { re : float; im : float }

let c_zero = { re = 0.0; im = 0.0 }
let c_one = { re = 1.0; im = 0.0 }
let c_add a b = { re = a.re +. b.re; im = a.im +. b.im }
let c_sub a b = { re = a.re -. b.re; im = a.im -. b.im }
let c_mul a b = { re = a.re *. b.re -. a.im *. b.im;
                  im = a.re *. b.im +. a.im *. b.re }
let c_scale s c = { re = s *. c.re; im = s *. c.im }
let c_div a b =
  let d = b.re *. b.re +. b.im *. b.im in
  { re = (a.re *. b.re +. a.im *. b.im) /. d;
    im = (a.im *. b.re -. a.re *. b.im) /. d }
let c_mag c = sqrt (c.re *. c.re +. c.im *. c.im)

(** A single damped harmonic oscillator. *)
type t = {
  omega0 : float;
  gamma  : float;
}

let create ~omega0 ~gamma =
  { omega0; gamma = Float.max 0.01 (Float.min 0.99 gamma) }

(** The transfer function — the entire model in one expression. *)
let transfer osc omega =
  let w0 = osc.omega0 in
  let g = osc.gamma in
  c_div c_one { re = w0 *. w0 -. omega *. omega;
                im = 2.0 *. g *. w0 *. omega }

(** Damped natural frequency *)
let omega_d osc =
  let g = osc.gamma in
  osc.omega0 *. sqrt (Float.max 1e-6 (1.0 -. g *. g))

(** Impulse response: h(t) = e^{-γω₀t} sin(ωd·t) / ωd *)
let impulse osc t =
  let wd = omega_d osc in
  exp (-. osc.gamma *. osc.omega0 *. t) *. sin (wd *. t) /. wd

(** Velocity impulse response *)
let impulse_vel osc t =
  let g = osc.gamma in
  let w0 = osc.omega0 in
  let wd = omega_d osc in
  let decay = exp (-. g *. w0 *. t) in
  decay *. (cos (wd *. t) -. (g *. w0 /. wd) *. sin (wd *. t))
