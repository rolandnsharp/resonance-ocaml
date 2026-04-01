(** The one equation: H(ω) = 1 / (ω₀² - ω² + 2iγω)

    Everything flows from this. Analysis, synthesis, coupling,
    prediction, error — all are compositions of this transfer function. *)

type complex = { re : float; im : float }

let complex re im = { re; im }
let c_zero = { re = 0.0; im = 0.0 }
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
let c_conj c = { re = c.re; im = -. c.im }

(** A single damped harmonic oscillator. Two parameters: frequency and damping. *)
type t = {
  omega0 : float;   (** natural frequency (rad/s) *)
  gamma  : float;    (** damping ratio (0 = undamped, 1 = critical) *)
}

let create ~omega0 ~gamma =
  { omega0; gamma = Float.max 0.01 (Float.min 0.99 gamma) }

(** The transfer function. This is the entire model.
    H(ω) = 1 / (ω₀² - ω² + 2iγω₀ω) *)
let transfer osc omega =
  let w0 = osc.omega0 in
  let g = osc.gamma in
  let denom = complex
    (w0 *. w0 -. omega *. omega)
    (2.0 *. g *. w0 *. omega)
  in
  c_div (complex 1.0 0.0) denom

(** Impulse response: h(t) = e^{-γω₀t} sin(ωd·t) / ωd
    where ωd = ω₀√(1-γ²) is the damped frequency. *)
let impulse_response osc t =
  let w0 = osc.omega0 in
  let g = osc.gamma in
  let wd = w0 *. sqrt (Float.max 1e-6 (1.0 -. g *. g)) in
  let decay = exp (-. g *. w0 *. t) in
  decay *. sin (wd *. t) /. wd

(** Analytical derivative of impulse response w.r.t. frequency.
    dh/dω₀ — how the response changes when we tune the oscillator. *)
let d_impulse_d_omega osc t =
  let w0 = osc.omega0 in
  let g = osc.gamma in
  let wd = w0 *. sqrt (Float.max 1e-6 (1.0 -. g *. g)) in
  let decay = exp (-. g *. w0 *. t) in
  t *. decay *. cos (wd *. t) /. wd

(** Analytical derivative w.r.t. damping.
    dh/dγ — how the response changes when we adjust damping. *)
let d_impulse_d_gamma osc t =
  let w0 = osc.omega0 in
  let g = osc.gamma in
  let wd = w0 *. sqrt (Float.max 1e-6 (1.0 -. g *. g)) in
  let decay = exp (-. g *. w0 *. t) in
  -. t *. decay *. sin (wd *. t) /. wd
