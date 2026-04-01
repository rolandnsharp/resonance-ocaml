(** Predictive coding through complex resonance coupling.

    The transfer function H(ω) is complex — it has magnitude AND phase.
    Position = real part. Velocity = imaginary part.
    The coupling between oscillators uses the full complex H(ω),
    so phase relationships matter. "th" creates a different
    interference pattern than "ht" — same oscillators, different phases.

    Learning = retuning ω and γ via analytical dH/dω, dH/dγ. *)

type layer = {
  activity : Vec.t;
  error : Vec.t;
}

type coupling = {
  omega : float array;
  gamma : float array;
  source_omega : float array;
  n_out : int;
  n_in : int;  (** number of oscillators in source (half of source state dim) *)
}

type t = {
  layers : layer array;
  couplings : coupling array;
  n_layers : int;
}

(** Complex transfer function components:
    H(ω) = 1 / (ω₀²-ω² + 2iγω₀ω)
    Re[H] = (ω₀²-ω²) / D
    Im[H] = -2γω₀ω / D
    where D = (ω₀²-ω²)² + (2γω₀ω)² *)
let transfer_re_im ~omega0 ~gamma ~omega =
  let w0sq = omega0 *. omega0 in
  let wsq = omega *. omega in
  let real_num = w0sq -. wsq in
  let imag_num = 2.0 *. gamma *. omega0 *. omega in
  let d = real_num *. real_num +. imag_num *. imag_num +. 1e-8 in
  (real_num /. d, -. imag_num /. d)

(** Predict via complex resonance.
    Input state is [pos_0..pos_n, vel_0..vel_n] — treat as complex:
      z_i = pos_i + i*vel_i
    Output: z_j = Σ_i z_i * H_j(ω_i)
    Expand: out_pos_j = Σ_i (pos_i * Re[H] - vel_i * Im[H])
            out_vel_j = Σ_i (pos_i * Im[H] + vel_i * Re[H]) *)
let resonate coupling (input : Vec.t) : Vec.t =
  let n_osc = coupling.n_in in
  Vec.create (2 * coupling.n_out) (fun k ->
    let j = k mod coupling.n_out in
    let is_vel = k >= coupling.n_out in
    let w0 = coupling.omega.(j) in
    let g = coupling.gamma.(j) in
    let acc = ref 0.0 in
    for i = 0 to n_osc - 1 do
      let wi = coupling.source_omega.(i) in
      let h_re, h_im = transfer_re_im ~omega0:w0 ~gamma:g ~omega:wi in
      let pos_i = input.(i) in
      let vel_i = input.(n_osc + i) in
      if is_vel then
        (* Imaginary part of complex multiply *)
        acc := !acc +. pos_i *. h_im +. vel_i *. h_re
      else
        (* Real part of complex multiply *)
        acc := !acc +. pos_i *. h_re -. vel_i *. h_im
    done;
    !acc
  )

(** Propagate error backward: same complex coupling, transposed *)
let propagate_error coupling (err : Vec.t) : Vec.t =
  let n_osc_out = coupling.n_out in
  Vec.create (2 * coupling.n_in) (fun k ->
    let i = k mod coupling.n_in in
    let is_vel = k >= coupling.n_in in
    let wi = coupling.source_omega.(i) in
    let acc = ref 0.0 in
    for j = 0 to n_osc_out - 1 do
      let w0 = coupling.omega.(j) in
      let g = coupling.gamma.(j) in
      let h_re, h_im = transfer_re_im ~omega0:w0 ~gamma:g ~omega:wi in
      let err_pos = err.(j) in
      let err_vel = err.(n_osc_out + j) in
      if is_vel then
        acc := !acc -. err_pos *. h_im +. err_vel *. h_re
      else
        acc := !acc +. err_pos *. h_re +. err_vel *. h_im
    done;
    !acc
  )

(** Retune: adjust ω and γ using the gradient through complex H *)
let retune ~lr coupling (input : Vec.t) (err : Vec.t) =
  let n_osc = coupling.n_in in
  for j = 0 to coupling.n_out - 1 do
    let w0 = coupling.omega.(j) in
    let g = coupling.gamma.(j) in
    let err_pos = err.(j) in
    let err_vel = err.(coupling.n_out + j) in
    let d_omega = ref 0.0 in
    let d_gamma = ref 0.0 in
    for i = 0 to n_osc - 1 do
      let wi = coupling.source_omega.(i) in
      let pos_i = input.(i) in
      let vel_i = input.(n_osc + i) in
      (* Numerical derivatives of H_re, H_im w.r.t. omega0 and gamma *)
      let eps = 1e-4 in
      let h_re, h_im = transfer_re_im ~omega0:w0 ~gamma:g ~omega:wi in
      let h_re_dw, h_im_dw = transfer_re_im ~omega0:(w0 +. eps) ~gamma:g ~omega:wi in
      let h_re_dg, h_im_dg = transfer_re_im ~omega0:w0 ~gamma:(g +. eps) ~omega:wi in
      let dre_dw = (h_re_dw -. h_re) /. eps in
      let dim_dw = (h_im_dw -. h_im) /. eps in
      let dre_dg = (h_re_dg -. h_re) /. eps in
      let dim_dg = (h_im_dg -. h_im) /. eps in
      (* Chain rule through complex multiply *)
      let d_out_pos_dw = pos_i *. dre_dw -. vel_i *. dim_dw in
      let d_out_vel_dw = pos_i *. dim_dw +. vel_i *. dre_dw in
      let d_out_pos_dg = pos_i *. dre_dg -. vel_i *. dim_dg in
      let d_out_vel_dg = pos_i *. dim_dg +. vel_i *. dre_dg in
      d_omega := !d_omega +. err_pos *. d_out_pos_dw +. err_vel *. d_out_vel_dw;
      d_gamma := !d_gamma +. err_pos *. d_out_pos_dg +. err_vel *. d_out_vel_dg
    done;
    coupling.omega.(j) <- Float.max 0.01
      (coupling.omega.(j) -. lr *. !d_omega);
    coupling.gamma.(j) <- Float.max 0.01 (Float.min 0.99
      (coupling.gamma.(j) -. lr *. !d_gamma))
  done

let create_coupling ~source_omega ~n_out =
  let n_in = Array.length source_omega in
  {
    omega = Array.init n_out (fun j ->
      let f = Float.of_int j /. Float.of_int (n_out - 1) in
      0.1 +. (Float.pi -. 0.1) *. f);
    gamma = Array.make n_out 0.1;
    source_omega = Array.copy source_omega;
    n_out; n_in;
  }

let create dims =
  let n = Array.length dims in
  (* Frequencies for each layer's oscillators *)
  let freqs_for dim =
    let n_osc = dim / 2 in
    Array.init n_osc (fun i ->
      let f = Float.of_int i /. Float.max 1.0 (Float.of_int (n_osc - 1)) in
      0.1 +. (Float.pi -. 0.1) *. f)
  in
  let couplings = Array.init (n - 1) (fun i ->
    let source = freqs_for dims.(i) in
    create_coupling ~source_omega:source ~n_out:(dims.(i + 1) / 2)
  ) in
  {
    n_layers = n;
    layers = Array.init n (fun i ->
      { activity = Vec.zeros dims.(i);
        error = Vec.zeros dims.(i) });
    couplings;
  }

let compute_errors t =
  let layers = Array.copy t.layers in
  for i = 1 to t.n_layers - 1 do
    let predicted = resonate t.couplings.(i - 1) layers.(i - 1).activity in
    layers.(i) <- { layers.(i) with error = Vec.sub layers.(i).activity predicted }
  done;
  { t with layers }

let settle ~lr t =
  let layers = Array.copy t.layers in
  for i = 1 to t.n_layers - 2 do
    let own_err = t.layers.(i).error in
    let feedback =
      if i < t.n_layers - 1 then
        propagate_error t.couplings.(i) t.layers.(i + 1).error
      else
        Vec.zeros (Vec.dim t.layers.(i).activity)
    in
    layers.(i) <- { layers.(i) with
      activity = Vec.add t.layers.(i).activity
        (Vec.scale lr (Vec.sub feedback own_err)) }
  done;
  { t with layers }

let learn ~lr t =
  for i = 0 to t.n_layers - 2 do
    retune ~lr t.couplings.(i) t.layers.(i).activity t.layers.(i + 1).error
  done;
  t

let clamp state t =
  let layers = Array.copy t.layers in
  layers.(0) <- { layers.(0) with activity = state };
  { t with layers }

let step ~activity_lr ~weight_lr (net : t) (inp : Vec.t) =
  net
  |> clamp inp
  |> compute_errors
  |> settle ~lr:activity_lr
  |> compute_errors
  |> learn ~lr:weight_lr

let top_activity t =
  t.layers.(t.n_layers - 1).activity

let bottom_error t =
  t.layers.(0).error
