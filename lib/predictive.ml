(** Predictive coding through resonance coupling.

    Each layer is an oscillator bank. Predictions between layers
    flow through the transfer function — the same equation that
    encodes tokens. H(ω) connects everything.

    Coupling strength: |H_j(ω_i)| — how strongly oscillator j
    at frequency ω_j responds to oscillator i at frequency ω_i.
    Close frequencies resonate. Far frequencies decouple.

    Learning = retuning. Adjust ω and γ to minimize prediction error.
    Two parameters per oscillator, not n² matrix entries. *)

type layer = {
  activity : Vec.t;
  error : Vec.t;
}

(** Coupling between two layers: oscillator bank with learnable frequencies *)
type coupling = {
  omega : float array;    (** natural frequencies, one per output osc *)
  gamma : float array;    (** damping ratios *)
  source_omega : float array;  (** frequencies of the source layer *)
  n_out : int;
  n_in : int;
}

type t = {
  layers : layer array;
  couplings : coupling array;  (** couplings.(i) connects layer i → i+1 *)
  n_layers : int;
}

(** Transfer function magnitude: |H(ω)| = 1/√((ω₀²-ω²)² + (2γω₀ω)²) *)
let transfer_mag ~omega0 ~gamma ~omega =
  let w0sq = omega0 *. omega0 in
  let wsq = omega *. omega in
  let real = w0sq -. wsq in
  let imag = 2.0 *. gamma *. omega0 *. omega in
  1.0 /. sqrt (real *. real +. imag *. imag +. 1e-8)

(** Derivative of |H| w.r.t. omega0 (for learning) *)
let d_transfer_d_omega0 ~omega0 ~gamma ~omega =
  let w0sq = omega0 *. omega0 in
  let wsq = omega *. omega in
  let real = w0sq -. wsq in
  let imag = 2.0 *. gamma *. omega0 *. omega in
  let denom_sq = real *. real +. imag *. imag +. 1e-8 in
  let d_real = 2.0 *. omega0 in
  let d_imag = 2.0 *. gamma *. omega in
  -. (real *. d_real +. imag *. d_imag) /. (denom_sq *. sqrt denom_sq)

(** Derivative of |H| w.r.t. gamma *)
let d_transfer_d_gamma ~omega0 ~gamma:_ ~omega =
  let w0sq = omega0 *. omega0 in
  let wsq = omega *. omega in
  let real = w0sq -. wsq in
  let imag = 2.0 *. omega0 *. omega in  (* d(2γω₀ω)/dγ = 2ω₀ω *)
  let denom_sq = real *. real +. imag *. imag +. 1e-8 in
  -. (imag *. 2.0 *. omega0 *. omega) /. (denom_sq *. sqrt denom_sq)

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

(** Predict via resonance: output_j = Σ_i input_i * |H_j(ω_i)| *)
let resonate coupling (input : Vec.t) : Vec.t =
  Vec.create coupling.n_out (fun j ->
    let w0 = coupling.omega.(j) in
    let g = coupling.gamma.(j) in
    let acc = ref 0.0 in
    for i = 0 to coupling.n_in - 1 do
      let wi = coupling.source_omega.(i) in
      let h = transfer_mag ~omega0:w0 ~gamma:g ~omega:wi in
      acc := !acc +. input.(i) *. h
    done;
    !acc
  )

(** Propagate error backward through coupling: feedback_i = Σ_j error_j * |H_j(ω_i)| *)
let propagate_error coupling (err : Vec.t) : Vec.t =
  Vec.create coupling.n_in (fun i ->
    let wi = coupling.source_omega.(i) in
    let acc = ref 0.0 in
    for j = 0 to coupling.n_out - 1 do
      let w0 = coupling.omega.(j) in
      let g = coupling.gamma.(j) in
      let h = transfer_mag ~omega0:w0 ~gamma:g ~omega:wi in
      acc := !acc +. err.(j) *. h
    done;
    !acc
  )

(** Learn: retune frequencies and damping to minimize prediction error.
    Δω_j = -lr * Σ_i error_j * input_i * d|H_j|/dω_j(ω_i)
    Δγ_j = -lr * Σ_i error_j * input_i * d|H_j|/dγ_j(ω_i) *)
let retune ~lr coupling (input : Vec.t) (err : Vec.t) =
  for j = 0 to coupling.n_out - 1 do
    let w0 = coupling.omega.(j) in
    let g = coupling.gamma.(j) in
    let ej = err.(j) in
    let d_omega = ref 0.0 in
    let d_gamma = ref 0.0 in
    for i = 0 to coupling.n_in - 1 do
      let wi = coupling.source_omega.(i) in
      let xi = input.(i) in
      d_omega := !d_omega +. ej *. xi *.
        d_transfer_d_omega0 ~omega0:w0 ~gamma:g ~omega:wi;
      d_gamma := !d_gamma +. ej *. xi *.
        d_transfer_d_gamma ~omega0:w0 ~gamma:g ~omega:wi
    done;
    coupling.omega.(j) <- Float.max 0.01
      (coupling.omega.(j) -. lr *. !d_omega);
    coupling.gamma.(j) <- Float.max 0.01 (Float.min 0.99
      (coupling.gamma.(j) -. lr *. !d_gamma))
  done

let create dims =
  let n = Array.length dims in
  (* Source frequencies for the encoding bank *)
  let base_omega = Array.init dims.(0) (fun i ->
    let f = Float.of_int i /. Float.of_int (dims.(0) - 1) in
    0.1 +. (Float.pi -. 0.1) *. f
  ) in
  let couplings = Array.init (n - 1) (fun i ->
    let source = if i = 0 then base_omega
      else Array.init dims.(i) (fun j ->
        let f = Float.of_int j /. Float.of_int (dims.(i) - 1) in
        0.1 +. (Float.pi -. 0.1) *. f)
    in
    create_coupling ~source_omega:source ~n_out:dims.(i + 1)
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
