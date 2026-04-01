(** Predictive coding with dendrite → oscillator → axon coupling.

    Each coupling has three parts (like a neuron):
      B (dendrite) — dense input projection, selects what to receive
      A (cell body) — oscillator kernel, resonance dynamics
      C (axon)     — dense output projection, selects what to broadcast

    Prediction: pred = C × resonate(A, B × activity)
    Learning: B, C update via Hebbian (local), A retunes via dH/dω *)

type layer = {
  activity : Vec.t;
  error : Vec.t;
  precision : Vec.t;   (** Σ⁻¹ per dimension — learned attention *)
  n_osc : int;
}

(** A coupling = dendrite (B) → oscillator (A) → axon (C) *)
type coupling = {
  b : Vec.mat;           (** dendrite: input_dim → n_osc *)
  omega : float array;   (** oscillator frequencies *)
  gamma : float array;   (** oscillator damping *)
  source_omega : float array;
  c : Vec.mat;           (** axon: n_osc → output_dim *)
  n_osc : int;           (** oscillators in this coupling *)
  n_in : int;            (** input state dimension *)
  n_out : int;           (** output state dimension *)
}

type t = {
  layers : layer array;
  couplings : coupling array;
  n_layers : int;
}

(** Complex transfer function components *)
let transfer_re_im ~omega0 ~gamma ~omega =
  let w0sq = omega0 *. omega0 in
  let wsq = omega *. omega in
  let re = w0sq -. wsq in
  let im = 2.0 *. gamma *. omega0 *. omega in
  let d = re *. re +. im *. im +. 1e-8 in
  (re /. d, -. im /. d)

(** Core resonance: complex coupling through H(ω).
    Input and output are n_osc-dimensional [pos; vel] vectors. *)
let resonate ~omega ~gamma ~source_omega n_osc (input : Vec.t) : Vec.t =
  Vec.create (2 * n_osc) (fun k ->
    let j = k mod n_osc in
    let is_vel = k >= n_osc in
    let w0 = omega.(j) in
    let g = gamma.(j) in
    let acc = ref 0.0 in
    for i = 0 to n_osc - 1 do
      let wi = source_omega.(i) in
      let h_re, h_im = transfer_re_im ~omega0:w0 ~gamma:g ~omega:wi in
      let p = input.(i) in
      let v = input.(n_osc + i) in
      if is_vel then acc := !acc +. p *. h_im +. v *. h_re
      else acc := !acc +. p *. h_re -. v *. h_im
    done;
    !acc
  )

(** Full prediction: dendrite → resonate → axon
    pred = C × resonate(A, B × input)
    Set USE_RESONANCE=0 to bypass A and test B→C directly *)
let use_resonance = try int_of_string (Sys.getenv "USE_RESONANCE") <> 0 with _ -> true

let predict coupling input =
  let n = coupling.n_osc in
  let projected = Vec.mat_t_vec coupling.b input in
  let through_a =
    if use_resonance then
      resonate ~omega:coupling.omega ~gamma:coupling.gamma
        ~source_omega:coupling.source_omega n projected
    else projected  (* bypass — just B → C *)
  in
  Vec.mat_t_vec coupling.c through_a

(** Backward: propagate error through axon → resonate → dendrite *)
let propagate_error coupling err =
  let n = coupling.n_osc in
  (* Through axon (C^T) *)
  let err_osc = Vec.mat_vec coupling.c err in
  (* Through resonance (transpose coupling) *)
  let err_resonated = Vec.create (2 * n) (fun k ->
    let i = k mod n in
    let is_vel = k >= n in
    let wi = coupling.source_omega.(i) in
    let acc = ref 0.0 in
    for j = 0 to n - 1 do
      let w0 = coupling.omega.(j) in
      let g = coupling.gamma.(j) in
      let h_re, h_im = transfer_re_im ~omega0:w0 ~gamma:g ~omega:wi in
      let ep = err_osc.(j) in
      let ev = err_osc.(n + j) in
      if is_vel then acc := !acc -. ep *. h_im +. ev *. h_re
      else acc := !acc +. ep *. h_re +. ev *. h_im
    done;
    !acc
  ) in
  (* Through dendrite (B^T) *)
  Vec.mat_vec coupling.b err_resonated

(** Learn: update dendrite (B), axon (C), and retune oscillators (A).
    All Hebbian — activity × error. *)
let learn_coupling ~lr coupling input err =
  let n = coupling.n_osc in
  (* Recompute intermediates for learning *)
  let projected = Vec.mat_t_vec coupling.b input in
  let resonated = resonate
    ~omega:coupling.omega ~gamma:coupling.gamma
    ~source_omega:coupling.source_omega n projected in

  (* Update axon C: Hebbian on resonated × error *)
  Vec.mat_outer_update ~lr:(-.lr) coupling.c resonated err;

  (* Error through axon for oscillator + dendrite learning *)
  let err_osc = Vec.mat_vec coupling.c err in

  (* Retune oscillators: adjust ω, γ *)
  let eps = 1e-4 in
  for j = 0 to n - 1 do
    let w0 = coupling.omega.(j) in
    let g = coupling.gamma.(j) in
    let ep = err_osc.(j) in
    let ev = err_osc.(n + j) in
    let d_omega = ref 0.0 in
    let d_gamma = ref 0.0 in
    for i = 0 to n - 1 do
      let wi = coupling.source_omega.(i) in
      let p = projected.(i) in
      let v = projected.(n + i) in
      let h_re, h_im = transfer_re_im ~omega0:w0 ~gamma:g ~omega:wi in
      let h_re', h_im' = transfer_re_im ~omega0:(w0 +. eps) ~gamma:g ~omega:wi in
      let g_re, g_im = transfer_re_im ~omega0:w0 ~gamma:(g +. eps) ~omega:wi in
      let dre_dw = (h_re' -. h_re) /. eps in
      let dim_dw = (h_im' -. h_im) /. eps in
      let dre_dg = (g_re -. h_re) /. eps in
      let dim_dg = (g_im -. h_im) /. eps in
      d_omega := !d_omega
        +. ep *. (p *. dre_dw -. v *. dim_dw)
        +. ev *. (p *. dim_dw +. v *. dre_dw);
      d_gamma := !d_gamma
        +. ep *. (p *. dre_dg -. v *. dim_dg)
        +. ev *. (p *. dim_dg +. v *. dre_dg)
    done;
    coupling.omega.(j) <- Float.max 0.01 (coupling.omega.(j) -. lr *. !d_omega);
    coupling.gamma.(j) <- Float.max 0.01 (Float.min 0.99 (coupling.gamma.(j) -. lr *. !d_gamma))
  done;

  (* Update dendrite B: Hebbian on input × error_through_resonance *)
  let err_pre = Vec.create (2 * n) (fun k ->
    let i = k mod n in
    let is_vel = k >= n in
    let wi = coupling.source_omega.(i) in
    let acc = ref 0.0 in
    for j = 0 to n - 1 do
      let w0 = coupling.omega.(j) in
      let g = coupling.gamma.(j) in
      let h_re, h_im = transfer_re_im ~omega0:w0 ~gamma:g ~omega:wi in
      let epp = err_osc.(j) in
      let evv = err_osc.(n + j) in
      if is_vel then acc := !acc -. epp *. h_im +. evv *. h_re
      else acc := !acc +. epp *. h_re +. evv *. h_im
    done;
    !acc
  ) in
  Vec.mat_outer_update ~lr:(-.lr) coupling.b input err_pre

let make_freqs n =
  Array.init n (fun i ->
    let f = Float.of_int i /. Float.max 1.0 (Float.of_int (n - 1)) in
    0.1 +. (Float.pi -. 0.1) *. f)

let create_coupling ~n_in ~n_out ~n_osc =
  let scale = sqrt (2.0 /. Float.of_int (n_in + n_osc)) in
  {
    b = Vec.mat_rand ~rows:n_in ~cols:(2 * n_osc) ~scale;
    omega = make_freqs n_osc;
    gamma = Array.make n_osc 0.1;
    source_omega = make_freqs n_osc;
    c = Vec.mat_rand ~rows:(2 * n_osc) ~cols:n_out ~scale;
    n_osc; n_in; n_out;
  }

let create dims =
  let n = Array.length dims in
  let osc_per_layer = Array.map (fun d -> d / 2) dims in
  {
    n_layers = n;
    layers = Array.init n (fun i ->
      { activity = Vec.zeros dims.(i);
        error = Vec.zeros dims.(i);
        precision = Vec.create dims.(i) (fun _ -> 1.0);  (* start uniform *)
        n_osc = osc_per_layer.(i) });
    couplings = Array.init (n - 1) (fun i ->
      let n_osc = Int.min osc_per_layer.(i) osc_per_layer.(i + 1) in
      create_coupling ~n_in:dims.(i) ~n_out:dims.(i + 1) ~n_osc);
  }

(** Precision-weighted errors: ε = Σ⁻¹ × (actual - predicted)
    High precision = this error matters. Low precision = ignore it. *)
let compute_errors t =
  let layers = Array.copy t.layers in
  for i = 1 to t.n_layers - 1 do
    let predicted = predict t.couplings.(i - 1) layers.(i - 1).activity in
    let raw_err = Vec.sub layers.(i).activity predicted in
    let weighted_err = Vec.hadamard layers.(i).precision raw_err in
    layers.(i) <- { layers.(i) with error = weighted_err }
  done;
  { t with layers }

let settle ~lr t =
  let layers = Array.copy t.layers in
  for i = 1 to t.n_layers - 2 do
    let own_err = t.layers.(i).error in
    let feedback =
      if i < t.n_layers - 1 then
        propagate_error t.couplings.(i) t.layers.(i + 1).error
      else Vec.zeros (Vec.dim t.layers.(i).activity)
    in
    let new_act = Vec.add t.layers.(i).activity
      (Vec.scale lr (Vec.sub feedback own_err)) in
    let sharpened = Inhibit.sharpen ~n_osc:t.layers.(i).n_osc new_act in
    layers.(i) <- { layers.(i) with activity = sharpened }
  done;
  { t with layers }

(** Learn couplings AND precision.
    Precision update: ΔΣ⁻¹ ∝ (ε² - 1)
    If error is large → decrease precision (this dimension is noisy)
    If error is small → increase precision (this dimension is reliable) *)
let learn ~lr t =
  for i = 0 to t.n_layers - 2 do
    learn_coupling ~lr t.couplings.(i)
      t.layers.(i).activity t.layers.(i + 1).error
  done;
  (* Update precision at each layer *)
  let layers = Array.copy t.layers in
  for i = 1 to t.n_layers - 1 do
    let err = t.layers.(i).error in
    let prec = t.layers.(i).precision in
    let new_prec = Vec.mapi (fun j p ->
      let e2 = err.(j) *. err.(j) in
      let dp = lr *. 0.1 *. (1.0 -. e2) in  (* push toward ε²=1 *)
      Float.max 0.01 (Float.min 10.0 (p +. dp))
    ) prec in
    layers.(i) <- { layers.(i) with precision = new_prec }
  done;
  { t with layers }

let clamp state t =
  let layers = Array.copy t.layers in
  layers.(0) <- { layers.(0) with activity = state };
  { t with layers }

let step ~activity_lr ~weight_lr input (net : t) =
  net
  |> clamp input
  |> compute_errors
  |> settle ~lr:activity_lr
  |> compute_errors
  |> learn ~lr:weight_lr

let top_activity t = t.layers.(t.n_layers - 1).activity
let bottom_error t = t.layers.(0).error
